#!/usr/bin/env python3
"""
iron_condor_strat.py — 7DTE SPY Iron Condor Strategy (Mode A: combined stop).

STANDALONE: no imports from credit_spread_strat.py, scanner.py, or paper_execution.py.
Runs ALONGSIDE credit_spread_strat.py as a second Railway worker — same Alpaca
paper account, completely separate Postgres tables and state files.
DORMANT until ACTIVE = True.

Structure: sells a $5-wide put spread AND a $5-wide call spread, same expiration,
both legs 20-delta. Backtest reference: backtests_strategies/backtest_iron_condor.py,
Mode A (combined stop — the whole condor closes if EITHER leg hits 200% of its own
credit, or the combined position hits 50% profit).

Alpaca API: raw requests only (no SDK; requirements.txt: requests==2.32.5).
  POST /v2/orders  order_class='mleg'  4 legs in a single order (Alpaca's mleg
  orders support up to 4 legs — confirmed via API reference, maxItems: 4).
  IMPORTANT — limit_price sign convention on multi-leg orders (per Alpaca API
  reference): positive = net DEBIT, negative = net CREDIT. Opening a condor is
  a net credit, so the open order's limit_price must be NEGATIVE. Closing it
  (buying it back) is a net debit, so the close order's limit_price is positive.

State files (all independent of credit_spread_strat.py's and scanner.py's files):
  iron_condor_positions.json — open positions + daily summary flag
  iron_condor_state.json     — weekly_realized_loss, cooldown_active, week_start_date
  iron_condor_trade_log.json — permanent trade history (never truncated)
  iron_condor_log.json       — rolling 500-entry scan log

Cross-strategy coexistence with credit_spread_strat.py:
  Both strategies trade SPY options in the same Alpaca account. Startup
  reconciliation and the daily-summary Alpaca cross-check both need to count
  ONLY this strategy's own legs, so both sides read the other's Postgres table
  (credit_spread_positions) directly to exclude its legs from this strategy's
  "untracked leg" detection — see _reconcile_on_startup(). credit_spread_strat.py
  was patched symmetrically to exclude iron_condor_positions' legs from its own
  reconcile — without that, both strategies would immediately false-alarm on
  each other's positions and block new entries.

Deploy as a third Railway worker (separate service — do not add to the existing
Procfile/railway.json; configure directly in the new service's settings):
  Start command:  python3 iron_condor_strat.py
"""

import json
import math
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
import schedule
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm

try:
    import psycopg2
except ImportError:
    psycopg2 = None

load_dotenv()

# ── ACTIVATION FLAG ────────────────────────────────────────────────────────────
ACTIVE = True   # Set to True to enable live order placement. Dormant by default.

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK_URL')
ALPACA_KEY      = os.getenv('ALPACA_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET')
PAPER_BASE_URL  = 'https://paper-api.alpaca.markets'
DATA_URL        = 'https://data.alpaca.markets'
DATABASE_URL    = os.getenv('DATABASE_URL')
ET              = pytz.timezone('US/Eastern')
_DIR            = os.path.dirname(os.path.abspath(__file__))

DISCORD_PREFIX  = '[CONDOR] '

# Strategy parameters — matches backtests_strategies/backtest_iron_condor.py Mode A
TARGET_DTE        = 7
DTE_MIN           = 6
DTE_MAX           = 8
TARGET_DELTA      = 0.20
DELTA_TOLERANCE   = 0.05     # reject if no strike within 0.05 of target delta
SPREAD_WIDTH      = 5.0
MIN_CREDIT_LEG    = 0.25     # minimum credit per individual leg (put AND call)
MAX_POSITIONS     = 2
TICKERS           = ['SPY']
PROFIT_TARGET_PCT = 0.50     # of COMBINED (put + call) credit
STOP_LOSS_PCT     = 2.00     # of EACH leg's OWN credit, checked independently
ORDER_FILL_TIMEOUT = 300
WEEKLY_LOSS_LIMIT = 1_000.0  # isolated from credit_spread_strat.py's own weekly tracking
RISK_FREE_RATE    = 0.045
VIX_IVR_WINDOW    = 252
MIN_IVR           = 30.0
MAX_VIX           = 35.0
SMA_PERIOD        = 20

# ── MACRO EVENT CALENDAR ───────────────────────────────────────────────────────
# Identical to credit_spread_strat.py's calendar — kept in sync manually since
# both files are standalone (no shared imports). Update each January using:
#   FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI:  bls.gov/schedule/news_release/cpi.htm
#   GDP:  bea.gov/news/schedule
# Jobs Report (NFP) is always the first Friday of each month — computed in code.

FOMC_DAYS = {
    # 2026 — both Day 1 and Day 2 of each meeting
    '2026-01-28', '2026-01-29',
    '2026-03-18', '2026-03-19',
    '2026-04-29', '2026-04-30',
    '2026-06-10', '2026-06-11',
    '2026-07-29', '2026-07-30',
    '2026-09-16', '2026-09-17',
    '2026-11-04', '2026-11-05',
    '2026-12-09', '2026-12-10',
}

CPI_DAYS = {
    # 2026 — BLS CPI release dates (prior month's data, ~12 days after month end)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-13', '2026-06-11', '2026-07-14', '2026-08-12',
    '2026-09-09', '2026-10-14', '2026-11-12', '2026-12-10',
}

GDP_DAYS = {
    # 2026 — BEA advance GDP estimates (~30 days after each quarter end)
    '2026-01-29',   # Q4 2025
    '2026-04-29',   # Q1 2026
    '2026-07-30',   # Q2 2026
    '2026-10-29',   # Q3 2026
}

# Timing — same cadence as credit_spread_strat.py
ENTRY_HOUR_START, ENTRY_MIN_START =  9, 45
ENTRY_HOUR_END,   ENTRY_MIN_END   = 15, 30
SUMMARY_HOUR,     SUMMARY_MIN     = 15, 35

# Files
POSITIONS_FILE = os.path.join(_DIR, 'iron_condor_positions.json')
WEEKLY_FILE    = os.path.join(_DIR, 'iron_condor_state.json')
TRADE_LOG_FILE = os.path.join(_DIR, 'iron_condor_trade_log.json')
SCAN_LOG_FILE  = os.path.join(_DIR, 'iron_condor_log.json')
LOG_MAX        = 500


# ── ALPACA HEADERS ─────────────────────────────────────────────────────────────

def _headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type':        'application/json',
    }


def _data_headers():
    return {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
    }


def _alpaca_get(url, **kwargs):
    """Alpaca GET with 429 backoff retry (max 3 attempts). Returns Response or None."""
    kwargs.setdefault('timeout', 10)
    for attempt in range(3):
        try:
            r = requests.get(url, **kwargs)
            if r.status_code == 429:
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Alpaca 429, retrying in {wait:.1f}s …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            print(f'Alpaca GET error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                time.sleep(1)
    return None


# ── SCHEDULER TIMING ──────────────────────────────────────────────────────────

_last_scan_start:    datetime = None
_last_scan_duration: float    = 0.0

# ── MONITOR RELIABILITY STATE ──────────────────────────────────────────────────
# Keyed generically by a tuple of the 4 leg symbols — works the same way as
# credit_spread_strat.py's 2-symbol key, just longer.
_last_known_cost:     dict = {}  # leg_key → {'put_cost': float, 'call_cost': float, 'ts': datetime}
_monitor_consec_fail: dict = {}  # leg_key → consecutive failure count
_monitor_alert_sent:  set  = set()

STALE_COST_MAX_MINUTES  = 30
MONITOR_ALERT_THRESHOLD = 3


# ── DATABASE ───────────────────────────────────────────────────────────────────

_DB = None  # module-level psycopg2 connection


def _get_db():
    """Return a live psycopg2 connection, or None if DATABASE_URL is not set."""
    global _DB
    if not DATABASE_URL or psycopg2 is None:
        return None
    try:
        if _DB is None or _DB.closed:
            _DB = psycopg2.connect(DATABASE_URL)
        return _DB
    except Exception as e:
        print(f'[db] Connection failed: {e}')
        return None


def _init_db():
    """Create tables if they don't exist. Logs connection on success."""
    conn = _get_db()
    if conn is None:
        if DATABASE_URL and psycopg2 is None:
            print('[db] WARNING — DATABASE_URL set but psycopg2 not installed; falling back to JSON')
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iron_condor_positions (
                id                      SERIAL PRIMARY KEY,
                short_put_symbol        TEXT NOT NULL,
                long_put_symbol         TEXT NOT NULL,
                short_call_symbol       TEXT NOT NULL,
                long_call_symbol        TEXT NOT NULL,
                short_put_strike        DOUBLE PRECISION,
                long_put_strike         DOUBLE PRECISION,
                short_call_strike       DOUBLE PRECISION,
                long_call_strike        DOUBLE PRECISION,
                expiration              TEXT,
                put_credit              DOUBLE PRECISION,
                call_credit             DOUBLE PRECISION,
                total_credit            DOUBLE PRECISION,
                max_risk                DOUBLE PRECISION,
                put_breakeven           DOUBLE PRECISION,
                call_breakeven          DOUBLE PRECISION,
                profit_target           DOUBLE PRECISION,
                put_stop_cost           DOUBLE PRECISION,
                call_stop_cost          DOUBLE PRECISION,
                open_time               TEXT,
                entry_order_id          TEXT,
                short_put_delta         DOUBLE PRECISION,
                short_call_delta        DOUBLE PRECISION,
                spy_entry_px            DOUBLE PRECISION,
                reconciled              BOOLEAN DEFAULT FALSE,
                note                    TEXT,
                pending_close_order_id  TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iron_condor_state (
                id                   INTEGER PRIMARY KEY DEFAULT 1,
                weekly_realized_loss DOUBLE PRECISION DEFAULT 0.0,
                cooldown_active      BOOLEAN DEFAULT FALSE,
                week_start_date      TEXT,
                daily_summary_sent   TEXT,
                morning_vitals_sent  TEXT
            )
        """)
        conn.commit()
        print('[db] Connected to PostgreSQL — persistent storage active')

        # One-time migration: seed from JSON files if tables are empty on first boot
        cur.execute('SELECT COUNT(*) FROM iron_condor_positions')
        if cur.fetchone()[0] == 0 and os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE) as f:
                    j = json.load(f)
                if isinstance(j, dict) and j.get('positions'):
                    _db_save_positions(j)
                    print(f'[db] Migrated {len(j["positions"])} position(s) from JSON')
            except Exception as me:
                print(f'[db] JSON positions migration failed: {me}')

        cur.execute('SELECT COUNT(*) FROM iron_condor_state')
        if cur.fetchone()[0] == 0 and os.path.exists(WEEKLY_FILE):
            try:
                with open(WEEKLY_FILE) as f:
                    j = json.load(f)
                if isinstance(j, dict) and j.get('week_start_date'):
                    _db_save_weekly(j)
                    print('[db] Migrated weekly state from JSON')
            except Exception as me:
                print(f'[db] JSON weekly migration failed: {me}')

        # Ensure row id=1 always exists so _db_load_weekly()'s SELECT can tell
        # "no row yet" apart from "connection failed" — without this, a fresh
        # table with zero rows makes _db_load_weekly() return None either way,
        # silently falling back to ephemeral JSON until a week rollover or a
        # trade close happens to call _db_save_weekly() for the first time.
        cur.execute("""
            INSERT INTO iron_condor_state (id, weekly_realized_loss, cooldown_active, week_start_date)
            VALUES (1, 0.0, FALSE, %s)
            ON CONFLICT (id) DO NOTHING
        """, (_this_monday(),))
        conn.commit()

    except Exception as e:
        print(f'[db] _init_db failed: {e}')


_POSITION_COLUMNS = (
    'short_put_symbol', 'long_put_symbol', 'short_call_symbol', 'long_call_symbol',
    'short_put_strike', 'long_put_strike', 'short_call_strike', 'long_call_strike',
    'expiration', 'put_credit', 'call_credit', 'total_credit', 'max_risk',
    'put_breakeven', 'call_breakeven', 'profit_target', 'put_stop_cost', 'call_stop_cost',
    'open_time', 'entry_order_id', 'short_put_delta', 'short_call_delta', 'spy_entry_px',
)


def _db_load_positions():
    """Load positions list and daily_summary_sent from PostgreSQL. Returns None on failure."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT {', '.join(_POSITION_COLUMNS)}, reconciled, note, pending_close_order_id
            FROM iron_condor_positions ORDER BY id
        """)
        positions = []
        for row in cur.fetchall():
            pos = dict(zip(_POSITION_COLUMNS, row[:len(_POSITION_COLUMNS)]))
            if row[len(_POSITION_COLUMNS)]:
                pos['reconciled'] = True
            if row[len(_POSITION_COLUMNS) + 1]:
                pos['note'] = row[len(_POSITION_COLUMNS) + 1]
            if row[len(_POSITION_COLUMNS) + 2]:
                pos['pending_close_order_id'] = row[len(_POSITION_COLUMNS) + 2]
            positions.append(pos)

        cur.execute("""
            SELECT daily_summary_sent, morning_vitals_sent
            FROM iron_condor_state WHERE id = 1
        """)
        row = cur.fetchone()
        return {
            'positions':           positions,
            'daily_summary_sent':  row[0] if row else None,
            'morning_vitals_sent': row[1] if row else None,
        }
    except Exception as e:
        print(f'[db] _db_load_positions failed: {e}')
        global _DB
        _DB = None
        return None


def _db_save_positions(ps):
    """Replace all rows in iron_condor_positions and update daily_summary_sent."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute('DELETE FROM iron_condor_positions')
        cols = _POSITION_COLUMNS + ('reconciled', 'note', 'pending_close_order_id')
        placeholders = ','.join(['%s'] * len(cols))
        for pos in ps.get('positions', []):
            values = tuple(pos.get(c) for c in _POSITION_COLUMNS) + (
                bool(pos.get('reconciled', False)),
                pos.get('note'),
                pos.get('pending_close_order_id'),
            )
            cur.execute(
                f"INSERT INTO iron_condor_positions ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
        cur.execute("""
            INSERT INTO iron_condor_state (id, daily_summary_sent, morning_vitals_sent)
            VALUES (1, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                daily_summary_sent  = EXCLUDED.daily_summary_sent,
                morning_vitals_sent = EXCLUDED.morning_vitals_sent
        """, (ps.get('daily_summary_sent'), ps.get('morning_vitals_sent')))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_save_positions failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return False


def _db_read_summary_flags():
    conn = _get_db()
    if conn is None:
        return None, None
    try:
        cur = conn.cursor()
        cur.execute(
            'SELECT daily_summary_sent, morning_vitals_sent '
            'FROM iron_condor_state WHERE id = 1'
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)
    except Exception as e:
        print(f'[db] _db_read_summary_flags failed: {e}')
        return None, None


def _db_write_summary_flags(daily_str, vitals_str):
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE iron_condor_state
               SET daily_summary_sent  = %s,
                   morning_vitals_sent = %s
             WHERE id = 1
            """,
            (daily_str, vitals_str),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_write_summary_flags failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return False


def _db_load_weekly():
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT weekly_realized_loss, cooldown_active, week_start_date
            FROM iron_condor_state WHERE id = 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        return {
            'weekly_realized_loss': float(row[0] or 0.0),
            'cooldown_active':      bool(row[1]),
            'week_start_date':      row[2],
        }
    except Exception as e:
        print(f'[db] _db_load_weekly failed: {e}')
        global _DB
        _DB = None
        return None


def _db_save_weekly(w):
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO iron_condor_state
                (id, weekly_realized_loss, cooldown_active, week_start_date)
            VALUES (1, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                weekly_realized_loss = EXCLUDED.weekly_realized_loss,
                cooldown_active      = EXCLUDED.cooldown_active,
                week_start_date      = EXCLUDED.week_start_date
        """, (
            w.get('weekly_realized_loss', 0.0),
            bool(w.get('cooldown_active', False)),
            w.get('week_start_date'),
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_save_weekly failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return False


def _other_strategy_leg_symbols():
    """
    Read credit_spread_strat.py's currently-tracked leg symbols directly from its
    Postgres table. Both strategies share one Alpaca account and one DATABASE_URL;
    without this, each strategy's startup reconciliation and daily-summary Alpaca
    cross-check would count the OTHER strategy's real, legitimate option legs as
    "untracked" and fire false-positive mismatch alerts / add blocker placeholders —
    exactly the false-positive-mismatch bug class credit_spread_strat.py's own
    reconcile fix (2026-08-16) was built to eliminate, just from a different cause.
    Returns a set of symbols. Empty set (not None) on any failure — a failed
    lookup should never itself trigger a false mismatch alert.
    """
    conn = _get_db()
    if conn is None:
        return set()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT to_regclass('public.credit_spread_positions')
        """)
        if cur.fetchone()[0] is None:
            return set()   # table doesn't exist yet — sibling strategy never booted
        cur.execute('SELECT short_symbol, long_symbol FROM credit_spread_positions')
        symbols = set()
        for short_sym, long_sym in cur.fetchall():
            if short_sym:
                symbols.add(short_sym)
            if long_sym:
                symbols.add(long_sym)
        return symbols
    except Exception as e:
        print(f'[db] _other_strategy_leg_symbols failed: {e}')
        return set()


# ── MARKET HOURS ───────────────────────────────────────────────────────────────

def is_market_hours():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def _in_entry_window():
    now  = datetime.now(ET)
    mins = now.hour * 60 + now.minute
    return (ENTRY_HOUR_START * 60 + ENTRY_MIN_START) <= mins <= (ENTRY_HOUR_END * 60 + ENTRY_MIN_END)


def _is_summary_time():
    now = datetime.now(ET)
    return now.hour == SUMMARY_HOUR and now.minute >= SUMMARY_MIN


# ── DISCORD ────────────────────────────────────────────────────────────────────

def _discord(msg):
    """Rate-limit-aware Discord post, prefixed to distinguish from credit_spread_strat.py's
    unprefixed alerts (that file posts no prefix — confirmed by reading its _discord() calls).
    ACTIVE-gated."""
    if not ACTIVE:
        return
    full_msg = f'{DISCORD_PREFIX}{msg}'
    if not DISCORD_WEBHOOK:
        print(f'[Discord] {full_msg}')
        return
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK, json={'content': full_msg}, timeout=10)
            if r.status_code == 429:
                wait = float((r.json() if r.content else {}).get('retry_after', 1.0))
                print(f'Discord 429, retrying in {wait:.1f}s …')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return
        except Exception as e:
            print(f'Discord error (attempt {attempt + 1}/3): {e}')
            if attempt < 2:
                time.sleep(1)


# ── LOGGING ────────────────────────────────────────────────────────────────────

def _log(entry):
    """Rolling 500-entry scan log. Always runs regardless of ACTIVE."""
    try:
        try:
            with open(SCAN_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        if len(log) > LOG_MAX:
            log = log[-LOG_MAX:]
        with open(SCAN_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log] write failed: {e}')


def _log_trade(entry):
    """Permanent trade history — never truncated. Same caveat as credit_spread_strat.py:
    this file lives on Railway's ephemeral container filesystem, NOT synced to Postgres.
    It resets on redeploy. Alpaca's own order/activity history is the durable source of
    truth for closed trades, not this file."""
    try:
        try:
            with open(TRADE_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log_trade] write failed: {e}')


# ── POSITIONS / WEEKLY STATE ────────────────────────────────────────────────────

def _load_positions():
    if DATABASE_URL:
        result = _db_load_positions()
        if result is not None:
            return result
        print('[db] _load_positions: DB unavailable, falling back to JSON')
    try:
        with open(POSITIONS_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'positions' in d:
            return d
    except Exception:
        pass
    return {'positions': [], 'daily_summary_sent': None}


def _save_positions(ps):
    if DATABASE_URL:
        if _db_save_positions(ps):
            return
        print('[db] _save_positions: DB write failed, falling back to JSON')
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(ps, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_save_positions] failed: {e}')


def _this_monday():
    today = datetime.now(ET).date()
    return (today - timedelta(days=today.weekday())).isoformat()


def _empty_weekly():
    return {
        'weekly_realized_loss': 0.0,
        'cooldown_active':      False,
        'week_start_date':      _this_monday(),
    }


def _load_weekly():
    if DATABASE_URL:
        result = _db_load_weekly()
        if result is not None:
            return result
        print('[db] _load_weekly: DB unavailable, falling back to JSON')
    try:
        with open(WEEKLY_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'week_start_date' in d:
            return d
    except Exception:
        pass
    return _empty_weekly()


def _save_weekly(w):
    if DATABASE_URL:
        if _db_save_weekly(w):
            return
        print('[db] _save_weekly: DB write failed, falling back to JSON')
    try:
        with open(WEEKLY_FILE, 'w') as f:
            json.dump(w, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_save_weekly] failed: {e}')


def _reset_weekly_if_needed(w):
    monday = _this_monday()
    if w.get('week_start_date') != monday:
        print(f'  [weekly reset] New week {monday} — loss and cooldown cleared')
        w = _empty_weekly()
        _save_weekly(w)
    return w


# ── FILE INIT ──────────────────────────────────────────────────────────────────

def _init_files():
    defaults = [
        (POSITIONS_FILE, {'positions': [], 'daily_summary_sent': None}),
        (WEEKLY_FILE,    _empty_weekly()),
        (TRADE_LOG_FILE, []),
        (SCAN_LOG_FILE,  []),
    ]
    for path, empty in defaults:
        if DATABASE_URL and path in (POSITIONS_FILE, WEEKLY_FILE):
            continue
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(empty, f, indent=2)
            print(f'[init] Created {os.path.basename(path)}')


# ── STARTUP RECONCILIATION ─────────────────────────────────────────────────────

def _reconcile_on_startup():
    """
    Load state files, cross-check against Alpaca open options positions, and
    return (pos_state, weekly). Mirrors credit_spread_strat.py's reconcile
    (pending-close-first, then leg-count mismatch), with one addition: legs
    belonging to credit_spread_strat.py (read from its own Postgres table) are
    excluded before computing "orphaned" legs, since both strategies share
    this Alpaca account.
    """
    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)
    now_str   = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')

    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is None:
            raise RuntimeError('positions fetch returned None after retries')
        today_ymd = datetime.now(ET).strftime('%y%m%d')
        other_legs = _other_strategy_leg_symbols()
        option_legs = [
            p for p in r.json()
            if p.get('asset_class') == 'us_option'
            and any(str(p.get('symbol', '')).startswith(t) for t in TICKERS)
            and str(p.get('symbol', ''))[3:9] != today_ymd   # ignore 0DTE legs
            and p.get('symbol', '') not in other_legs        # ignore credit_spread_strat.py's legs
        ]

        # Resolve positions whose close order filled while the process was down —
        # must happen before expected_legs is computed, same ordering rationale as
        # credit_spread_strat.py's 2026-08-16 fix.
        resolved_pending = []
        for pos in pos_state.get('positions', []):
            pending_id = pos.get('pending_close_order_id')
            if not pending_id:
                continue
            order = _get_order(pending_id)
            if order and order.get('status') == 'filled':
                put_cost, call_cost = _leg_close_costs_from_order(order, pos)
                label = _pos_label(pos)
                print(f'[reconcile] Pending close {pending_id[:8]}… confirmed filled — recording exit')
                _log({'timestamp': now_str, 'event': 'CLOSE_CONFIRMED_ON_RECONCILE',
                      'label': label, 'order_id': pending_id,
                      'put_close_cost': put_cost, 'call_close_cost': call_cost})
                # pending_close_order_id is only ever set from a profit-target limit
                # close (stop-loss/expiration closes are market orders, recorded
                # synchronously) — same invariant as credit_spread_strat.py.
                _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET',
                             put_cost, call_cost, now_str)
                resolved_pending.append(pos)
        for pos in resolved_pending:
            try:
                pos_state['positions'].remove(pos)
            except ValueError:
                pass
        if resolved_pending:
            _save_positions(pos_state)
            _save_weekly(weekly)

        expected_legs = len(pos_state.get('positions', [])) * 4
        actual_legs   = len(option_legs)

        if actual_legs == expected_legs:
            print(f'[reconcile] Alpaca options OK — '
                  f'{actual_legs} leg(s) / {len(pos_state["positions"])} condor(s)')
        else:
            orphaned = actual_legs - expected_legs
            direction = f'{orphaned} untracked leg(s) — new entries blocked' if orphaned > 0 \
                        else f'{-orphaned} extra state entry(ies) — stale entries possible'
            msg = (
                f'⚠️ OPTIONS MISMATCH on startup | '
                f'Alpaca: {actual_legs} option leg(s) (excluding credit_spread_strat.py legs), '
                f'state file expects {expected_legs} ({len(pos_state["positions"])} condor(s)). '
                f'{direction}. Manual review required.'
            )
            print(f'[reconcile] {msg}')
            _log({'timestamp': now_str, 'event': 'RECONCILE_OPTIONS_MISMATCH',
                  'alpaca_legs': actual_legs, 'expected_legs': expected_legs,
                  'alpaca_symbols': [p['symbol'] for p in option_legs]})
            _discord(msg)

            for _ in range(max(0, orphaned // 4)):
                pos_state['positions'].append(_unknown_placeholder(now_str))
            if orphaned > 0:
                _save_positions(pos_state)

    except Exception as e:
        print(f'[reconcile] Alpaca positions check failed (startup continues): {e}')
        _log({'timestamp': now_str, 'event': 'RECONCILE_API_ERROR', 'error': str(e)})

    positions = pos_state.get('positions', [])
    tracked   = [p for p in positions if not p.get('reconciled')]
    if positions:
        labels = [_pos_label(p) + (' [UNTRACKED]' if p.get('reconciled') else '') for p in positions]
        print(f'[startup] Resumed {len(tracked)} tracked + '
              f'{len(positions) - len(tracked)} untracked condor(s):')
        for lbl in labels:
            print(f'  {lbl}')
        _log({'timestamp': now_str, 'event': 'STARTUP_RESUME',
              'positions': len(positions), 'labels': labels})
    else:
        print('[startup] No open positions found — starting clean.')
        _log({'timestamp': now_str, 'event': 'STARTUP_CLEAN'})

    print(f'[startup] Weekly loss: ${weekly["weekly_realized_loss"]:.2f}  '
          f'cooldown: {weekly["cooldown_active"]}  '
          f'week_start: {weekly["week_start_date"]}')

    return pos_state, weekly


def _unknown_placeholder(now_str):
    return {
        'short_put_symbol': 'UNKNOWN', 'long_put_symbol': 'UNKNOWN',
        'short_call_symbol': 'UNKNOWN', 'long_call_symbol': 'UNKNOWN',
        'short_put_strike': 0.0, 'long_put_strike': 0.0,
        'short_call_strike': 0.0, 'long_call_strike': 0.0,
        'expiration': '2099-01-01',   # far future — never triggers expiry close
        'put_credit': 0.0, 'call_credit': 0.0, 'total_credit': 0.0,
        'max_risk': 0.0, 'put_breakeven': 0.0, 'call_breakeven': 0.0,
        'profit_target': 0.0, 'put_stop_cost': 999.0, 'call_stop_cost': 999.0,
        'open_time': now_str, 'entry_order_id': None,
        'short_put_delta': None, 'short_call_delta': None, 'spy_entry_px': None,
        'reconciled': True,
        'note': 'Untracked Alpaca position — manual close required',
    }


def _pos_label(pos):
    return (f'{pos.get("short_put_strike", 0):.0f}/{pos.get("long_put_strike", 0):.0f}P '
            f'{pos.get("short_call_strike", 0):.0f}/{pos.get("long_call_strike", 0):.0f}C '
            f'exp={pos.get("expiration", "?")}  credit=${pos.get("total_credit", 0):.2f}')


# ── MARKET DATA ────────────────────────────────────────────────────────────────

def _flatten_columns(df):
    if hasattr(df.columns, 'levels'):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass
    return df


def _above_sma20(ticker):
    try:
        df = yf.download(ticker, period='40d', interval='1d',
                         progress=False, auto_adjust=True, timeout=10)
        if df.empty:
            return None, None, None
        df     = _flatten_columns(df)
        closes = df['Close'].dropna().values
        if len(closes) < SMA_PERIOD:
            return None, None, None
        sma     = float(closes[-SMA_PERIOD:].mean())
        current = float(closes[-1])
        return current > sma, current, sma
    except Exception as e:
        print(f'  [sma20] {e}')
        return None, None, None


def _vix_ivrank():
    """VIX percentile fallback. Return (ivr_pct: float|None, vix: float|None)."""
    try:
        df = yf.download('^VIX', period=f'{VIX_IVR_WINDOW + 60}d',
                         interval='1d', progress=False, auto_adjust=False, timeout=10)
        if df.empty:
            return None, None
        df          = _flatten_columns(df)
        closes      = df['Close'].dropna().values
        if len(closes) < 2:
            return None, None
        window      = closes[-VIX_IVR_WINDOW:] if len(closes) >= VIX_IVR_WINDOW else closes
        current_vix = float(closes[-1])
        ivr         = float((window < current_vix).sum()) / len(window) * 100
        return round(ivr, 1), round(current_vix, 2)
    except Exception as e:
        print(f'  [vix_ivrank] {e}')
        return None, None


def _bs_iv_solve(S, K, T, mkt_price, is_call):
    """Bisection IV solver from an observed option price. Returns decimal IV."""
    if T <= 0 or mkt_price <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 0.001, 5.0
    r = RISK_FREE_RATE
    for _ in range(80):
        mid = (lo + hi) / 2
        d1  = (math.log(S / K) + (r + 0.5 * mid ** 2) * T) / (mid * math.sqrt(T))
        d2  = d1 - mid * math.sqrt(T)
        if is_call:
            p = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            p = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        if p < mkt_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


def _spy_ivrank():
    """
    Return (ivr_pct: float|None, current_iv_pct: float|None).
    IVR = percentile of current SPY ATM IV within 252-day Parkinson vol range.
    Identical methodology to credit_spread_strat.py's _spy_ivrank — same live
    SPY ATM put IV (via Alpaca), same Parkinson realized-vol window as the
    denominator. Falls back to _vix_ivrank() if the SPY options fetch fails.
    """
    current_iv = None
    try:
        spy_df = yf.download('SPY', period='2d', interval='1m',
                              progress=False, auto_adjust=True, timeout=10)
        if spy_df.empty:
            raise ValueError('SPY price unavailable')
        spy_df = _flatten_columns(spy_df)
        S = float(spy_df['Close'].dropna().iloc[-1])

        today = date.today()
        exps  = yf.Ticker('SPY').options
        if not exps:
            raise ValueError('SPY options list unavailable')

        best_exp, best_diff = None, float('inf')
        for e in exps:
            dte = (date.fromisoformat(e) - today).days
            if 5 <= dte <= 14:
                diff = abs(dte - 7)
                if diff < best_diff:
                    best_exp, best_diff = e, diff
        if best_exp is None:
            raise ValueError('No 5-14 DTE expiry available')

        for lo_pct, hi_pct in [(0.99, 1.01), (0.98, 1.02)]:
            r = _alpaca_get(
                f'{PAPER_BASE_URL}/v2/options/contracts',
                headers=_headers(),
                params={
                    'underlying_symbol': 'SPY',
                    'expiration_date':   best_exp,
                    'type':              'put',
                    'strike_price_gte':  str(int(S * lo_pct)),
                    'strike_price_lte':  str(int(S * hi_pct)),
                    'limit': 10,
                    'status': 'active',
                }
            )
            contracts = r.json().get('option_contracts', []) if r else []
            if contracts:
                break
        if not contracts:
            raise ValueError('No ATM contracts found')

        atm = min(contracts, key=lambda c: abs(float(c.get('strike_price', 0)) - S))
        sym = atm['symbol']

        rs = _alpaca_get(
            f'{DATA_URL}/v1beta1/options/snapshots',
            headers=_data_headers(),
            params={'symbols': sym, 'feed': 'indicative'}
        )
        if rs is None:
            raise ValueError('Snapshot endpoint unavailable')
        snap = rs.json().get('snapshots', {}).get(sym)
        if not snap:
            raise ValueError(f'No snapshot for {sym}')

        iv_raw = snap.get('impliedVolatility')
        if iv_raw is not None:
            current_iv = float(iv_raw)
        else:
            q   = snap.get('latestQuote', {})
            bid = float(q.get('bp') or 0)
            ask = float(q.get('ap') or 0)
            if bid + ask > 0:
                K  = float(atm.get('strike_price', S))
                T  = (date.fromisoformat(best_exp) - today).days / 365.0
                current_iv = _bs_iv_solve(S, K, T, (bid + ask) / 2, is_call=False)
            if current_iv is None:
                raise ValueError('IV not available in snapshot; back-solve also failed')

    except Exception as e:
        print(f'  [ivrank] SPY IV fetch failed — using VIX proxy fallback: {e}')
        return _vix_ivrank()

    current_iv_pct = current_iv * 100

    try:
        df = yf.download('SPY', period=f'{VIX_IVR_WINDOW + 60}d',
                         interval='1d', progress=False, auto_adjust=True, timeout=10)
        if df.empty or len(df) < 20:
            raise ValueError('SPY OHLC data insufficient')
        df     = _flatten_columns(df)
        highs  = df['High'].dropna().values.astype(float)
        lows   = df['Low'].dropna().values.astype(float)
        n      = min(len(highs), len(lows))
        log_hl    = np.log(highs[-n:] / lows[-n:])
        park_ann  = np.sqrt(log_hl ** 2 / (4 * math.log(2))) * math.sqrt(252) * 100
        window    = park_ann[-VIX_IVR_WINDOW:] if len(park_ann) >= VIX_IVR_WINDOW else park_ann
        iv_low    = float(window.min())
        iv_high   = float(window.max())

        if iv_high <= iv_low:
            raise ValueError('Parkinson IV range is zero')

        ivr = (current_iv_pct - iv_low) / (iv_high - iv_low) * 100
        ivr = max(0.0, min(100.0, ivr))
        return round(ivr, 1), round(current_iv_pct, 2)

    except Exception as e:
        print(f'  [ivrank] Parkinson window failed ({e}) — using VIX proxy fallback')
        return _vix_ivrank()


# ── OPTIONS CHAIN ──────────────────────────────────────────────────────────────

def _find_target_expiration(ticker):
    """Return expiry closest to TARGET_DTE. Widens to 5–10 DTE if needed."""
    try:
        expirations = yf.Ticker(ticker).options
        if not expirations:
            return None
        today     = date.today()
        best      = None
        best_diff = float('inf')
        for exp_str in expirations:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                diff = abs(dte - TARGET_DTE)
                if diff < best_diff:
                    best, best_diff = exp, diff
        if best:
            return best
        for exp_str in expirations:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if 5 <= dte <= 10:
                diff = abs(dte - TARGET_DTE)
                if diff < best_diff:
                    best, best_diff = exp, diff
        return best
    except Exception as e:
        print(f'  [find_expiry] {e}')
        return None


def _fetch_options_chain_side(ticker, expiry, opt_type, now_str):
    """
    Fetch put OR call chain for ticker/expiry (opt_type: 'put' | 'call').
    Same two-step pattern as credit_spread_strat.py: contracts (trading API),
    then snapshots (market data API) for bid/ask/mid/delta/iv.
    Returns dict {symbol: {strike, bid, ask, mid, delta, iv, type}} or None.
    """
    try:
        r = _alpaca_get(
            f'{PAPER_BASE_URL}/v2/options/contracts',
            headers=_headers(),
            params={
                'underlying_symbols': ticker,
                'type':               opt_type,
                'expiration_date':    expiry.isoformat(),
                'limit':              200,
            },
            timeout=15,
        )
        if r is None:
            reason = f'{opt_type} contracts fetch returned None after retries'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR',
                  'opt_type': opt_type, 'expiry': str(expiry), 'reason': reason})
            return None
        contracts = r.json().get('option_contracts', [])
        if not contracts:
            reason = f'no {opt_type} contracts returned by Alpaca for expiry {expiry}'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'opt_type': opt_type,
                  'expiry': str(expiry), 'reason': reason})
            return None

        symbols    = [c['symbol'] for c in contracts]
        strike_map = {c['symbol']: float(c['strike_price']) for c in contracts}

        chain = {}
        for batch_start in range(0, len(symbols), 100):
            batch = symbols[batch_start:batch_start + 100]
            try:
                rs = _alpaca_get(
                    f'{DATA_URL}/v1beta1/options/snapshots',
                    headers=_data_headers(),
                    params={'symbols': ','.join(batch), 'feed': 'indicative'},
                    timeout=15,
                )
                if rs is None:
                    reason = f'{opt_type} snapshots batch failed after retries (offset {batch_start})'
                    print(f'  [chain] {reason}')
                    _log({'timestamp': now_str, 'event': 'CHAIN_BATCH_ERROR', 'reason': reason})
                    continue
                for sym, snap in rs.json().get('snapshots', {}).items():
                    q   = snap.get('latestQuote', {})
                    bid = float(q.get('bp') or 0)
                    ask = float(q.get('ap') or 0)
                    mid = round((bid + ask) / 2, 4) if (bid + ask) > 0 else 0.0
                    g   = snap.get('greeks') or {}
                    chain[sym] = {
                        'strike': strike_map.get(sym, 0.0),
                        'bid':    bid,
                        'ask':    ask,
                        'mid':    mid,
                        'delta':  float(g['delta']) if g.get('delta') is not None else None,
                        'iv':     snap.get('impliedVolatility'),
                        'type':   opt_type,
                    }
            except Exception as e:
                reason = f'{opt_type} snapshot batch error: {type(e).__name__}: {e}'
                print(f'  [chain] {reason}')
                _log({'timestamp': now_str, 'event': 'CHAIN_BATCH_ERROR', 'reason': reason})

        if not chain:
            reason = f'all {opt_type} snapshot batches returned empty'
            print(f'  [chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'opt_type': opt_type,
                  'expiry': str(expiry), 'reason': reason})
            return None

        return chain

    except Exception as e:
        reason = f'{type(e).__name__}: {e}'
        print(f'  [chain] {opt_type} fetch failed: {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR', 'opt_type': opt_type,
              'expiry': str(expiry), 'reason': reason})
        return None


def _fetch_combined_chain(ticker, expiry, now_str):
    """Fetch and merge put + call chains into one {symbol: {...}} dict."""
    put_chain  = _fetch_options_chain_side(ticker, expiry, 'put', now_str)
    call_chain = _fetch_options_chain_side(ticker, expiry, 'call', now_str)
    if put_chain is None or call_chain is None:
        return None
    merged = {}
    merged.update(put_chain)
    merged.update(call_chain)
    return merged


def _bs_put_delta(S, K, T_years, sigma):
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return -1.0 if K > S else 0.0
        d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T_years) / (
            sigma * math.sqrt(T_years)
        )
        return norm.cdf(d1) - 1.0
    except Exception:
        return None


def _bs_call_delta(S, K, T_years, sigma):
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 1.0 if K < S else 0.0
        d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T_years) / (
            sigma * math.sqrt(T_years)
        )
        return norm.cdf(d1)
    except Exception:
        return None


def _find_short_put_strike(chain, spy_px, expiry, vix, now_str):
    """Find OTM put with |delta| closest to TARGET_DELTA. Same priority order as
    credit_spread_strat.py: Alpaca greeks → per-contract IV via BS → VIX fallback."""
    T_years        = max((expiry - date.today()).days / 365.0, 1 / 365)
    sigma_fallback = (vix or 20.0) / 100.0

    best_sym, best_strike, best_delta = None, None, None
    best_diff = float('inf')
    bs_used = bs_failed = 0

    for sym, data in chain.items():
        if data.get('type') != 'put':
            continue
        strike = data['strike']
        if strike <= 0 or data['mid'] <= 0 or strike >= spy_px:
            continue

        delta = data['delta']
        if delta is None:
            per_iv = data.get('iv')
            sigma  = float(per_iv) if per_iv else sigma_fallback
            delta  = _bs_put_delta(spy_px, strike, T_years, sigma)
            if delta is not None:
                bs_used += 1
            else:
                bs_failed += 1
                continue

        diff = abs(abs(delta) - TARGET_DELTA)
        if diff < best_diff:
            best_diff, best_sym, best_strike, best_delta = diff, sym, strike, delta

    if bs_used or bs_failed:
        print(f'  [chain] put BS fallback: {bs_used} used, {bs_failed} failed')

    if best_sym is None:
        _log({'timestamp': now_str, 'event': 'CHAIN_NO_DELTA', 'side': 'put',
              'reason': 'no OTM puts with computable delta in chain'})
        return None, None, None

    if best_diff > DELTA_TOLERANCE:
        _log({'timestamp': now_str, 'event': 'CHAIN_DELTA_TOO_FAR', 'side': 'put',
              'best_strike': best_strike, 'best_delta': round(best_delta, 4)})
        return None, None, None

    return best_sym, best_strike, best_delta


def _find_short_call_strike(chain, spy_px, expiry, vix, now_str):
    """Find OTM call with delta closest to TARGET_DELTA."""
    T_years        = max((expiry - date.today()).days / 365.0, 1 / 365)
    sigma_fallback = (vix or 20.0) / 100.0

    best_sym, best_strike, best_delta = None, None, None
    best_diff = float('inf')
    bs_used = bs_failed = 0

    for sym, data in chain.items():
        if data.get('type') != 'call':
            continue
        strike = data['strike']
        if strike <= 0 or data['mid'] <= 0 or strike <= spy_px:
            continue

        delta = data['delta']
        if delta is None:
            per_iv = data.get('iv')
            sigma  = float(per_iv) if per_iv else sigma_fallback
            delta  = _bs_call_delta(spy_px, strike, T_years, sigma)
            if delta is not None:
                bs_used += 1
            else:
                bs_failed += 1
                continue

        diff = abs(delta - TARGET_DELTA)
        if diff < best_diff:
            best_diff, best_sym, best_strike, best_delta = diff, sym, strike, delta

    if bs_used or bs_failed:
        print(f'  [chain] call BS fallback: {bs_used} used, {bs_failed} failed')

    if best_sym is None:
        _log({'timestamp': now_str, 'event': 'CHAIN_NO_DELTA', 'side': 'call',
              'reason': 'no OTM calls with computable delta in chain'})
        return None, None, None

    if best_diff > DELTA_TOLERANCE:
        _log({'timestamp': now_str, 'event': 'CHAIN_DELTA_TOO_FAR', 'side': 'call',
              'best_strike': best_strike, 'best_delta': round(best_delta, 4)})
        return None, None, None

    return best_sym, best_strike, best_delta


def _find_long_put_symbol(chain, short_strike):
    target = short_strike - SPREAD_WIDTH
    best_sym = best_k = None
    best_diff = float('inf')
    for sym, data in chain.items():
        if data.get('type') != 'put':
            continue
        diff = abs(data['strike'] - target)
        if diff < best_diff:
            best_diff, best_sym, best_k = diff, sym, data['strike']
    return best_sym, best_k


def _find_long_call_symbol(chain, short_strike):
    target = short_strike + SPREAD_WIDTH
    best_sym = best_k = None
    best_diff = float('inf')
    for sym, data in chain.items():
        if data.get('type') != 'call':
            continue
        diff = abs(data['strike'] - target)
        if diff < best_diff:
            best_diff, best_sym, best_k = diff, sym, data['strike']
    return best_sym, best_k


def _spread_mid(chain, short_sym, long_sym):
    """Net credit = short_mid − long_mid. Works for either put or call spread."""
    try:
        s_mid = chain[short_sym]['mid']
        l_mid = chain[long_sym]['mid']
        if s_mid <= 0 or l_mid < 0:
            return None
        return round(s_mid - l_mid, 4)
    except (KeyError, TypeError):
        return None


# ── SPREAD VALUE FOR MONITORING ────────────────────────────────────────────────

def _current_cost_to_close(short_sym, long_sym):
    """
    Current debit to close ONE spread (put or call) = short_mid − long_mid.
    Positive = we pay to close. Returns 0.0 when both legs are zero-priced
    (deeply OTM near expiry — valid, means max profit reached on that leg).
    Returns None only on hard API failures.
    Identical logic to credit_spread_strat.py's version — works for either
    the put spread or the call spread, called once per side.
    """
    ts = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    try:
        rs = None
        for _outer in range(2):
            rs = _alpaca_get(
                f'{DATA_URL}/v1beta1/options/snapshots',
                headers=_data_headers(),
                params={'symbols': f'{short_sym},{long_sym}', 'feed': 'indicative'},
            )
            if rs is not None:
                break
            if _outer == 0:
                time.sleep(5)
        if rs is None:
            reason = f'API returned None after retries ({short_sym}, {long_sym})'
            print(f'  [cost_to_close] {reason}')
            _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
                  'short_sym': short_sym, 'long_sym': long_sym, 'reason': 'api_none'})
            return None

        body  = rs.json()
        snaps = body.get('snapshots', {})

        missing = [s for s in (short_sym, long_sym) if s not in snaps]
        if missing:
            keys_sample = list(snaps.keys())[:4]
            reason = (f'symbols absent from snapshot response: {missing} '
                      f'(response had: {keys_sample})')
            print(f'  [cost_to_close] {reason}')
            _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
                  'short_sym': short_sym, 'long_sym': long_sym,
                  'reason': 'symbols_absent', 'missing': missing,
                  'response_keys': keys_sample})
            return None

        def _mid(sym):
            q   = snaps[sym].get('latestQuote', {})
            bid = float(q.get('bp') or 0)
            ask = float(q.get('ap') or 0)
            if bid + ask > 0:
                return (bid + ask) / 2.0
            return 0.0

        s_mid = _mid(short_sym)
        l_mid = _mid(long_sym)
        cost  = round(s_mid - l_mid, 4)
        if s_mid == 0.0 and l_mid == 0.0:
            print(f'  [cost_to_close] both legs zero-quoted ({short_sym}, {long_sym})'
                  f' → cost=0.00 (near-expiry worthless)')
        return cost

    except Exception as e:
        print(f'  [cost_to_close] {type(e).__name__}: {e}')
        _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
              'short_sym': short_sym, 'long_sym': long_sym,
              'reason': f'{type(e).__name__}: {e}'})
        return None


# ── ORDER MANAGEMENT ───────────────────────────────────────────────────────────
# No Alpaca SDK. Uses Alpaca v2 REST API via requests==2.32.5.
# Multi-leg spread orders: POST /v2/orders with order_class='mleg', up to 4 legs
# (Alpaca API reference: legs array maxItems=4). One 4-leg order opens/closes
# the whole condor atomically, rather than two separate 2-leg orders — avoids the
# risk of only the put spread or only the call spread filling and leaving a
# naked, unintended position.
#
# limit_price sign convention (Alpaca API reference, confirmed 2026-08-20):
#   positive = net DEBIT (you pay)     negative = net CREDIT (you receive)
# Opening this condor is a net credit  → limit_price must be NEGATIVE.
# Closing it (buying it back) is a net debit → limit_price stays POSITIVE.

def _place_open_order(short_put, long_put, short_call, long_call, total_credit):
    """Limit order to open the condor (4 legs, 1 order). ACTIVE-gated."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          'limit',
            'time_in_force': 'day',
            'order_class':   'mleg',
            'limit_price':   str(round(-total_credit, 2)),   # negative = credit
            'legs': [
                {'symbol': short_put,  'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_open'},
                {'symbol': long_put,   'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_open'},
                {'symbol': short_call, 'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_open'},
                {'symbol': long_call,  'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_open'},
            ],
        }
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get('id')
    except Exception as e:
        print(f'  [place_open] {e}')
        return None


def _place_close_order(short_put, long_put, short_call, long_call,
                        order_type='market', limit_price=None):
    """Order to close all 4 legs of the condor in one order. ACTIVE-gated.
    order_type: 'market' | 'limit'. limit_price (if given) is the max net
    debit willing to pay — positive, matching Alpaca's debit convention."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          order_type,
            'time_in_force': 'day',
            'order_class':   'mleg',
            'legs': [
                {'symbol': short_put,  'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_close'},
                {'symbol': long_put,   'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_close'},
                {'symbol': short_call, 'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_close'},
                {'symbol': long_call,  'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_close'},
            ],
        }
        if order_type == 'limit' and limit_price is not None:
            payload['limit_price'] = str(round(limit_price, 2))
        r = requests.post(f'{PAPER_BASE_URL}/v2/orders',
                          headers=_headers(), json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get('id')
    except Exception as e:
        print(f'  [place_close] {e}')
        return None


def _get_order(order_id):
    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/orders/{order_id}', headers=_headers())
        if r is None:
            return None
        return r.json()
    except Exception as e:
        print(f'  [get_order] {e}')
        return None


def _cancel_order(order_id):
    if not ACTIVE:
        return
    try:
        r = requests.delete(f'{PAPER_BASE_URL}/v2/orders/{order_id}',
                            headers=_headers(), timeout=10)
        if r.status_code not in (200, 204):
            print(f'  [cancel_order] status {r.status_code}')
    except Exception as e:
        print(f'  [cancel_order] {e}')


def _leg_fill_prices(order):
    """Extract per-leg filled_avg_price from a filled mleg order's legs array.
    Returns {symbol: price} for legs that reported a fill."""
    out = {}
    for leg in order.get('legs') or []:
        px = leg.get('filled_avg_price')
        if px is not None:
            out[leg['symbol']] = abs(float(px))
    return out


def _leg_close_costs_from_order(order, pos):
    """Given a filled 4-leg close order and the position it closed, split the
    combined fill into (put_close_cost, call_close_cost) using per-leg fills.
    Falls back to the order-level filled_avg_price split proportionally by
    stop/target thresholds if per-leg data is missing (should not normally
    happen for a filled mleg order, but this must never crash the reconcile
    or monitor loop)."""
    fills = _leg_fill_prices(order)
    sp, lp = pos.get('short_put_symbol'), pos.get('long_put_symbol')
    sc, lc = pos.get('short_call_symbol'), pos.get('long_call_symbol')
    if all(s in fills for s in (sp, lp, sc, lc)):
        put_cost  = round(fills[sp] - fills[lp], 4)
        call_cost = round(fills[sc] - fills[lc], 4)
        return put_cost, call_cost
    # fallback: split the order-level combined price using the pre-close ratio
    # of each side's credit as a weight — imperfect, but only used if Alpaca
    # omits per-leg fills, which normal mleg fills always include.
    raw = order.get('filled_avg_price')
    combined = abs(float(raw)) if raw is not None else (pos.get('profit_target') or 0.0)
    put_credit  = pos.get('put_credit', 0.0) or 0.0
    call_credit = pos.get('call_credit', 0.0) or 0.0
    total = put_credit + call_credit
    if total <= 0:
        return combined / 2, combined / 2
    return round(combined * put_credit / total, 4), round(combined * call_credit / total, 4)


# ── ENTRY EXECUTION ────────────────────────────────────────────────────────────

def _attempt_entry(pos_state, weekly, now_str, ticker,
                   short_put, long_put, short_call, long_call,
                   short_put_strike, long_put_strike,
                   short_call_strike, long_call_strike,
                   expiry, put_credit, call_credit, stock_px,
                   short_put_delta, short_call_delta):
    """
    Place the 4-leg opening order, poll for fill up to ORDER_FILL_TIMEOUT seconds.
    Updates pos_state['positions'] in place on fill. Returns True if filled.
    ACTIVE-gated. Mirrors credit_spread_strat.py's _attempt_entry exactly, just
    with 4 legs and put/call credit split from per-leg fills instead of 2.
    """
    if not ACTIVE:
        return False

    total_credit = round(put_credit + call_credit, 4)
    order_id = _place_open_order(short_put, long_put, short_call, long_call, total_credit)
    if not order_id:
        _log({'timestamp': now_str, 'event': 'ORDER_PLACE_FAILED',
              'short_put': short_put, 'short_call': short_call, 'credit': total_credit})
        print('  [entry] order placement failed')
        return False

    print(f'  [entry] order {order_id} placed — polling fill (max {ORDER_FILL_TIMEOUT}s)…')
    deadline = time.time() + ORDER_FILL_TIMEOUT

    while time.time() < deadline:
        time.sleep(30)
        order = _get_order(order_id)
        if order is None:
            continue
        status = order.get('status', '')

        if status == 'filled':
            fills = _leg_fill_prices(order)
            if all(s in fills for s in (short_put, long_put, short_call, long_call)):
                fill_put_credit  = round(fills[short_put]  - fills[long_put], 4)
                fill_call_credit = round(fills[short_call] - fills[long_call], 4)
            else:
                # Extremely defensive fallback — should not happen on a filled mleg order.
                raw_price = float(order.get('filled_avg_price') or total_credit)
                fill_combined = abs(raw_price)
                if fill_combined != total_credit:
                    print(f'  [entry] per-leg fills missing — using order-level '
                          f'combined ${fill_combined:.2f} split by expected ratio')
                fill_put_credit  = round(fill_combined * put_credit  / total_credit, 4) if total_credit else 0.0
                fill_call_credit = round(fill_combined * call_credit / total_credit, 4) if total_credit else 0.0

            fill_total = round(fill_put_credit + fill_call_credit, 4)
            # Max loss for an iron condor with equal-width wings = width*100 - total_credit*100
            # (only one side can be breached at expiry).
            max_risk = round((SPREAD_WIDTH - fill_total) * 100, 2)
            put_breakeven  = round(short_put_strike - fill_total, 2)
            call_breakeven = round(short_call_strike + fill_total, 2)

            pos = {
                'ticker':             ticker,
                'short_put_symbol':   short_put,
                'long_put_symbol':    long_put,
                'short_call_symbol':  short_call,
                'long_call_symbol':   long_call,
                'short_put_strike':   short_put_strike,
                'long_put_strike':    long_put_strike,
                'short_call_strike':  short_call_strike,
                'long_call_strike':   long_call_strike,
                'expiration':         expiry.isoformat(),
                'put_credit':         fill_put_credit,
                'call_credit':        fill_call_credit,
                'total_credit':       fill_total,
                'max_risk':           max_risk,
                'put_breakeven':      put_breakeven,
                'call_breakeven':     call_breakeven,
                'profit_target':      round(fill_total * PROFIT_TARGET_PCT, 4),
                'put_stop_cost':      round(fill_put_credit * STOP_LOSS_PCT, 4),
                'call_stop_cost':     round(fill_call_credit * STOP_LOSS_PCT, 4),
                'open_time':          now_str,
                'entry_order_id':     order_id,
                'short_put_delta':    round(short_put_delta, 4) if short_put_delta else None,
                'short_call_delta':   round(short_call_delta, 4) if short_call_delta else None,
                'spy_entry_px':       stock_px,
            }
            pos_state['positions'].append(pos)
            _save_positions(pos_state)
            _log({'timestamp': now_str, 'event': 'ENTRY_FILLED', **pos})
            _log_trade({'timestamp': now_str, 'type': 'OPEN', **pos})
            _discord(
                f'🟢 CONDOR OPEN | '
                f'{ticker} {short_put_strike:.0f}/{long_put_strike:.0f}P '
                f'{short_call_strike:.0f}/{long_call_strike:.0f}C  exp {expiry} | '
                f'Credit: ${fill_total:.2f} (put ${fill_put_credit:.2f} / call ${fill_call_credit:.2f}) | '
                f'Max risk: ${max_risk:.2f} | '
                f'Breakevens: ${put_breakeven:.2f} / ${call_breakeven:.2f} | '
                f'Positions open: {len(pos_state["positions"])}'
            )
            print(f'  [entry] FILLED put ${fill_put_credit:.2f} / call ${fill_call_credit:.2f}  '
                  f'{short_put} / {long_put} / {short_call} / {long_call}')
            return True

        if status == 'partially_filled':
            print(f'  [entry] partial fill detected on 4-leg order — cancelling {order_id}')
            _log({'timestamp': now_str, 'event': 'ORDER_PARTIAL_FILL', 'order_id': order_id})
            _cancel_order(order_id)
            _discord(
                f'⚠️ Partial fill on condor entry order — cancelled. '
                f'Check Alpaca for any open legs that need manual closing.'
            )
            return False

        if status in ('cancelled', 'expired', 'rejected', 'done_for_day'):
            print(f'  [entry] order {status} — no fill')
            _log({'timestamp': now_str, 'event': f'ORDER_{status.upper()}',
                  'order_id': order_id})
            return False

    print(f'  [entry] fill timeout ({ORDER_FILL_TIMEOUT}s) — cancelling {order_id}')
    _cancel_order(order_id)
    _log({'timestamp': now_str, 'event': 'ENTRY_FILL_TIMEOUT', 'order_id': order_id})
    return False


# ── POSITION MONITORING ────────────────────────────────────────────────────────

def _record_exit(pos_state, weekly, pos, reason, put_close_cost, call_close_cost, now_str):
    """
    Log exit, update weekly_realized_loss, fire Discord.
    Does NOT remove pos from pos_state — caller handles that and saves both files.
    Mirrors credit_spread_strat.py's _record_exit; pnl combines both legs.
    """
    put_credit  = pos['put_credit']
    call_credit = pos['call_credit']
    pnl = round((put_credit - put_close_cost + call_credit - call_close_cost) * 100, 2)
    tkr = pos.get('ticker', 'SPY')
    label = _pos_label(pos)

    old_loss = weekly.get('weekly_realized_loss', 0.0)
    new_loss = round(max(0.0, old_loss - pnl), 2)
    weekly['weekly_realized_loss'] = new_loss

    exit_entry = {
        'timestamp':            now_str,
        'type':                 'CLOSE',
        'event':                'EXIT',
        'reason':                reason,
        'label':                 label,
        'short_put_symbol':      pos['short_put_symbol'],
        'long_put_symbol':       pos['long_put_symbol'],
        'short_call_symbol':     pos['short_call_symbol'],
        'long_call_symbol':      pos['long_call_symbol'],
        'expiration':            pos['expiration'],
        'put_credit':            put_credit,
        'call_credit':           call_credit,
        'put_close_cost':        put_close_cost,
        'call_close_cost':       call_close_cost,
        'pnl':                   pnl,
        'weekly_realized_loss':  new_loss,
    }
    _log(exit_entry)
    _log_trade(exit_entry)

    reason_labels = {
        'PROFIT_TARGET': 'profit target',
        'PUT_STOP':      'put-side stop',
        'CALL_STOP':     'call-side stop',
        'EXPIRATION':    'expiration close',
    }
    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
    _discord(
        f'{"✅" if pnl >= 0 else "🔴"} CONDOR CLOSED '
        f'({reason_labels.get(reason, reason)}) | '
        f'{label} | '
        f'Credit: ${put_credit + call_credit:.2f}  '
        f'Close: put ${put_close_cost:.4f} / call ${call_close_cost:.4f} | '
        f'P&L: {pnl_str} | '
        f'Week loss: ${new_loss:.2f} / ${WEEKLY_LOSS_LIMIT:.0f}'
    )

    if new_loss >= WEEKLY_LOSS_LIMIT and not weekly.get('cooldown_active'):
        weekly['cooldown_active'] = True
        _discord(
            f'🚨 WEEKLY LOSS LIMIT HIT — ${new_loss:.2f} in net losses this week. '
            f'No new entries until Monday.'
        )
        _log({'timestamp': now_str, 'event': 'WEEKLY_LOSS_COOLDOWN',
              'weekly_realized_loss': new_loss})
        print(f'  [weekly limit] COOLDOWN activated — week loss ${new_loss:.2f}')


def _monitor_positions(pos_state, weekly, now_str):
    """
    Check all open condors for profit target / either-leg stop / expiration close.
    Mode A behavior: the WHOLE condor closes if EITHER leg's cost reaches 200% of
    its OWN credit, OR the combined (put+call) cost reaches 50% of combined credit.
    ACTIVE-gated. Same pending-close verification, stale-price fallback, and
    expiry force-close structure as credit_spread_strat.py's _monitor_positions —
    adapted for 4-leg positions and two independent stop thresholds.
    """
    if not ACTIVE:
        return

    positions = pos_state.get('positions', [])
    if not positions:
        return

    today             = datetime.now(ET).date()
    now_et            = datetime.now(ET)
    to_close          = []
    positions_updated = False

    for pos in positions:
        if pos.get('reconciled') and pos.get('short_put_symbol') == 'UNKNOWN':
            print(f'  [monitor] Skipping untracked reconciled condor — manual review required')
            continue

        sp, lp = pos['short_put_symbol'], pos['long_put_symbol']
        sc, lc = pos['short_call_symbol'], pos['long_call_symbol']
        expiry = date.fromisoformat(pos['expiration'])
        label  = _pos_label(pos)
        leg_key = (sp, lp, sc, lc)

        try:
            # ── Check pending close order (profit-target limit) ────────────────
            pending_id = pos.get('pending_close_order_id')
            if pending_id:
                order    = _get_order(pending_id)
                o_status = order.get('status', '') if order else ''

                if o_status == 'filled':
                    put_cost, call_cost = _leg_close_costs_from_order(order, pos)
                    print(f'  {label}: pending close CONFIRMED FILLED '
                          f'(put=${put_cost:.4f} call=${call_cost:.4f})')
                    _log({'timestamp': now_str, 'event': 'PROFIT_TARGET_CLOSE_CONFIRMED',
                          'label': label, 'order_id': pending_id,
                          'put_close_cost': put_cost, 'call_close_cost': call_cost})
                    _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', put_cost, call_cost, now_str)
                    to_close.append(pos)
                    continue

                if o_status in ('expired', 'cancelled', 'rejected', 'done_for_day'):
                    print(f'  {label}: pending close {o_status} — retrying with market order')
                    _log({'timestamp': now_str, 'event': 'PROFIT_TARGET_CLOSE_EXPIRED',
                          'label': label, 'order_id': pending_id, 'order_status': o_status})
                    _discord(
                        f'⚠️ PROFIT-TARGET CLOSE EXPIRED UNFILLED | {label} | '
                        f'Day-limit {pending_id[:8]}… {o_status}. '
                        f'Retrying with market order now.'
                    )
                    pos['pending_close_order_id'] = None
                    positions_updated = True
                    put_cost  = _current_cost_to_close(sp, lp)
                    call_cost = _current_cost_to_close(sc, lc)
                    if put_cost is None:
                        put_cost = pos['put_credit'] * PROFIT_TARGET_PCT
                    if call_cost is None:
                        call_cost = pos['call_credit'] * PROFIT_TARGET_PCT
                    retry_ok = _place_close_order(sp, lp, sc, lc, order_type='market')
                    if retry_ok:
                        _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', put_cost, call_cost, now_str)
                        to_close.append(pos)
                    else:
                        _discord(
                            f'🚨 MARKET RETRY ALSO FAILED | {label} | '
                            f'Could not close after limit expiry. Manual close required in Alpaca.'
                        )
                    continue

                # Order still open/pending — check either stop before waiting
                put_check  = _current_cost_to_close(sp, lp)
                call_check = _current_cost_to_close(sc, lc)
                put_stop_hit  = put_check  is not None and put_check  >= pos['put_stop_cost']
                call_stop_hit = call_check is not None and call_check >= pos['call_stop_cost']
                if put_stop_hit or call_stop_hit:
                    stop_reason = 'PUT_STOP' if put_stop_hit else 'CALL_STOP'
                    print(f'  {label}: {stop_reason} overrides pending profit-target close')
                    _cancel_order(pending_id)
                    pos['pending_close_order_id'] = None
                    positions_updated = True
                    ok = _place_close_order(sp, lp, sc, lc, order_type='market')
                    if ok:
                        pc = put_check if put_check is not None else pos['put_credit']
                        cc = call_check if call_check is not None else pos['call_credit']
                        _record_exit(pos_state, weekly, pos, stop_reason, pc, cc, now_str)
                        to_close.append(pos)
                    else:
                        _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                              'label': label, 'reason': stop_reason})
                        _discord(f'⚠️ Close order failed | {label} | {stop_reason} | Retrying next scan')
                    continue

                if order is None:
                    print(f'  {label}: pending close {pending_id[:8]}… — order query failed, skipping')
                else:
                    print(f'  {label}: pending close {pending_id[:8]}… still {o_status} — waiting')
                continue

            # ── Expiration force-close (9:45am ET on expiry day) ─────────────
            if today == expiry and now_et.hour == 9 and now_et.minute >= 45:
                print(f'  {label}: EXPIRATION CLOSE')
                put_cost  = _current_cost_to_close(sp, lp) or 0.0
                call_cost = _current_cost_to_close(sc, lc) or 0.0
                ok = _place_close_order(sp, lp, sc, lc, order_type='market')
                if ok:
                    _record_exit(pos_state, weekly, pos, 'EXPIRATION', put_cost, call_cost, now_str)
                    to_close.append(pos)
                    _log({'timestamp': now_str, 'event': 'MONITOR_EXPIRY_CLOSE',
                          'label': label, 'put_close_cost': put_cost, 'call_close_cost': call_cost})
                else:
                    print(f'  {label}: EXPIRY CLOSE ORDER FAILED')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'EXPIRATION'})
                    _discord(
                        f'🚨 CLOSE ORDER FAILED | {label} | EXPIRATION DAY | '
                        f'Manual close required in Alpaca immediately.'
                    )
                continue

            # ── Current spread values (put side, call side) ───────────────────
            put_cost  = _current_cost_to_close(sp, lp)
            call_cost = _current_cost_to_close(sc, lc)

            if put_cost is None or call_cost is None:
                fail_count = _monitor_consec_fail.get(leg_key, 0) + 1
                _monitor_consec_fail[leg_key] = fail_count
                cached = _last_known_cost.get(leg_key)
                stale_put = stale_call = stale_age_min = None
                if cached:
                    stale_age_min = (datetime.now(ET) - cached['ts']).total_seconds() / 60
                    if stale_age_min <= STALE_COST_MAX_MINUTES:
                        stale_put  = cached['put_cost']
                        stale_call = cached['call_cost']
                age_str = f'{stale_age_min:.0f}min ago' if stale_age_min is not None else 'never'
                print(f'  {label}: value unavailable — skipping this cycle '
                      f'({fail_count} consecutive; last good: {age_str})')
                _log({'timestamp': now_str, 'event': 'MONITOR_VALUE_UNAVAILABLE',
                      'label': label, 'consecutive_failures': fail_count,
                      'stale_put_cost': stale_put, 'stale_call_cost': stale_call,
                      'stale_age_min': round(stale_age_min, 1) if stale_age_min is not None else None})
                if fail_count >= MONITOR_ALERT_THRESHOLD and leg_key not in _monitor_alert_sent:
                    _monitor_alert_sent.add(leg_key)
                    _discord(
                        f'⚠️ STOP-LOSS MONITORING DEGRADED | {label} | '
                        f'{fail_count} consecutive pricing failures (~{fail_count * 5} min). '
                        f'Last good price: {age_str}. '
                        f'Stops at put ${pos["put_stop_cost"]:.2f} / call ${pos["call_stop_cost"]:.2f} '
                        f'cannot be auto-enforced. Manual monitoring required.'
                    )
                # Use whichever side has a usable value (live or stale-within-window);
                # if EITHER side clears its own stop, close the whole condor.
                eff_put  = put_cost  if put_cost  is not None else stale_put
                eff_call = call_cost if call_cost is not None else stale_call
                put_stop_hit  = eff_put  is not None and eff_put  >= pos['put_stop_cost']
                call_stop_hit = eff_call is not None and eff_call >= pos['call_stop_cost']
                if put_stop_hit or call_stop_hit:
                    stop_reason = 'PUT_STOP' if put_stop_hit else 'CALL_STOP'
                    print(f'  {label}: STALE-PRICE {stop_reason} '
                          f'(put={eff_put} call={eff_call} from {stale_age_min:.0f}min ago)'
                          if stale_age_min is not None else
                          f'  {label}: {stop_reason} on partially-live pricing')
                    _log({'timestamp': now_str, 'event': 'MONITOR_STALE_STOP_LOSS',
                          'label': label, 'stop_reason': stop_reason,
                          'stale_put_cost': eff_put, 'stale_call_cost': eff_call,
                          'stale_age_min': round(stale_age_min, 1) if stale_age_min is not None else None})
                    _discord(
                        f'⚠️ STOP-LOSS TRIGGERED ON STALE/PARTIAL PRICE | {label} | '
                        f'{stop_reason} — put ${eff_put or 0:.4f} / call ${eff_call or 0:.4f} '
                        f'vs stops put ${pos["put_stop_cost"]:.4f} / call ${pos["call_stop_cost"]:.4f}. '
                        f'Placing market close order.'
                    )
                    ok = _place_close_order(sp, lp, sc, lc, order_type='market')
                    if ok:
                        _record_exit(pos_state, weekly, pos, stop_reason,
                                     eff_put or 0.0, eff_call or 0.0, now_str)
                        to_close.append(pos)
                    else:
                        _discord(
                            f'🚨 STALE-PRICE STOP LOSS ORDER FAILED | {label} | '
                            f'Manual close required immediately.'
                        )
                continue

            # Update last-known-good cache and reset failure counters on success
            _last_known_cost[leg_key] = {'put_cost': put_cost, 'call_cost': call_cost, 'ts': datetime.now(ET)}
            _monitor_consec_fail[leg_key] = 0
            _monitor_alert_sent.discard(leg_key)

            combined_cost = round(put_cost + call_cost, 4)
            print(f'  {label}: put_cost={put_cost:.4f}  call_cost={call_cost:.4f}  '
                  f'combined={combined_cost:.4f}  tgt≤{pos["profit_target"]:.4f}  '
                  f'put_stop≥{pos["put_stop_cost"]:.4f}  call_stop≥{pos["call_stop_cost"]:.4f}')

            # ── Profit target: combined cost ≤ 50% of combined credit ────────
            if combined_cost <= pos['profit_target']:
                print(f'  {label}: PROFIT TARGET')
                order_id = _place_close_order(sp, lp, sc, lc,
                                              order_type='limit',
                                              limit_price=pos['profit_target'])
                if order_id:
                    pos['pending_close_order_id'] = order_id
                    positions_updated = True
                    _log({'timestamp': now_str, 'event': 'PROFIT_TARGET_CLOSE_SUBMITTED',
                          'label': label, 'order_id': order_id,
                          'limit_price': pos['profit_target']})
                else:
                    print(f'  {label}: profit-target close failed — retrying next cycle')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'PROFIT_TARGET'})
                    _discord(f'⚠️ Close order failed | {label} | Profit target | Retrying next scan')
                continue

            # ── Either-leg stop: put_cost ≥ 200% of put credit, OR call_cost ≥ 200% of call credit ──
            put_stop_hit  = put_cost  >= pos['put_stop_cost']
            call_stop_hit = call_cost >= pos['call_stop_cost']
            if put_stop_hit or call_stop_hit:
                stop_reason = 'PUT_STOP' if put_stop_hit else 'CALL_STOP'
                print(f'  {label}: {stop_reason}')
                ok = _place_close_order(sp, lp, sc, lc, order_type='market')
                if ok:
                    _record_exit(pos_state, weekly, pos, stop_reason, put_cost, call_cost, now_str)
                    to_close.append(pos)
                else:
                    print(f'  {label}: {stop_reason} close failed — retrying next cycle')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': stop_reason})
                    _discord(f'⚠️ Close order failed | {label} | {stop_reason} | Retrying next scan')

        except Exception as e:
            print(f'  {label}: MONITOR ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'MONITOR_POSITION_ERROR',
                  'label': label, 'error': f'{type(e).__name__}: {e}'})

    if to_close:
        for pos in to_close:
            try:
                pos_state['positions'].remove(pos)
            except ValueError:
                pass
        _save_positions(pos_state)
        _save_weekly(weekly)
    elif positions_updated:
        _save_positions(pos_state)


# ── MACRO EVENT FILTER ─────────────────────────────────────────────────────────

def _macro_event_today():
    today = datetime.now(ET).date()
    today_str = today.isoformat()

    if today_str in FOMC_DAYS:
        return 'FOMC meeting day'
    if today_str in CPI_DAYS:
        return 'CPI release day'
    if today_str in GDP_DAYS:
        return 'GDP release day'

    first          = today.replace(day=1)
    days_to_friday = (4 - first.weekday()) % 7
    if today == first + timedelta(days=days_to_friday):
        return 'Jobs Report day (NFP)'

    return None


# ── ENTRY CONDITIONS ───────────────────────────────────────────────────────────

def _check_entry_conditions(ticker, pos_state, weekly):
    """Evaluate all gates cheapest-first. Same structure/order as credit_spread_strat.py."""
    conds    = {}
    stock_px = ivr = vix = None

    def _c(name, passed, detail):
        conds[name] = {'passed': bool(passed), 'detail': str(detail)}

    macro = _macro_event_today()
    _c('macro_event', macro is None, macro or 'none')
    if macro:
        return False, conds, stock_px, ivr, vix

    _c('active',          ACTIVE, 'True' if ACTIVE else 'False — DORMANT')
    _c('entry_window',    _in_entry_window(), '9:45–3:30 ET')
    _c('position_limit',  len(pos_state.get('positions', [])) < MAX_POSITIONS,
       f'{len(pos_state.get("positions", []))}/{MAX_POSITIONS} open')
    _c('weekly_cooldown', not weekly.get('cooldown_active', False),
       f'loss=${weekly.get("weekly_realized_loss", 0):.2f} / ${WEEKLY_LOSS_LIMIT:.0f}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, stock_px, ivr, vix

    # Underwater gate: with 1 open condor, block a 2nd if either alive leg is underwater
    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_put_symbol') == 'UNKNOWN')]
    if len(real_pos) == 1:
        p = real_pos[0]
        put_cost  = _current_cost_to_close(p['short_put_symbol'], p['long_put_symbol'])
        call_cost = _current_cost_to_close(p['short_call_symbol'], p['long_call_symbol'])
        put_underwater  = put_cost  is not None and put_cost  > p.get('put_credit', 0)
        call_underwater = call_cost is not None and call_cost > p.get('call_credit', 0)
        if put_underwater or call_underwater:
            _c('underwater_block', False,
               f'put_cost={put_cost}  call_cost={call_cost}')
            return False, conds, stock_px, ivr, vix

    above_sma, stock_close, sma_val = _above_sma20(ticker)
    stock_px = stock_close
    sma_key  = f'{ticker.lower()}_above_sma'
    if above_sma is None:
        _c(sma_key, False, 'data unavailable')
    else:
        _c(sma_key, above_sma,
           f'{ticker}={stock_close:.2f}  SMA20={sma_val:.2f}  '
           f'{"above" if above_sma else "BELOW"}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, stock_px, ivr, vix

    ivr, vix = _spy_ivrank()
    if ivr is None:
        _c('iv_rank', False, 'SPY IV data unavailable')
        _c('vix_cap',  False, 'SPY IV data unavailable')
    else:
        _c('iv_rank', ivr >= MIN_IVR,
           f'IVR={ivr:.1f}% (need ≥{MIN_IVR}%) — SPY IV={vix:.1f}%')
        _c('vix_cap',  vix < MAX_VIX,
           f'SPY IV={vix:.1f}% {"<" if vix < MAX_VIX else "≥"} {MAX_VIX:.0f}%')

    return all(v['passed'] for v in conds.values()), conds, stock_px, ivr, vix


def _evaluate_ticker(ticker, pos_state, weekly, now_str):
    """
    Full entry evaluation for one ticker. Returns (signal_dict | None, conditions_dict).
    Requires BOTH a valid 20-delta short put AND a valid 20-delta short call, each
    meeting MIN_CREDIT_LEG independently — matches backtest_iron_condor.py exactly.
    """
    passed, conds, stock_px, ivr, vix = _check_entry_conditions(ticker, pos_state, weekly)
    if not passed:
        return None, conds

    expiry = _find_target_expiration(ticker)
    if expiry is None:
        conds['expiry_found'] = {'passed': False, 'detail': 'no valid expiration found'}
        return None, conds
    dte = (expiry - date.today()).days
    conds['expiry_found'] = {'passed': True, 'detail': f'{expiry} ({dte} DTE)'}

    chain = _fetch_combined_chain(ticker, expiry, now_str)
    if chain is None:
        conds['chain_fetched'] = {'passed': False,
                                   'detail': 'chain unavailable (see CHAIN_* log event)'}
        return None, conds
    conds['chain_fetched'] = {'passed': True, 'detail': f'{len(chain)} contracts'}

    short_put, short_put_strike, short_put_delta = _find_short_put_strike(
        chain, stock_px, expiry, vix, now_str
    )
    short_call, short_call_strike, short_call_delta = _find_short_call_strike(
        chain, stock_px, expiry, vix, now_str
    )
    if short_put is None or short_call is None:
        conds['short_strikes'] = {
            'passed': False,
            'detail': f'put={"ok" if short_put else "MISSING"}  call={"ok" if short_call else "MISSING"}',
        }
        return None, conds
    conds['short_strikes'] = {
        'passed': True,
        'detail': f'put ${short_put_strike:.0f} Δ{short_put_delta:.3f}  '
                  f'call ${short_call_strike:.0f} Δ{short_call_delta:.3f}',
    }

    long_put, long_put_strike = _find_long_put_symbol(chain, short_put_strike)
    long_call, long_call_strike = _find_long_call_symbol(chain, short_call_strike)
    if long_put is None or long_call is None:
        conds['long_strikes'] = {'passed': False, 'detail': 'not found in chain'}
        return None, conds
    conds['long_strikes'] = {'passed': True,
                              'detail': f'put ${long_put_strike:.0f}  call ${long_call_strike:.0f}'}

    put_credit  = _spread_mid(chain, short_put, long_put)
    call_credit = _spread_mid(chain, short_call, long_call)
    if put_credit is None or call_credit is None or \
       put_credit < MIN_CREDIT_LEG or call_credit < MIN_CREDIT_LEG:
        conds['min_credit'] = {
            'passed': False,
            'detail': f'put=${put_credit}  call=${call_credit}  (need ≥${MIN_CREDIT_LEG} each)',
        }
        return None, conds
    conds['min_credit'] = {'passed': True,
                            'detail': f'put=${put_credit:.4f}  call=${call_credit:.4f}'}

    return {
        'ticker':             ticker,
        'expiry':              expiry,
        'short_put':            short_put,
        'long_put':             long_put,
        'short_call':           short_call,
        'long_call':            long_call,
        'short_put_strike':     short_put_strike,
        'long_put_strike':      long_put_strike,
        'short_call_strike':    short_call_strike,
        'long_call_strike':     long_call_strike,
        'short_put_delta':      short_put_delta,
        'short_call_delta':     short_call_delta,
        'put_credit':           put_credit,
        'call_credit':          call_credit,
        'stock_px':             stock_px,
        'ivr':                  ivr,
        'vix':                  vix,
    }, conds


# ── DAILY SUMMARY ──────────────────────────────────────────────────────────────

def _check_daily_summary(pos_state, weekly, now_str):
    """Send once-daily summary at 3:35pm ET. ACTIVE-gated. Same dedup pattern
    (DB-first, then JSON, to survive a DB-write-failure/JSON-fallback split
    brain) as credit_spread_strat.py."""
    if not ACTIVE:
        return
    today_str = datetime.now(ET).date().isoformat()
    if pos_state.get('daily_summary_sent') == today_str:
        return
    if not _is_summary_time():
        return

    daily_db, vitals_db = _db_read_summary_flags()
    if daily_db == today_str:
        pos_state['daily_summary_sent'] = today_str
        return
    try:
        with open(POSITIONS_FILE) as _f:
            _j = json.load(_f)
        if _j.get('daily_summary_sent') == today_str:
            pos_state['daily_summary_sent'] = today_str
            _db_write_summary_flags(today_str, _j.get('morning_vitals_sent'))
            return
    except Exception:
        pass

    pos_state['daily_summary_sent'] = today_str
    _save_positions(pos_state)

    positions = pos_state.get('positions', [])
    real_pos  = [p for p in positions
                 if not (p.get('reconciled') and p.get('short_put_symbol') == 'UNKNOWN')]
    week_loss = weekly.get('weekly_realized_loss', 0.0)

    try:
        with open(TRADE_LOG_FILE) as f:
            tlog = json.load(f)
        today_opens  = [t for t in tlog
                        if t.get('type') == 'OPEN'
                        and str(t.get('timestamp', '')).startswith(today_str)]
        today_closes = [t for t in tlog
                        if t.get('type') == 'CLOSE'
                        and str(t.get('timestamp', '')).startswith(today_str)]
        realized_pnl  = sum(t.get('pnl', 0) for t in today_closes)
        entries_today = len(today_opens)
        exits_today   = len(today_closes)
    except Exception:
        realized_pnl = entries_today = exits_today = 0

    unrealized_pnl = 0.0
    pos_costs: dict = {}
    for pos in real_pos:
        sp, lp = pos.get('short_put_symbol', ''), pos.get('long_put_symbol', '')
        sc, lc = pos.get('short_call_symbol', ''), pos.get('long_call_symbol', '')
        put_cost  = _current_cost_to_close(sp, lp)  if sp and lp else None
        call_cost = _current_cost_to_close(sc, lc)  if sc and lc else None
        pos_costs[id(pos)] = (put_cost, call_cost)
        if put_cost is not None:
            unrealized_pnl += round((pos.get('put_credit', 0) - put_cost) * 100, 2)
        if call_cost is not None:
            unrealized_pnl += round((pos.get('call_credit', 0) - call_cost) * 100, 2)

    alpaca_note = ''
    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is not None:
            today_ymd = datetime.now(ET).strftime('%y%m%d')
            other_legs = _other_strategy_leg_symbols()
            tracked_legs = [p for p in r.json()
                            if p.get('asset_class') == 'us_option'
                            and any(str(p.get('symbol', '')).startswith(t) for t in TICKERS)
                            and str(p.get('symbol', ''))[3:9] != today_ymd
                            and p.get('symbol', '') not in other_legs]
            alpaca_condors = len(tracked_legs) // 4
            file_condors   = len(real_pos)
            if alpaca_condors != file_condors:
                alpaca_note = (f' ⚠️ Alpaca shows {len(tracked_legs)} leg(s) '
                               f'({alpaca_condors} condor(s)) — state file has {file_condors}')
            else:
                alpaca_note = f' ✓ Alpaca confirms {len(tracked_legs)} leg(s)'
    except Exception:
        pass

    pos_detail = ''
    for pos in real_pos:
        put_cost, call_cost = pos_costs.get(id(pos), (None, None))
        unr_bits = []
        if put_cost is not None:
            unr_bits.append(f'put ${round((pos.get("put_credit",0)-put_cost)*100,2):+.2f}')
        if call_cost is not None:
            unr_bits.append(f'call ${round((pos.get("call_credit",0)-call_cost)*100,2):+.2f}')
        unr_str = ('  unreal ' + ' / '.join(unr_bits)) if unr_bits else ''
        tkr = pos.get('ticker', 'SPY')
        pos_detail += (
            f'\n  [{tkr}] {pos["short_put_strike"]:.0f}/{pos["long_put_strike"]:.0f}P '
            f'{pos["short_call_strike"]:.0f}/{pos["long_call_strike"]:.0f}C '
            f'exp={pos["expiration"]}  credit=${pos["total_credit"]:.2f}{unr_str}'
        )

    _discord(
        f'📊 **Iron Condor Daily Summary**\n'
        f'Open positions: {len(real_pos)}{alpaca_note}{pos_detail}\n'
        f'Entries today: {entries_today}  |  Exits today: {exits_today}\n'
        f'Realized P&L: ${realized_pnl:+.2f}  |  Unrealized: ${unrealized_pnl:+.2f}\n'
        f'Week-to-date loss: ${week_loss:.2f} / ${WEEKLY_LOSS_LIMIT:.0f} limit'
    )


# ── MORNING VITALS ─────────────────────────────────────────────────────────────

def _send_morning_vitals():
    """Post a market-open snapshot to Discord once per trading day at 9:30 ET.
    Same dedup/holiday-check pattern as credit_spread_strat.py."""
    if not ACTIVE:
        return

    today_str = datetime.now(ET).date().isoformat()
    pos_state = _load_positions()

    if pos_state.get('morning_vitals_sent') == today_str:
        return
    try:
        with open(POSITIONS_FILE) as _f:
            _j = json.load(_f)
        if _j.get('morning_vitals_sent') == today_str:
            _db_write_summary_flags(_j.get('daily_summary_sent'), today_str)
            return
    except Exception:
        pass

    pos_state['morning_vitals_sent'] = today_str
    _save_positions(pos_state)

    try:
        cal = _alpaca_get(
            f'{PAPER_BASE_URL}/v2/calendar',
            headers=_headers(),
            params={'start': today_str, 'end': today_str},
        )
        if cal is None or not cal.json():
            print(f'  [morning_vitals] market holiday ({today_str}) — skipping')
            return
    except Exception as e:
        print(f'  [morning_vitals] calendar check failed: {e} — sending anyway')

    weekly   = _load_weekly()
    now_et   = datetime.now(ET)
    date_str = now_et.strftime('%A %B %-d, %Y')

    spy_above, spy_px, spy_sma = _above_sma20('SPY')

    def _sma_line(ticker, px, sma, above):
        if px is None or sma is None:
            return f'{ticker}:  N/A'
        status = '✅ Above' if above else '❌ Below'
        return f'{ticker}:  ${px:.2f}  |  SMA20: ${sma:.2f}  |  {status}'

    spy_line = _sma_line('SPY', spy_px, spy_sma, spy_above)

    ivr, vix = _spy_ivrank()
    if vix is not None:
        vix_status = '✅ Clear' if vix < MAX_VIX else '❌ Elevated'
        vix_line = f'SPY IV: {vix:.1f}%  |  Limit: <{MAX_VIX:.0f}%     |  {vix_status}'
    else:
        vix_line = 'SPY IV: N/A'
    if ivr is not None:
        ivr_status = '✅ Clear' if ivr >= MIN_IVR else '❌ Low'
        ivr_line = f'IVR:  {ivr:.1f}%  |  Min: {MIN_IVR:.0f}%       |  {ivr_status}'
    else:
        ivr_line = 'IVR:  N/A'

    macro = _macro_event_today()
    if macro:
        macro_line = f'MACRO: {macro}  |  ⚠️ Blocked'
    else:
        macro_line = 'MACRO: None scheduled  |  ✅ Clear'

    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_put_symbol') == 'UNKNOWN')]
    pos_lines = f'OPEN POSITIONS: {len(real_pos)} of {MAX_POSITIONS}'
    for pos in real_pos:
        tkr = pos.get('ticker', 'SPY')
        put_cost  = _current_cost_to_close(pos.get('short_put_symbol', ''), pos.get('long_put_symbol', ''))
        call_cost = _current_cost_to_close(pos.get('short_call_symbol', ''), pos.get('long_call_symbol', ''))
        total_credit = pos.get('total_credit', 0)
        if put_cost is not None and call_cost is not None and total_credit > 0:
            combined = put_cost + call_cost
            pct_str = f'  ({(total_credit - combined) / total_credit * 100:.0f}% to target)'
        else:
            pct_str = ''
        cost_str = (f'put ${put_cost:.4f}/call ${call_cost:.4f}'
                    if put_cost is not None and call_cost is not None else 'N/A')
        pos_lines += (
            f'\n  [{tkr}] {pos["short_put_strike"]:.0f}/{pos["long_put_strike"]:.0f}P '
            f'{pos["short_call_strike"]:.0f}/{pos["long_call_strike"]:.0f}C'
            f'  exp {pos["expiration"]}'
            f'  cost {cost_str} / credit ${total_credit:.2f}'
            f'{pct_str}'
        )

    week_loss = weekly.get('weekly_realized_loss', 0.0)
    week_str  = f'-${week_loss:.2f}' if week_loss > 0 else '+$0.00'
    system_line = f'SYSTEM: ✅ Active  |  Week P&L: {week_str}'

    msg = (
        f'📊 **MORNING VITALS — {date_str}**\n\n'
        f'{spy_line}\n'
        f'{vix_line}\n'
        f'{ivr_line}\n'
        f'{macro_line}\n\n'
        f'{pos_lines}\n\n'
        f'{system_line}'
    )

    _discord(msg)
    print(f'  [morning_vitals] sent for {today_str}')
    _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
          'event': 'MORNING_VITALS_SENT', 'date': today_str})


# ── MAIN SCAN ──────────────────────────────────────────────────────────────────

def run_scan():
    global _last_scan_start, _last_scan_duration
    _t0 = datetime.now(ET)
    _last_scan_start = _t0

    if not is_market_hours():
        print(f'[{_t0.strftime("%H:%M ET")}] '
              f'Iron condor: outside market hours, skipping.')
        _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
        return

    now_et  = datetime.now(ET)
    now_str = now_et.strftime('%Y-%m-%d %H:%M ET')

    if now_et.hour == 9 and 30 <= now_et.minute < 35:
        try:
            _send_morning_vitals()
        except Exception as e:
            print(f'  [morning_vitals] ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'MORNING_VITALS_ERROR',
                  'error': f'{type(e).__name__}: {e}'})

    print(f'\n[{now_str}] Iron condor scan  (ACTIVE={ACTIVE})…')

    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)

    try:
        _monitor_positions(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [monitor] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'MONITOR_ERROR',
              'error': f'{type(e).__name__}: {e}'})

    try:
        _check_daily_summary(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [summary] ERROR — {type(e).__name__}: {e}')

    ticker_signals = {}
    ticker_conds   = {}
    scan_result    = 'no_signal'
    chosen         = None

    for ticker in TICKERS:
        try:
            signal, conds = _evaluate_ticker(ticker, pos_state, weekly, now_str)
            ticker_conds[ticker] = conds
            if signal:
                ticker_signals[ticker] = signal
        except Exception as e:
            print(f'  [{ticker}] conditions ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'CONDITIONS_ERROR',
                  'ticker': ticker, 'error': f'{type(e).__name__}: {e}'})

    if ticker_signals:
        chosen = max(ticker_signals.values(),
                     key=lambda s: s['put_credit'] + s['call_credit'])

    if chosen:
        t = chosen['ticker']
        total_credit = chosen['put_credit'] + chosen['call_credit']
        if not ACTIVE:
            scan_result = 'dormant_would_enter'
            print(
                f'  DORMANT MODE — entry skipped | '
                f'{t} {chosen["short_put_strike"]:.0f}/{chosen["long_put_strike"]:.0f}P '
                f'{chosen["short_call_strike"]:.0f}/{chosen["long_call_strike"]:.0f}C  '
                f'exp={chosen["expiry"]}  credit=${total_credit:.4f}'
            )
        else:
            print(f'  ENTERING: {t} {chosen["short_put_strike"]:.0f}/{chosen["long_put_strike"]:.0f}P '
                  f'{chosen["short_call_strike"]:.0f}/{chosen["long_call_strike"]:.0f}C  '
                  f'exp={chosen["expiry"]}  credit=${total_credit:.4f}')
            filled = _attempt_entry(
                pos_state, weekly, now_str, t,
                chosen['short_put'], chosen['long_put'],
                chosen['short_call'], chosen['long_call'],
                chosen['short_put_strike'], chosen['long_put_strike'],
                chosen['short_call_strike'], chosen['long_call_strike'],
                chosen['expiry'], chosen['put_credit'], chosen['call_credit'],
                chosen['stock_px'], chosen['short_put_delta'], chosen['short_call_delta'],
            )
            scan_result = 'entry_filled' if filled else 'entry_not_filled'
    else:
        macro_detail = next(
            (c['macro_event']['detail']
             for c in ticker_conds.values()
             if not c.get('macro_event', {}).get('passed', True)),
            None,
        )
        if macro_detail:
            print(f'  No entry — macro event day: {macro_detail}')
        else:
            for t, conds in ticker_conds.items():
                fails = [k for k, v in conds.items() if not v.get('passed')]
                if 'underwater_block' in fails:
                    print(f'  [{t}] No entry — blocked: existing position underwater')
                elif fails:
                    print(f'  [{t}] No entry — failed: {", ".join(fails)}')

    _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
    print(f'  [run_scan] completed in {_last_scan_duration:.1f}s')

    log_entry = {
        'timestamp':      now_str,
        'event':          'SCAN',
        'active':         ACTIVE,
        'scan_result':    scan_result,
        'open_positions': len(pos_state.get('positions', [])),
        'weekly_loss':    weekly.get('weekly_realized_loss', 0.0),
        'cooldown':       weekly.get('cooldown_active', False),
        'scan_duration_s': round(_last_scan_duration, 1),
        'conditions':     ticker_conds,
    }
    if scan_result in ('dormant_would_enter', 'entry_filled', 'entry_not_filled') and chosen:
        log_entry['signal'] = {
            'ticker':           chosen['ticker'],
            'short_put':        chosen['short_put'],
            'long_put':         chosen['long_put'],
            'short_call':       chosen['short_call'],
            'long_call':        chosen['long_call'],
            'expiry':           str(chosen['expiry']),
            'put_credit':       chosen['put_credit'],
            'call_credit':      chosen['call_credit'],
            'short_put_delta':  round(chosen['short_put_delta'], 4) if chosen['short_put_delta'] else None,
            'short_call_delta': round(chosen['short_call_delta'], 4) if chosen['short_call_delta'] else None,
            'stock_px':         chosen['stock_px'],
        }
    _log(log_entry)


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print('  SPY IRON CONDOR  |  7 DTE  |  20Δ / $5-wide  |  Mode A (combined stop)')
    print(f'  ACTIVE = {ACTIVE}')
    if not ACTIVE:
        print('  *** DORMANT — scanning and logging, NO orders placed ***')
    print('  Alpaca v2 REST API  |  raw requests  |  no SDK  |  4-leg mleg orders')
    print('  Scan every 5 min, 9:30–16:00 ET, Mon–Fri')
    print('=' * 62)

    _init_db()
    _init_files()
    _reconcile_on_startup()

    schedule.every(5).minutes.do(run_scan)

    if ACTIVE:
        _discord(
            f'✅ Iron Condor system live | SPY 7DTE iron condor | '
            f'20Δ put + 20Δ call, $5-wide each | ${MIN_CREDIT_LEG:.2f} min credit/leg | '
            f'2 positions max'
        )
    else:
        print('  Dormant mode: conditions evaluated and logged each scan.')

    run_scan()

    while True:
        tick_et  = datetime.now(ET)
        tick_str = tick_et.strftime('%Y-%m-%d %H:%M:%S ET')

        if is_market_hours():
            if _last_scan_start is not None:
                since_s = (tick_et - _last_scan_start).total_seconds()
                print(f'[scheduler] tick {tick_str} | '
                      f'last_scan_start={_last_scan_start.strftime("%H:%M:%S")} | '
                      f'since={since_s:.0f}s | '
                      f'last_duration={_last_scan_duration:.1f}s')

                if since_s > 600:
                    lag_min = round(since_s / 60, 1)
                    print(f'[scheduler] ⚠️  SCHEDULER_LAG — {lag_min} min since last scan started')
                    _log({'timestamp': tick_str, 'event': 'SCHEDULER_LAG',
                          'minutes_since_last_scan': lag_min,
                          'last_scan_duration_s': _last_scan_duration})
            else:
                print(f'[scheduler] tick {tick_str} | awaiting first scan')

        try:
            schedule.run_pending()
        except Exception as e:
            print(f'[scheduler] ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': tick_str, 'event': 'SCHEDULER_ERROR',
                  'error': f'{type(e).__name__}: {e}'})
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f'[FATAL] {type(e).__name__}: {e}')
        traceback.print_exc()
