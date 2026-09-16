#!/usr/bin/env python3
"""
zero_dte_v2_strat.py — 0DTE SPY Put Credit Spread Strategy, V2 (more aggressive).

STANDALONE: no imports from credit_spread_strat.py, iron_condor_strat.py,
archive/dte0_strat.py, scanner.py, or paper_execution.py.
DORMANT until ACTIVE = True.

Rebuilt from the archived archive/dte0_strat.py's 0DTE structure (entry window,
same-day force close, intraday Black-Scholes timing) on top of the CURRENT
hardened infrastructure patterns from credit_spread_strat.py / iron_condor_strat.py
— NOT the archived file's data-handling code, which predates all of the fixes
below:
  - PostgreSQL is the primary store (zero_dte_v2_positions, zero_dte_v2_state).
    Local JSON is only ever a degraded-mode fallback, never primary — see the
    '_degraded' flag notes above _load_positions()/_load_weekly() below.
  - Fill verification on BOTH open and close orders, for BOTH limit and market
    order types — an "order submitted" response is never treated as "order
    filled." Opens: poll until status == 'filled' and read filled_avg_price
    directly. Profit-target closes (limit): tracked via pending_close_order_id
    and confirmed filled on a later scan/reconcile, never recorded as closed
    from the pre-order quote. Stop-loss / force closes (market): confirmed via
    _verify_close_fill() reading the real filled_avg_price, not the quote that
    triggered the order.
  - Every state load carries a '_degraded' flag; _save_positions()/_save_weekly()
    refuse to persist (DB or JSON) while it's set, and _check_entry_conditions()
    refuses to trade on it — a stale/empty JSON fallback must never overwrite
    real Postgres state or drive a live entry decision.
  - conn.rollback() on every read-only DB function; connect_timeout=10 on the
    DB connection.
  - PYTHONUNBUFFERED is set at the Railway service level, not in code.
  - Reconcile-on-startup excludes ALL THREE strategies' legs from each other.
    credit_spread_strat.py and iron_condor_strat.py only ever hold 6-8 DTE
    positions, so they can safely treat every option leg dated today as "not
    mine" (see their `!= today_ymd` filters). This strategy's own legs ARE
    dated today by definition, so it can't use that trick — a credit_spread or
    iron_condor position opened ~7 days ago that happens to expire TODAY is
    indistinguishable from one of this strategy's own legs by date alone. See
    _other_strategy_leg_symbols() below for the expiration-filtered fix.
    credit_spread_strat.py and iron_condor_strat.py were both patched
    (2026-09-15) to also exclude this strategy's legs from their own
    reconciliation.

STRATEGY PARAMETERS — from the 2026-09-15 skew-corrected backtest
(scratch variant of archive/backtest_0dte.py; see that session's results):
  Short put      25-30 delta (vs. the archived 15-delta)
  Spread width   $3.00 (vs. the archived $2.00)
  Entry window   9:45-11:00am ET (unchanged)
  Force close    3:45pm ET same day (unchanged)
  Profit target  50% of credit (unchanged)
  Stop loss      150% of credit (unchanged)
  Max positions  2 (up from the archived 1 — "more volume" per this rebuild's brief)
  Min credit     $0.25 — backtest observed min $0.28 / mean $0.379 across 393
                 trades (2020-06 to 2026-07, skew-corrected); $0.25 sits below
                 every observed trade as a genuine sanity floor, not a binding
                 filter, following the same ~proportional-to-width convention
                 as the other two strategies ($0.25 on 7DTE's $5-wide put leg;
                 $0.10 on the archived 0DTE's $2-wide spread).
  IVR threshold  20% (unchanged from the archived constant — confirmed against
                 the backtest's own IVR filter). Methodology swapped from the
                 backtest's VIX-percentile proxy to the same real ATM-IV,
                 252-day-Parkinson-window methodology already live in
                 credit_spread_strat.py / iron_condor_strat.py (_spy_ivrank()
                 below) — identical substitution to how those two files already
                 differ from their own backtests.
  Max VIX        35% (unchanged)

Weekly loss limit is isolated: own $1,000 limit, own zero_dte_v2_state row —
no carryover from credit_spread_strat.py's or iron_condor_strat.py's tracking.

Discord: same webhook as the other two strategies, every message prefixed
[0DTE-V2] to distinguish from the retired [0DTE] naming (archive/dte0_strat.py)
and from the other two strategies' own (unprefixed / [CONDOR]) messages.

Alpaca API: raw requests only (no SDK; requirements.txt: requests==2.32.5).
  POST /v2/orders  order_class='mleg'  for multi-leg spread orders.

Deploy as a fourth Railway worker (separate service — do not add to the
existing Procfile/railway.json; configure directly in the new service):
  Start command:  python3 zero_dte_v2_strat.py
"""

import json
import math
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
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
ACTIVE = False   # Dormant by default. Set to True to enable live order placement.

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK_URL')
ALPACA_KEY      = os.getenv('ALPACA_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET')
PAPER_BASE_URL  = 'https://paper-api.alpaca.markets'
DATA_URL        = 'https://data.alpaca.markets'
DATABASE_URL    = os.getenv('DATABASE_URL')
ET              = pytz.timezone('US/Eastern')
_DIR            = os.path.dirname(os.path.abspath(__file__))

DISCORD_PREFIX  = '[0DTE-V2] '

# Strategy parameters — see module docstring for backtest provenance
TARGET_DELTA       = 0.275     # center of the 25-30 delta band
DELTA_TOLERANCE    = 0.025     # accepts |delta| in [0.250, 0.300]
SPREAD_WIDTH       = 3.0
MIN_CREDIT         = 0.25
MAX_POSITIONS      = 2
TICKERS            = ['SPY']
PROFIT_TARGET_PCT  = 0.50
STOP_LOSS_PCT      = 1.50
ORDER_FILL_TIMEOUT = 120       # shorter than 7DTE's 300s — 0DTE moves fast
WEEKLY_LOSS_LIMIT  = 1_000.0   # isolated from the other two strategies
RISK_FREE_RATE     = 0.045
VIX_IVR_WINDOW     = 252
MIN_IVR            = 20.0
MAX_VIX            = 35.0
SMA_PERIOD         = 20

# ── MACRO EVENT CALENDAR ───────────────────────────────────────────────────────
# Identical to credit_spread_strat.py's / iron_condor_strat.py's calendars — kept
# in sync manually since all three files are standalone (no shared imports).
# Update each January using official sources:
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

# Timing (ET)
ENTRY_HOUR_START, ENTRY_MIN_START  =  9, 45
ENTRY_HOUR_END,   ENTRY_MIN_END    = 11,  0
FORCE_CLOSE_HOUR, FORCE_CLOSE_MIN  = 15, 45
SUMMARY_HOUR,     SUMMARY_MIN      = 15, 50
VITALS_HOUR,      VITALS_MIN_START =  9, 30
VITALS_MIN_END                     = 34

# Files — JSON is a degraded-mode fallback only, never the primary store
POSITIONS_FILE = os.path.join(_DIR, 'zero_dte_v2_positions.json')
WEEKLY_FILE    = os.path.join(_DIR, 'zero_dte_v2_state.json')
TRADE_LOG_FILE = os.path.join(_DIR, 'zero_dte_v2_trade_log.json')
SCAN_LOG_FILE  = os.path.join(_DIR, 'zero_dte_v2_log.json')
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
_last_known_cost:     dict = {}
_monitor_consec_fail: dict = {}
_monitor_alert_sent:  set  = set()

STALE_COST_MAX_MINUTES  = 30
MONITOR_ALERT_THRESHOLD = 3


# ── DATABASE ───────────────────────────────────────────────────────────────────

_DB = None


def _get_db():
    """Return a live psycopg2 connection, or None if DATABASE_URL is not set."""
    global _DB
    if not DATABASE_URL or psycopg2 is None:
        return None
    try:
        if _DB is None or _DB.closed:
            _DB = psycopg2.connect(DATABASE_URL, connect_timeout=10)
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
            CREATE TABLE IF NOT EXISTS zero_dte_v2_positions (
                id                     SERIAL PRIMARY KEY,
                short_symbol           TEXT NOT NULL,
                long_symbol            TEXT NOT NULL,
                short_strike           DOUBLE PRECISION,
                long_strike            DOUBLE PRECISION,
                expiration             TEXT,
                credit                 DOUBLE PRECISION,
                max_risk               DOUBLE PRECISION,
                breakeven              DOUBLE PRECISION,
                profit_target          DOUBLE PRECISION,
                stop_loss_cost         DOUBLE PRECISION,
                open_time              TEXT,
                entry_order_id         TEXT,
                short_delta            DOUBLE PRECISION,
                spy_entry_px           DOUBLE PRECISION,
                reconciled             BOOLEAN DEFAULT FALSE,
                note                   TEXT,
                pending_close_order_id TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS zero_dte_v2_state (
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
        cur.execute('SELECT COUNT(*) FROM zero_dte_v2_positions')
        if cur.fetchone()[0] == 0 and os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE) as f:
                    j = json.load(f)
                if isinstance(j, dict) and j.get('positions'):
                    _db_save_positions(j)
                    print(f'[db] Migrated {len(j["positions"])} position(s) from JSON')
            except Exception as me:
                print(f'[db] JSON positions migration failed: {me}')

        cur.execute('SELECT COUNT(*) FROM zero_dte_v2_state')
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
        # "no row yet" apart from "connection failed" — same reasoning as
        # credit_spread_strat.py's identical guard.
        cur.execute("""
            INSERT INTO zero_dte_v2_state (id, weekly_realized_loss, cooldown_active, week_start_date)
            VALUES (1, 0.0, FALSE, %s)
            ON CONFLICT (id) DO NOTHING
        """, (_this_monday(),))
        conn.commit()

    except Exception as e:
        print(f'[db] _init_db failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass


def _db_load_positions():
    """Load positions list and daily_summary_sent from PostgreSQL. Returns None on failure."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT short_symbol, long_symbol, short_strike, long_strike,
                   expiration, credit, max_risk, breakeven, profit_target,
                   stop_loss_cost, open_time, entry_order_id, short_delta,
                   spy_entry_px, reconciled, note, pending_close_order_id
            FROM zero_dte_v2_positions ORDER BY id
        """)
        positions = []
        for row in cur.fetchall():
            pos = {
                'short_symbol':   row[0],
                'long_symbol':    row[1],
                'short_strike':   row[2],
                'long_strike':    row[3],
                'expiration':     row[4],
                'credit':         row[5],
                'max_risk':       row[6],
                'breakeven':      row[7],
                'profit_target':  row[8],
                'stop_loss_cost': row[9],
                'open_time':      row[10],
                'entry_order_id': row[11],
                'short_delta':    row[12],
                'spy_entry_px':   row[13],
            }
            if row[14]:
                pos['reconciled'] = True
            if row[15]:
                pos['note'] = row[15]
            if row[16]:
                pos['pending_close_order_id'] = row[16]
            positions.append(pos)

        cur.execute("""
            SELECT daily_summary_sent, morning_vitals_sent
            FROM zero_dte_v2_state WHERE id = 1
        """)
        row = cur.fetchone()
        conn.rollback()   # close out this read-only transaction
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
    """Replace all rows in zero_dte_v2_positions and update daily_summary_sent."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute('DELETE FROM zero_dte_v2_positions')
        for pos in ps.get('positions', []):
            cur.execute("""
                INSERT INTO zero_dte_v2_positions (
                    short_symbol, long_symbol, short_strike, long_strike,
                    expiration, credit, max_risk, breakeven, profit_target,
                    stop_loss_cost, open_time, entry_order_id, short_delta,
                    spy_entry_px, reconciled, note, pending_close_order_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                pos.get('short_symbol'),    pos.get('long_symbol'),
                pos.get('short_strike'),    pos.get('long_strike'),
                pos.get('expiration'),      pos.get('credit'),
                pos.get('max_risk'),        pos.get('breakeven'),
                pos.get('profit_target'),   pos.get('stop_loss_cost'),
                pos.get('open_time'),       pos.get('entry_order_id'),
                pos.get('short_delta'),     pos.get('spy_entry_px'),
                bool(pos.get('reconciled', False)),
                pos.get('note'),
                pos.get('pending_close_order_id'),
            ))
        cur.execute("""
            INSERT INTO zero_dte_v2_state (id, daily_summary_sent, morning_vitals_sent)
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
    """Return (daily_summary_sent, morning_vitals_sent) directly from DB state row.
    Returns (None, None) on DB failure or missing row."""
    conn = _get_db()
    if conn is None:
        return None, None
    try:
        cur = conn.cursor()
        cur.execute(
            'SELECT daily_summary_sent, morning_vitals_sent '
            'FROM zero_dte_v2_state WHERE id = 1'
        )
        row = cur.fetchone()
        conn.rollback()   # close out this read-only transaction
        return (row[0], row[1]) if row else (None, None)
    except Exception as e:
        print(f'[db] _db_read_summary_flags failed: {e}')
        return None, None


def _db_write_summary_flags(daily_str, vitals_str):
    """Update only the two summary-flag columns in zero_dte_v2_state."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE zero_dte_v2_state
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
    """Load weekly state from PostgreSQL. Returns None on failure or no row."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT weekly_realized_loss, cooldown_active, week_start_date
            FROM zero_dte_v2_state WHERE id = 1
        """)
        row = cur.fetchone()
        conn.rollback()   # close out this read-only transaction
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
    """Upsert weekly state in PostgreSQL."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO zero_dte_v2_state
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
    Read credit_spread_strat.py's and iron_condor_strat.py's currently-tracked
    leg symbols directly from their Postgres tables, restricted to positions
    EXPIRING TODAY.

    This is deliberately different from credit_spread_strat.py's and
    iron_condor_strat.py's own _other_strategy_leg_symbols(): those two only
    ever hold 6-8 DTE positions, so they can cheaply and safely exclude every
    option leg dated today from their own reconciliation (a today-dated leg
    can never be theirs — see their `!= today_ymd` filters). This strategy's
    own legs ARE dated today by definition, so that trick doesn't work here:
    a credit_spread or iron_condor position opened ~7 days ago that happens to
    expire TODAY produces an OCC symbol indistinguishable, by date alone, from
    one of this strategy's own legs. Without filtering those out explicitly,
    this strategy's startup reconciliation and daily-summary Alpaca cross-check
    would double-count a sibling strategy's expiring position as an
    "untracked" 0DTE leg and block new entries.

    Returns a set of symbols. Empty set (not None) on any failure — a failed
    lookup should never itself trigger a false mismatch.
    """
    conn = _get_db()
    if conn is None:
        return set()
    today_str = datetime.now(ET).date().isoformat()
    symbols = set()
    try:
        cur = conn.cursor()
        for table, cols in (
            ('credit_spread_positions', ('short_symbol', 'long_symbol')),
            ('iron_condor_positions',   ('short_put_symbol', 'long_put_symbol',
                                          'short_call_symbol', 'long_call_symbol')),
        ):
            cur.execute("SELECT to_regclass(%s)", (f'public.{table}',))
            if cur.fetchone()[0] is None:
                conn.rollback()   # close out this read-only transaction
                continue   # sibling strategy never booted
            col_list = ', '.join(cols)
            cur.execute(f"""
                SELECT {col_list} FROM {table} WHERE expiration = %s
            """, (today_str,))
            for row in cur.fetchall():
                for sym in row:
                    if sym:
                        symbols.add(sym)
            conn.rollback()   # close out this read-only transaction
        return symbols
    except Exception as e:
        print(f'[db] _other_strategy_leg_symbols failed: {e}')
        return set()


# ── MARKET HOURS / TIMING ──────────────────────────────────────────────────────

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


def _is_force_close_time():
    now = datetime.now(ET)
    return now.hour > FORCE_CLOSE_HOUR or (
        now.hour == FORCE_CLOSE_HOUR and now.minute >= FORCE_CLOSE_MIN
    )


def _is_summary_time():
    now = datetime.now(ET)
    return now.hour == SUMMARY_HOUR and now.minute >= SUMMARY_MIN


def _is_vitals_window():
    now = datetime.now(ET)
    return now.hour == VITALS_HOUR and VITALS_MIN_START <= now.minute <= VITALS_MIN_END


# ── DISCORD ────────────────────────────────────────────────────────────────────

def _discord(msg):
    """Rate-limit-aware Discord post, prefixed [0DTE-V2] to distinguish from the
    retired [0DTE] naming and from the other two strategies' own messages.
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
    """Permanent trade history — never truncated. Lives on Railway's ephemeral
    container filesystem, NOT synced to Postgres; resets on redeploy. Alpaca's
    own order/activity history is the durable source of truth for closed
    trades, same caveat as the other two strategies."""
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


# ── POSITIONS STATE ────────────────────────────────────────────────────────────
# zero_dte_v2_positions (Postgres, primary) / zero_dte_v2_positions.json (fallback)
#
# '_degraded' marker (same fix as credit_spread_strat.py / iron_condor_strat.py,
# 2026-08-26 incident): a Postgres outage can make _load_positions() fall back
# to a LOCAL JSON file that's only ever updated when a *write* fails — frozen
# at container-boot state (empty) the whole time DB writes had been
# succeeding. _save_positions() is called unconditionally from
# _send_morning_vitals / _check_daily_summary / _attempt_entry regardless of
# whether the in-memory state was actually complete. The moment Postgres
# recovers and one of those unconditional saves fires, it would overwrite real
# Postgres rows with that stale/empty fallback via the full DELETE+reinsert in
# _db_save_positions(). Fix: every pos_state dict now carries a '_degraded'
# flag set by _load_positions() — True means "this came from the JSON
# fallback because a DB read that should have succeeded didn't" —
# _save_positions() refuses to persist anything (DB or JSON) while that flag
# is set, and _check_entry_conditions() refuses to trade on it.

_db_degraded_alert_sent = False   # dedup: alert once per outage, not every scan


def _alert_db_degraded(context):
    global _db_degraded_alert_sent
    if _db_degraded_alert_sent:
        return
    _db_degraded_alert_sent = True
    msg = (f'🚨 DATABASE READ FAILED ({context}) — operating in degraded mode. '
           f'Falling back to local JSON, which may be stale. All saves are now '
           f'BLOCKED until Postgres recovers, to avoid overwriting real data with '
           f'a stale fallback. Investigate DATABASE_URL / Postgres health now.')
    print(f'  [db] {msg}')
    _discord(msg)


def _load_positions():
    global _db_degraded_alert_sent
    if DATABASE_URL:
        result = _db_load_positions()
        if result is not None:
            _db_degraded_alert_sent = False   # DB healthy again — re-arm the alert
            result['_degraded'] = False
            return result
        print('[db] _load_positions: DB read failed, falling back to JSON — '
              'saves will be blocked until DB recovers')
        _alert_db_degraded('positions')
    try:
        with open(POSITIONS_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'positions' in d:
            d['_degraded'] = bool(DATABASE_URL)
            return d
    except Exception:
        pass
    return {'positions': [], 'daily_summary_sent': None, '_degraded': bool(DATABASE_URL)}


def _save_positions(ps):
    if ps.get('_degraded'):
        msg = ('🚨 REFUSING TO SAVE positions — state was loaded from the degraded/'
               'stale-JSON fallback (Postgres read failed), not from Postgres itself. '
               'Writing it now would overwrite real data. Not saving to DB or JSON; '
               'will retry once a healthy DB read repopulates the in-memory state.')
        print(f'  [save_positions] {msg}')
        _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
              'event': 'SAVE_REFUSED_DEGRADED_STATE'})
        _discord(msg)
        return
    if DATABASE_URL:
        if _db_save_positions(ps):
            return
        print('[db] _save_positions: DB write failed, falling back to JSON')
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(ps, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_save_positions] failed: {e}')


# ── WEEKLY STATE ───────────────────────────────────────────────────────────────
# zero_dte_v2_state — isolated from credit_spread_strat.py's and
# iron_condor_strat.py's own weekly tracking. Own $1,000 limit, own row.

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
    global _db_degraded_alert_sent
    if DATABASE_URL:
        result = _db_load_weekly()
        if result is not None:
            _db_degraded_alert_sent = False   # DB healthy again — re-arm the alert
            result['_degraded'] = False
            return result
        print('[db] _load_weekly: DB read failed, falling back to JSON — '
              'saves will be blocked until DB recovers')
        _alert_db_degraded('weekly')
    try:
        with open(WEEKLY_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and 'week_start_date' in d:
            d['_degraded'] = bool(DATABASE_URL)
            return d
    except Exception:
        pass
    empty = _empty_weekly()
    empty['_degraded'] = bool(DATABASE_URL)
    return empty


def _save_weekly(w):
    if w.get('_degraded'):
        msg = ('🚨 REFUSING TO SAVE weekly state — state was loaded from the degraded/'
               'stale-JSON fallback (Postgres read failed), not from Postgres itself. '
               'Writing it now would overwrite real data. Not saving to DB or JSON; '
               'will retry once a healthy DB read repopulates the in-memory state.')
        print(f'  [save_weekly] {msg}')
        _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
              'event': 'SAVE_REFUSED_DEGRADED_STATE'})
        _discord(msg)
        return
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
        # Carry the '_degraded' flag through explicitly — _empty_weekly() builds
        # a fresh dict with no '_degraded' key at all, which would read as falsy
        # and bypass the refuse-to-save guard in _save_weekly() below.
        degraded = w.get('_degraded', False)
        w = _empty_weekly()
        w['_degraded'] = degraded
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
            continue  # DB is the primary store for these two files
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(empty, f, indent=2)
            print(f'[init] Created {os.path.basename(path)}')


# ── STARTUP RECONCILIATION ─────────────────────────────────────────────────────

def _reconcile_on_startup():
    """
    Load state, cross-check against Alpaca open SPY option legs dated today,
    and return (pos_state, weekly).

    Unlike credit_spread_strat.py / iron_condor_strat.py (which exclude every
    today-dated leg outright), this strategy's own legs ARE dated today, so it
    must instead exclude the specific legs belonging to a sibling strategy's
    position that happens to expire today — see _other_strategy_leg_symbols().
    """
    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)
    now_str   = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')

    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is None:
            raise RuntimeError('positions fetch returned None after retries')
        yy_mm_dd   = datetime.now(ET).strftime('%y%m%d')
        other_legs = _other_strategy_leg_symbols()
        option_legs = [
            p for p in r.json()
            if p.get('asset_class') == 'us_option'
            and str(p.get('symbol', '')).startswith(f'SPY{yy_mm_dd}')
            and p.get('symbol', '') not in other_legs
        ]

        # Resolve positions whose profit-target close order filled while the
        # process was down. Must happen before expected_legs is computed —
        # same ordering rationale as credit_spread_strat.py's 2026-08-16 fix.
        resolved_pending = []
        for pos in pos_state.get('positions', []):
            pending_id = pos.get('pending_close_order_id')
            if not pending_id:
                continue
            order = _get_order(pending_id)
            if order and order.get('status') == 'filled':
                fill_cost = float(order['filled_avg_price'])
                label = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'
                print(f'[reconcile] Pending close {pending_id[:8]}… confirmed filled — recording exit')
                _log({'timestamp': now_str, 'event': 'PROFIT_TARGET_CLOSE_CONFIRMED_ON_RECONCILE',
                      'label': label, 'order_id': pending_id, 'fill_cost': fill_cost})
                _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', fill_cost, now_str)
                resolved_pending.append(pos)
        for pos in resolved_pending:
            try:
                pos_state['positions'].remove(pos)
            except ValueError:
                pass
        if resolved_pending:
            _save_positions(pos_state)
            _save_weekly(weekly)

        expected_legs = len(pos_state.get('positions', [])) * 2
        actual_legs   = len(option_legs)

        if actual_legs == expected_legs:
            print(f'[reconcile] Alpaca 0DTE-V2 OK — '
                  f'{actual_legs} leg(s) / {len(pos_state["positions"])} spread(s)')
        else:
            orphaned = actual_legs - expected_legs
            direction = f'{orphaned} untracked leg(s) — new entries blocked' if orphaned > 0 \
                        else f'{-orphaned} extra state entry(ies) — stale entries possible'
            msg = (
                f'⚠️ OPTIONS MISMATCH on startup | '
                f'Alpaca: {actual_legs} 0DTE option leg(s) (excluding sibling-strategy '
                f'expiring legs), state file expects {expected_legs} '
                f'({len(pos_state["positions"])} spread(s)). '
                f'{direction}. Manual review required.'
            )
            print(f'[reconcile] {msg}')
            _log({'timestamp': now_str, 'event': 'RECONCILE_OPTIONS_MISMATCH',
                  'alpaca_legs': actual_legs, 'expected_legs': expected_legs,
                  'alpaca_symbols': [p['symbol'] for p in option_legs]})
            _discord(msg)

            for _ in range(max(0, orphaned // 2)):
                pos_state['positions'].append({
                    'short_symbol':   'UNKNOWN',
                    'long_symbol':    'UNKNOWN',
                    'short_strike':   0.0,
                    'long_strike':    0.0,
                    'expiration':     '2099-01-01',
                    'credit':         0.0,
                    'max_risk':       0.0,
                    'breakeven':      0.0,
                    'profit_target':  0.0,
                    'stop_loss_cost': 999.0,
                    'open_time':      now_str,
                    'entry_order_id': None,
                    'short_delta':    None,
                    'spy_entry_px':   None,
                    'reconciled':     True,
                    'note':           'Untracked Alpaca position — manual close required',
                })
            if orphaned > 0:
                _save_positions(pos_state)

    except Exception as e:
        print(f'[reconcile] Alpaca positions check failed (startup continues): {e}')
        _log({'timestamp': now_str, 'event': 'RECONCILE_API_ERROR', 'error': str(e)})

    positions = pos_state.get('positions', [])
    tracked   = [p for p in positions if not p.get('reconciled')]
    if positions:
        labels = [
            f'{p.get("short_strike", 0):.0f}/{p.get("long_strike", 0):.0f}P '
            f'exp={p.get("expiration", "?")}  credit=${p.get("credit", 0):.2f}'
            + (' [UNTRACKED]' if p.get('reconciled') else '')
            for p in positions
        ]
        print(f'[startup] Resumed {len(tracked)} tracked + '
              f'{len(positions) - len(tracked)} untracked position(s):')
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


# ── MARKET DATA ────────────────────────────────────────────────────────────────

def _flatten_columns(df):
    """Handle MultiIndex columns returned by yfinance >=0.2 for single-ticker downloads."""
    if hasattr(df.columns, 'levels'):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass
    return df


def _above_sma20():
    """Return (above_sma: bool|None, close: float|None, sma: float|None)."""
    try:
        df = yf.download('SPY', period='40d', interval='1d',
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


def _spy_ivrank():
    """
    Return (ivr_pct: float|None, current_iv_pct: float|None).
    IVR = percentile of current SPY ATM IV within 252-day Parkinson vol range.
    Identical implementation to credit_spread_strat.py._spy_ivrank() /
    iron_condor_strat.py._spy_ivrank(). Falls back to _vix_ivrank() if the SPY
    options fetch fails.
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
            params={'symbols': sym}   # feed omitted — Alpaca defaults to opra if subscribed, else indicative
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
                current_iv = _bs_iv_solve(S, K, T, (bid + ask) / 2)
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


# ── BLACK-SCHOLES HELPERS ──────────────────────────────────────────────────────

def _0dte_T_years():
    """
    Remaining fraction of trading year for a 0DTE option right now.
    Wall-clock hours remaining until 4:00pm ET, divided by
    (6.5 trading hours × 252 trading days). Floored at ~1 minute.
    """
    now_et   = datetime.now(ET)
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    remaining_hours = max((close_et - now_et).total_seconds() / 3600.0, 1 / 60)
    return remaining_hours / (6.5 * 252)


def _bs_put_delta(S, K, T_years, sigma):
    """Black-Scholes European put delta in [-1, 0]. Returns None on error."""
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return -1.0 if K > S else 0.0
        d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T_years) / (
            sigma * math.sqrt(T_years)
        )
        return norm.cdf(d1) - 1.0
    except Exception:
        return None


def _bs_iv_solve(S, K, T_years, mkt_price):
    """Bisection IV solver from observed put price. Returns decimal IV (e.g. 0.13)."""
    if T_years <= 0 or mkt_price <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 0.001, 5.0
    r = RISK_FREE_RATE
    for _ in range(80):
        mid = (lo + hi) / 2
        d1  = (math.log(S / K) + (r + 0.5 * mid ** 2) * T_years) / (mid * math.sqrt(T_years))
        d2  = d1 - mid * math.sqrt(T_years)
        p   = K * math.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
        if p < mkt_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


# ── OPTIONS CHAIN ──────────────────────────────────────────────────────────────

def _find_today_expiration():
    """Return today's date if SPY has 0DTE options today, else None."""
    try:
        today = datetime.now(ET).date()
        if today.weekday() >= 5:
            return None
        expirations = yf.Ticker('SPY').options
        if today.isoformat() in expirations:
            return today
        # yfinance sometimes omits today pre-open — check Alpaca directly
        r = _alpaca_get(
            f'{PAPER_BASE_URL}/v2/options/contracts',
            headers=_headers(),
            params={
                'underlying_symbol': 'SPY',
                'type':              'put',
                'expiration_date':   today.isoformat(),
                'limit': 1,
                'status': 'active',
            }
        )
        if r and r.json().get('option_contracts'):
            return today
        return None
    except Exception as e:
        print(f'  [find_0dte_expiry] {e}')
        return None


def _fetch_0dte_chain(expiry, now_str):
    """
    Fetch the full SPY 0DTE put chain for today.
    Contracts: paper-api.alpaca.markets/v2/options/contracts (trading API)
    Snapshots: data.alpaca.markets/v1beta1/options/snapshots (market data API)
    Returns dict {symbol: {strike, bid, ask, mid, delta, iv}} or None on failure.
    """
    try:
        r = _alpaca_get(
            f'{PAPER_BASE_URL}/v2/options/contracts',
            headers=_headers(),
            params={
                'underlying_symbols': 'SPY',
                'type':               'put',
                'expiration_date':    expiry.isoformat(),
                'limit':              200,
            },
            timeout=15,
        )
        if r is None:
            reason = 'contracts fetch returned None after retries'
            print(f'  [0dte chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR', 'reason': reason})
            return None
        contracts = r.json().get('option_contracts', [])
        if not contracts:
            reason = f'no contracts returned by Alpaca for expiry {expiry}'
            print(f'  [0dte chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry),
                  'reason': reason})
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
                    params={'symbols': ','.join(batch)},   # feed omitted — Alpaca defaults to opra if subscribed, else indicative
                    timeout=15,
                )
                if rs is None:
                    reason = f'snapshots batch failed after retries (offset {batch_start})'
                    print(f'  [0dte chain] {reason}')
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
                    }
            except Exception as e:
                reason = f'snapshot batch error: {type(e).__name__}: {e}'
                print(f'  [0dte chain] {reason}')
                _log({'timestamp': now_str, 'event': 'CHAIN_BATCH_ERROR', 'reason': reason})

        if not chain:
            reason = 'all snapshot batches returned empty'
            print(f'  [0dte chain] {reason}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry),
                  'reason': reason})
            return None

        return chain

    except Exception as e:
        reason = f'{type(e).__name__}: {e}'
        print(f'  [0dte chain] fetch failed: {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR', 'expiry': str(expiry),
              'reason': reason})
        return None


def _find_short_strike(chain, spy_px, vix, now_str):
    """
    Find OTM put with |delta| closest to TARGET_DELTA (0.275, i.e. the 25-30
    delta band via DELTA_TOLERANCE=0.025).

    Delta priority per contract:
      1. Alpaca greeks.delta  (most accurate)
      2. Per-contract impliedVolatility via Black-Scholes with intraday T
      3. VIX / 100 via Black-Scholes (last resort)

    Rejects result if best |delta| is > DELTA_TOLERANCE from target.
    Returns (symbol, strike, delta) or (None, None, None).
    """
    T_years        = _0dte_T_years()
    sigma_fallback = (vix or 20.0) / 100.0

    best_sym, best_strike, best_delta = None, None, None
    best_diff = float('inf')
    bs_used = bs_failed = 0

    for sym, data in chain.items():
        strike = data['strike']
        if strike <= 0 or data['mid'] <= 0 or strike >= spy_px:
            continue  # skip zero-mid and ITM puts

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
        print(f'  [0dte chain] BS fallback: {bs_used} used, {bs_failed} failed')

    if best_sym is None:
        reason = 'no OTM puts with computable delta in chain'
        print(f'  [0dte chain] {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_NO_DELTA', 'reason': reason})
        return None, None, None

    if best_diff > DELTA_TOLERANCE:
        reason = (f'closest delta={abs(best_delta):.3f} at ${best_strike:.0f} is '
                  f'{best_diff:.3f} from target {TARGET_DELTA} '
                  f'(tolerance {DELTA_TOLERANCE})')
        print(f'  [0dte chain] {reason}')
        _log({'timestamp': now_str, 'event': 'CHAIN_DELTA_TOO_FAR',
              'reason': reason, 'best_strike': best_strike,
              'best_delta': round(best_delta, 4)})
        return None, None, None

    return best_sym, best_strike, best_delta


def _find_long_symbol(chain, short_strike):
    """Find option symbol for the long leg (short_strike − SPREAD_WIDTH = $3 lower)."""
    target    = short_strike - SPREAD_WIDTH
    best_sym  = best_k = None
    best_diff = float('inf')
    for sym, data in chain.items():
        diff = abs(data['strike'] - target)
        if diff < best_diff:
            best_diff, best_sym, best_k = diff, sym, data['strike']
    return best_sym, best_k


def _spread_mid(chain, short_sym, long_sym):
    """Net credit = short_mid − long_mid. Returns None if data missing or degenerate."""
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
    Current debit to close = short_mid − long_mid.
    Positive = we pay to close (normal for a short spread).
    Returns 0.0 when both legs are zero-priced (deeply OTM near expiry — valid,
    means max profit reached). Returns None only on hard API failures.
    """
    ts = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    try:
        rs = None
        for _outer in range(2):
            rs = _alpaca_get(
                f'{DATA_URL}/v1beta1/options/snapshots',
                headers=_data_headers(),
                params={'symbols': f'{short_sym},{long_sym}'},   # feed omitted — Alpaca defaults to opra if subscribed, else indicative
            )
            if rs is not None:
                break
            if _outer == 0:
                time.sleep(5)   # one 5s pause before second full attempt
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
# Multi-leg spread orders: POST /v2/orders with order_class='mleg'.

def _place_open_order(short_sym, long_sym, credit):
    """Limit order to open the credit spread. ACTIVE-gated."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          'limit',
            'time_in_force': 'day',
            'order_class':   'mleg',
            'limit_price':   str(round(-credit, 2)),   # negative = credit (Alpaca mleg convention)
            'legs': [
                {'symbol': short_sym, 'side': 'sell',
                 'ratio_qty': '1', 'position_intent': 'sell_to_open'},
                {'symbol': long_sym,  'side': 'buy',
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


def _place_close_order(short_sym, long_sym, order_type='market', limit_price=None):
    """Order to close the spread. ACTIVE-gated. order_type: 'market' | 'limit'."""
    if not ACTIVE:
        return None
    try:
        payload = {
            'qty':           '1',
            'type':          order_type,
            'time_in_force': 'day',
            'order_class':   'mleg',
            'legs': [
                {'symbol': short_sym, 'side': 'buy',
                 'ratio_qty': '1', 'position_intent': 'buy_to_close'},
                {'symbol': long_sym,  'side': 'sell',
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


def _verify_close_fill(order_id, quoted_cost, label, now_str):
    """After a market close fills, pull the real filled_avg_price instead of
    trusting the pre-order quoted cost — market orders can slip between the
    quote that triggered the order and the actual fill. Falls back to the
    quoted cost (with an explicit log entry) if the fill can't be confirmed
    within a few seconds."""
    for _ in range(3):
        order = _get_order(order_id)
        if order and order.get('status') == 'filled':
            raw = order.get('filled_avg_price')
            if raw is not None:
                return abs(float(raw))
            break
        time.sleep(2)
    print(f'  {label}: could not verify market close fill — using quoted cost ${quoted_cost:.4f}')
    _log({'timestamp': now_str, 'event': 'CLOSE_FILL_UNVERIFIED',
          'label': label, 'order_id': order_id, 'quoted_cost': quoted_cost})
    return quoted_cost


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


# ── ENTRY EXECUTION ────────────────────────────────────────────────────────────

def _attempt_entry(pos_state, weekly, now_str, short_sym, long_sym,
                   short_strike, long_strike, expiry, credit, spy_px, short_delta):
    """
    Place opening order, poll for fill up to ORDER_FILL_TIMEOUT seconds.
    Updates pos_state['positions'] in place on fill — fill is only ever
    confirmed via a polled status == 'filled' read of filled_avg_price, never
    assumed from the submitted order. Returns True if filled. ACTIVE-gated.
    """
    if not ACTIVE:
        return False

    order_id = _place_open_order(short_sym, long_sym, credit)
    if not order_id:
        _log({'timestamp': now_str, 'event': 'ORDER_PLACE_FAILED',
              'short': short_sym, 'long': long_sym, 'credit': credit})
        print('  [entry] order placement failed')
        return False

    print(f'  [entry] order {order_id} placed — polling fill (max {ORDER_FILL_TIMEOUT}s)…')
    deadline = time.time() + ORDER_FILL_TIMEOUT

    while time.time() < deadline:
        time.sleep(15)
        order = _get_order(order_id)
        if order is None:
            continue
        status = order.get('status', '')

        if status == 'filled':
            raw_price   = float(order.get('filled_avg_price') or credit)
            fill_credit = abs(raw_price)   # Alpaca returns negative for net credit received
            if raw_price < 0:
                print(f'  [entry] filled_avg_price was negative ({raw_price}) — using abs()')
            max_risk    = round((SPREAD_WIDTH - fill_credit) * 100, 2)
            breakeven   = round(short_strike - fill_credit, 2)
            pos = {
                'short_symbol':   short_sym,
                'long_symbol':    long_sym,
                'short_strike':   short_strike,
                'long_strike':    long_strike,
                'expiration':     expiry.isoformat(),
                'credit':         round(fill_credit, 4),
                'max_risk':       max_risk,
                'breakeven':      breakeven,
                'profit_target':  round(fill_credit * PROFIT_TARGET_PCT, 4),
                'stop_loss_cost': round(fill_credit * STOP_LOSS_PCT, 4),
                'open_time':      now_str,
                'entry_order_id': order_id,
                'short_delta':    round(short_delta, 4) if short_delta else None,
                'spy_entry_px':   spy_px,
            }
            pos_state['positions'].append(pos)
            _save_positions(pos_state)
            _log({'timestamp': now_str, 'event': 'ENTRY_FILLED', **pos})
            _log_trade({'timestamp': now_str, 'type': 'OPEN', **pos})
            _discord(
                f'🟢 SPREAD OPEN | '
                f'SPY {short_strike:.0f}/{long_strike:.0f}P  exp {expiry} | '
                f'Credit: ${fill_credit:.2f} | Max risk: ${max_risk:.2f} | '
                f'Breakeven: ${breakeven:.2f} | Delta: {short_delta:.3f} | '
                f'Positions open: {len(pos_state["positions"])}/{MAX_POSITIONS}'
            )
            print(f'  [entry] FILLED ${fill_credit:.4f}  {short_sym} / {long_sym}')
            return True

        if status == 'partially_filled':
            print(f'  [entry] partial fill detected on multi-leg order — cancelling {order_id}')
            _log({'timestamp': now_str, 'event': 'ORDER_PARTIAL_FILL', 'order_id': order_id})
            _cancel_order(order_id)
            _discord(
                f'⚠️ Partial fill on spread entry order — cancelled. '
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

def _record_exit(pos_state, weekly, pos, reason, close_cost, now_str):
    """
    Log exit, update weekly_realized_loss, fire Discord.
    Does NOT remove pos from pos_state — caller handles that and saves both.
    """
    credit  = pos['credit']
    pnl     = round((credit - close_cost) * 100, 2)
    label   = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'

    old_loss = weekly.get('weekly_realized_loss', 0.0)
    new_loss = round(max(0.0, old_loss - pnl), 2)
    weekly['weekly_realized_loss'] = new_loss

    exit_entry = {
        'timestamp':            now_str,
        'type':                 'CLOSE',
        'event':                'EXIT',
        'reason':               reason,
        'label':                label,
        'short_symbol':         pos['short_symbol'],
        'long_symbol':          pos['long_symbol'],
        'expiration':           pos['expiration'],
        'credit':               credit,
        'close_cost':           close_cost,
        'pnl':                  pnl,
        'weekly_realized_loss': new_loss,
    }
    _log(exit_entry)
    _log_trade(exit_entry)

    reason_labels = {
        'PROFIT_TARGET': 'profit target',
        'STOP_LOSS':     'stop loss',
        'FORCE_CLOSE':   'force close 3:45pm',
    }
    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
    _discord(
        f'{"✅" if pnl >= 0 else "🔴"} SPREAD CLOSED '
        f'({reason_labels.get(reason, reason)}) | '
        f'{label} | '
        f'Credit: ${credit:.2f}  Close: ${close_cost:.4f} | '
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
    Check all open positions for profit target / stop loss / 3:45pm force
    close. ACTIVE-gated. Modifies pos_state['positions'] in place; saves both
    files on any change.

    Force close takes priority over a still-pending profit-target limit order,
    same way a fresh stop-loss reading does — both cancel the pending limit
    and place a market order instead.
    """
    if not ACTIVE:
        return

    positions = pos_state.get('positions', [])
    if not positions:
        return

    to_close          = []
    positions_updated = False

    for pos in positions:
        if pos.get('reconciled') and pos.get('short_symbol') == 'UNKNOWN':
            print(f'  [monitor] Skipping untracked reconciled position — manual review required')
            continue

        short_sym = pos['short_symbol']
        long_sym  = pos['long_symbol']
        credit    = pos['credit']
        label     = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'

        try:
            # ── Check pending profit-target close order ────────────────────────
            pending_id = pos.get('pending_close_order_id')
            if pending_id:
                order    = _get_order(pending_id)
                o_status = order.get('status', '') if order else ''

                if o_status == 'filled':
                    fill_cost = float(order['filled_avg_price'])
                    print(f'  {label}: pending close CONFIRMED FILLED (cost=${fill_cost:.4f})')
                    _log({'timestamp': now_str, 'event': 'PROFIT_TARGET_CLOSE_CONFIRMED',
                          'label': label, 'order_id': pending_id, 'fill_cost': fill_cost})
                    _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', fill_cost, now_str)
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
                    curr_cost = _current_cost_to_close(short_sym, long_sym) or pos['profit_target']
                    retry_ok  = _place_close_order(short_sym, long_sym, order_type='market')
                    if retry_ok:
                        fill_cost = _verify_close_fill(retry_ok, curr_cost, label, now_str)
                        _record_exit(pos_state, weekly, pos, 'PROFIT_TARGET', fill_cost, now_str)
                        to_close.append(pos)
                    else:
                        _discord(
                            f'🚨 MARKET RETRY ALSO FAILED | {label} | '
                            f'Could not close after limit expiry. Manual close required in Alpaca.'
                        )
                    continue

                # Force close overrides a still-pending profit-target order
                if _is_force_close_time():
                    print(f'  {label}: FORCE CLOSE overrides pending profit-target order')
                    _cancel_order(pending_id)
                    pos['pending_close_order_id'] = None
                    positions_updated = True
                    cost = _current_cost_to_close(short_sym, long_sym) or 0.0
                    ok   = _place_close_order(short_sym, long_sym, order_type='market')
                    if ok:
                        fill_cost = _verify_close_fill(ok, cost, label, now_str)
                        _record_exit(pos_state, weekly, pos, 'FORCE_CLOSE', fill_cost, now_str)
                        to_close.append(pos)
                    else:
                        _discord(
                            f'🚨 FORCE CLOSE FAILED (after cancelling pending profit-target) | '
                            f'{label} | Manual close required in Alpaca immediately.'
                        )
                    continue

                # Order still open/pending — check stop-loss override before waiting
                curr_cost_check = _current_cost_to_close(short_sym, long_sym)
                if curr_cost_check is not None and curr_cost_check >= pos['stop_loss_cost']:
                    print(f'  {label}: STOP LOSS overrides pending profit-target close')
                    _cancel_order(pending_id)
                    pos['pending_close_order_id'] = None
                    positions_updated = True
                    ok = _place_close_order(short_sym, long_sym, order_type='market')
                    if ok:
                        fill_cost = _verify_close_fill(ok, curr_cost_check, label, now_str)
                        _record_exit(pos_state, weekly, pos, 'STOP_LOSS', fill_cost, now_str)
                        to_close.append(pos)
                    else:
                        _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                              'label': label, 'reason': 'STOP_LOSS'})
                        _discord(f'⚠️ Close order failed | {label} | Stop loss | Retrying next scan')
                    continue

                if order is None:
                    print(f'  {label}: pending close {pending_id[:8]}… — order query failed, skipping')
                else:
                    print(f'  {label}: pending close {pending_id[:8]}… still {o_status} — waiting')
                continue

            # ── 3:45pm force close ──────────────────────────────────────────────
            if _is_force_close_time():
                print(f'  {label}: FORCE CLOSE (3:45pm)')
                cost = _current_cost_to_close(short_sym, long_sym)
                if cost is None:
                    cost = 0.0
                    print(f'  {label}: cost unavailable for force close — using 0.0')
                ok = _place_close_order(short_sym, long_sym, order_type='market')
                if ok:
                    fill_cost = _verify_close_fill(ok, cost, label, now_str)
                    _record_exit(pos_state, weekly, pos, 'FORCE_CLOSE', fill_cost, now_str)
                    to_close.append(pos)
                else:
                    print(f'  {label}: FORCE CLOSE ORDER FAILED')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'FORCE_CLOSE'})
                    _discord(
                        f'🚨 FORCE CLOSE FAILED | {label} | '
                        f'Manual close required in Alpaca immediately — expires today.'
                    )
                continue

            # ── Current spread value ──────────────────────────────────────────
            _cost_key = (short_sym, long_sym)
            cost = _current_cost_to_close(short_sym, long_sym)
            if cost is None:
                fail_count = _monitor_consec_fail.get(_cost_key, 0) + 1
                _monitor_consec_fail[_cost_key] = fail_count
                cached = _last_known_cost.get(_cost_key)
                stale_cost = stale_age_min = None
                if cached:
                    stale_age_min = (datetime.now(ET) - cached['ts']).total_seconds() / 60
                    if stale_age_min <= STALE_COST_MAX_MINUTES:
                        stale_cost = cached['cost']
                age_str = f'{stale_age_min:.0f}min ago' if stale_age_min is not None else 'never'
                print(f'  {label}: value unavailable — skipping this cycle '
                      f'({fail_count} consecutive; last good: {age_str})')
                _log({'timestamp': now_str, 'event': 'MONITOR_VALUE_UNAVAILABLE',
                      'label': label, 'consecutive_failures': fail_count,
                      'stale_cost': stale_cost,
                      'stale_age_min': round(stale_age_min, 1) if stale_age_min is not None else None})
                if fail_count >= MONITOR_ALERT_THRESHOLD and _cost_key not in _monitor_alert_sent:
                    _monitor_alert_sent.add(_cost_key)
                    _discord(
                        f'⚠️ STOP-LOSS MONITORING DEGRADED | {label} | '
                        f'{fail_count} consecutive pricing failures. '
                        f'Last good price: {age_str}. '
                        f'Stop at ${pos["stop_loss_cost"]:.2f} (150% of credit ${credit:.2f}) '
                        f'cannot be auto-enforced. Manual monitoring required.'
                    )
                if stale_cost is not None and stale_cost >= pos['stop_loss_cost']:
                    print(f'  {label}: STALE-PRICE STOP LOSS '
                          f'(cached {stale_cost:.4f} from {stale_age_min:.0f}min ago '
                          f'>= stop {pos["stop_loss_cost"]:.4f})')
                    _log({'timestamp': now_str, 'event': 'MONITOR_STALE_STOP_LOSS',
                          'label': label, 'stale_cost': stale_cost,
                          'stale_age_min': round(stale_age_min, 1)})
                    _discord(
                        f'⚠️ STOP-LOSS TRIGGERED ON STALE PRICE | {label} | '
                        f'Live quotes unavailable; cached cost ${stale_cost:.4f} '
                        f'({stale_age_min:.0f}min old) ≥ stop ${pos["stop_loss_cost"]:.4f}. '
                        f'Placing market close order.'
                    )
                    ok = _place_close_order(short_sym, long_sym, order_type='market')
                    if ok:
                        fill_cost = _verify_close_fill(ok, stale_cost, label, now_str)
                        _record_exit(pos_state, weekly, pos, 'STOP_LOSS', fill_cost, now_str)
                        to_close.append(pos)
                    else:
                        _discord(
                            f'🚨 STALE-PRICE STOP LOSS ORDER FAILED | {label} | '
                            f'Manual close required immediately.'
                        )
                continue

            _last_known_cost[_cost_key] = {'cost': cost, 'ts': datetime.now(ET)}
            _monitor_consec_fail[_cost_key] = 0
            _monitor_alert_sent.discard(_cost_key)

            print(f'  {label}: cost={cost:.4f}  credit={credit:.4f}  '
                  f'tgt≤{pos["profit_target"]:.4f}  stop≥{pos["stop_loss_cost"]:.4f}')

            # ── Profit target: cost ≤ 50% of original credit ─────────────────
            if cost <= pos['profit_target']:
                print(f'  {label}: PROFIT TARGET')
                order_id = _place_close_order(short_sym, long_sym,
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

            # ── Stop loss: cost ≥ 150% of original credit ────────────────────
            if cost >= pos['stop_loss_cost']:
                print(f'  {label}: STOP LOSS')
                ok = _place_close_order(short_sym, long_sym, order_type='market')
                if ok:
                    fill_cost = _verify_close_fill(ok, cost, label, now_str)
                    _record_exit(pos_state, weekly, pos, 'STOP_LOSS', fill_cost, now_str)
                    to_close.append(pos)
                else:
                    print(f'  {label}: stop-loss close failed — retrying next cycle')
                    _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                          'label': label, 'reason': 'STOP_LOSS'})
                    _discord(f'⚠️ Close order failed | {label} | Stop loss | Retrying next scan')

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
    """
    Return an event name string if today (ET) is a blocked macro event day, else None.
    Checked before all other entry conditions — no API calls required.
    """
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

_entry_blocked_alert_sent = False


def _check_entry_conditions(pos_state, weekly):
    """
    Evaluate all gates cheapest-first. Short-circuits on first group failure.
    Returns (all_passed: bool, conditions: dict, spy_px, ivr, vix).
    """
    global _entry_blocked_alert_sent
    conds  = {}
    spy_px = ivr = vix = None

    def _c(name, passed, detail):
        conds[name] = {'passed': bool(passed), 'detail': str(detail)}

    # Hard guard: never evaluate or pass ANY gate on state loaded via the
    # degraded fallback path.
    if pos_state.get('_degraded') or weekly.get('_degraded'):
        _c('degraded_state', False,
           'pos_state or weekly loaded from degraded fallback — refusing to trade')
        if not _entry_blocked_alert_sent:
            _entry_blocked_alert_sent = True
            msg = ('🚨 ENTRY BLOCKED — degraded DB state, refusing to trade on '
                   'unreliable position/cooldown data.')
            print(f'  [entry] {msg}')
            _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
                  'event': 'ENTRY_BLOCKED_DEGRADED_STATE'})
            _discord(msg)
        return False, conds, spy_px, ivr, vix
    _entry_blocked_alert_sent = False

    macro = _macro_event_today()
    _c('macro_event', macro is None, macro or 'none')
    if macro:
        return False, conds, spy_px, ivr, vix

    _c('active',          ACTIVE,
       'True' if ACTIVE else 'False — DORMANT')
    _c('entry_window',    _in_entry_window(),
       '9:45–11:00 ET')
    _c('position_limit',  len(pos_state.get('positions', [])) < MAX_POSITIONS,
       f'{len(pos_state.get("positions", []))}/{MAX_POSITIONS} open')
    _c('weekly_cooldown', not weekly.get('cooldown_active', False),
       f'loss=${weekly.get("weekly_realized_loss", 0):.2f} / ${WEEKLY_LOSS_LIMIT:.0f}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, spy_px, ivr, vix

    # Underwater position gate — don't add a second spread if the first is at a loss
    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    if len(real_pos) == 1:
        p    = real_pos[0]
        cost = _current_cost_to_close(p['short_symbol'], p['long_symbol'])
        if cost is not None and cost > p.get('credit', 0):
            _c('underwater_block', False,
               f'cost={cost:.4f} > credit={p.get("credit", 0):.4f}')
            return False, conds, spy_px, ivr, vix

    above_sma, spy_close, sma_val = _above_sma20()
    spy_px = spy_close
    if above_sma is None:
        _c('spy_above_sma', False, 'data unavailable')
    else:
        _c('spy_above_sma', above_sma,
           f'SPY={spy_close:.2f}  SMA20={sma_val:.2f}  {"above" if above_sma else "BELOW"}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, spy_px, ivr, vix

    ivr, vix = _spy_ivrank()
    if ivr is None:
        _c('iv_rank', False, 'SPY IV data unavailable')
        _c('vix_cap',  False, 'SPY IV data unavailable')
    else:
        _c('iv_rank', ivr >= MIN_IVR,
           f'IVR={ivr:.1f}% (need ≥{MIN_IVR}%) — SPY IV={vix:.1f}%')
        _c('vix_cap',  vix < MAX_VIX,
           f'SPY IV={vix:.1f}% {"<" if vix < MAX_VIX else "≥"} {MAX_VIX:.0f}%')

    return all(v['passed'] for v in conds.values()), conds, spy_px, ivr, vix


# ── SIGNAL EVALUATION ──────────────────────────────────────────────────────────

def _evaluate_entry(pos_state, weekly, now_str):
    """
    Full entry evaluation. Returns (signal_dict | None, conditions_dict).
    signal_dict has keys: expiry, short_sym, long_sym, short_strike, long_strike,
                          short_delta, credit, spy_px, ivr, vix.
    """
    passed, conds, spy_px, ivr, vix = _check_entry_conditions(pos_state, weekly)
    if not passed:
        return None, conds

    expiry = _find_today_expiration()
    if expiry is None:
        conds['expiry_found'] = {'passed': False, 'detail': 'no 0DTE expiry today'}
        return None, conds
    conds['expiry_found'] = {'passed': True, 'detail': expiry.isoformat()}

    chain = _fetch_0dte_chain(expiry, now_str)
    if chain is None:
        conds['chain_fetched'] = {'passed': False,
                                   'detail': 'chain unavailable (see CHAIN_* log event)'}
        return None, conds
    conds['chain_fetched'] = {'passed': True, 'detail': f'{len(chain)} contracts'}

    short_sym, short_strike, short_delta = _find_short_strike(chain, spy_px, vix, now_str)
    if short_sym is None:
        conds['short_strike'] = {'passed': False,
                                  'detail': 'no suitable 25-30Δ put (see CHAIN_* log)'}
        return None, conds
    conds['short_strike'] = {'passed': True,
                              'detail': f'${short_strike:.0f}  delta={short_delta:.3f}'}

    long_sym, long_strike = _find_long_symbol(chain, short_strike)
    if long_sym is None:
        conds['long_strike'] = {'passed': False, 'detail': 'not found in chain'}
        return None, conds
    conds['long_strike'] = {'passed': True, 'detail': f'${long_strike:.0f}'}

    credit = _spread_mid(chain, short_sym, long_sym)
    if credit is None or credit < MIN_CREDIT:
        conds['min_credit'] = {
            'passed': False,
            'detail': (f'credit=${credit:.4f} < ${MIN_CREDIT}'
                       if credit is not None else 'mid unavailable'),
        }
        return None, conds
    conds['min_credit'] = {'passed': True, 'detail': f'credit=${credit:.4f}'}

    return {
        'expiry':       expiry,
        'short_sym':    short_sym,
        'long_sym':     long_sym,
        'short_strike': short_strike,
        'long_strike':  long_strike,
        'short_delta':  short_delta,
        'credit':       credit,
        'spy_px':       spy_px,
        'ivr':          ivr,
        'vix':          vix,
    }, conds


# ── DAILY SUMMARY ──────────────────────────────────────────────────────────────

def _check_daily_summary(pos_state, weekly, now_str):
    """Send once-daily summary at 3:50pm ET. ACTIVE-gated."""
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

    positions  = pos_state.get('positions', [])
    real_pos   = [p for p in positions
                  if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    week_loss  = weekly.get('weekly_realized_loss', 0.0)

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
        wins   = sum(1 for t in today_closes if t.get('pnl', 0) > 0)
        losses = exits_today - wins
    except Exception:
        realized_pnl = entries_today = exits_today = wins = losses = 0

    # 0DTE always resolves same day — by the 3:50pm summary, real_pos should
    # be empty except for a force-close that just failed and needs a human.
    pos_detail = ''
    for pos in real_pos:
        pos_detail += (
            f'\n  ⚠️ STILL OPEN [needs manual attention]: '
            f'{pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P '
            f'credit=${pos["credit"]:.2f}'
        )

    reason_str = f'wins: {wins}  losses: {losses}'
    pnl_str    = f'+${realized_pnl:.2f}' if realized_pnl >= 0 else f'-${abs(realized_pnl):.2f}'

    _discord(
        f'📊 **0DTE-V2 Daily Summary**\n'
        f'Entries today: {entries_today}  |  Exits today: {exits_today}  ({reason_str})\n'
        f'Realized P&L: {pnl_str}{pos_detail}\n'
        f'Week-to-date loss: ${week_loss:.2f} / ${WEEKLY_LOSS_LIMIT:.0f} limit'
    )
    _log({'timestamp': now_str, 'event': 'DAILY_SUMMARY_SENT', 'date': today_str,
          'entries': entries_today, 'exits': exits_today, 'realized_pnl': realized_pnl})


# ── MORNING VITALS ─────────────────────────────────────────────────────────────

def _send_morning_vitals():
    """
    Post a market-open snapshot to Discord once per trading day at 9:30 ET.
    Dedup flag: pos_state['morning_vitals_sent'] == today's date (stored in DB).
    Holiday check: Alpaca /v2/calendar — skips silently if market is closed.
    """
    if not ACTIVE:
        return
    if not _is_vitals_window():
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

    spy_above, spy_px, spy_sma = _above_sma20()
    if spy_px is not None and spy_sma is not None:
        status   = '✅ Above' if spy_above else '❌ Below'
        spy_line = f'SPY:  ${spy_px:.2f}  |  SMA20: ${spy_sma:.2f}  |  {status}'
    else:
        spy_line = 'SPY:  N/A'

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
    macro_line = (f'MACRO: {macro}  |  ⚠️ Blocked' if macro
                  else 'MACRO: None scheduled  |  ✅ Clear')

    real_pos = [p for p in pos_state.get('positions', [])
                if not (p.get('reconciled') and p.get('short_symbol') == 'UNKNOWN')]
    pos_lines = f'OPEN POSITIONS: {len(real_pos)} of {MAX_POSITIONS}'
    for pos in real_pos:
        credit = pos.get('credit', 0)
        cost   = _current_cost_to_close(pos.get('short_symbol', ''), pos.get('long_symbol', ''))
        pct_str  = f'  ({(credit - cost) / credit * 100:.0f}% to target)' if (cost is not None and credit > 0) else ''
        cost_str = f'${cost:.4f}' if cost is not None else 'N/A'
        pos_lines += (
            f'\n  {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P'
            f'  cost {cost_str} / credit ${credit:.2f}{pct_str}'
        )

    week_loss = weekly.get('weekly_realized_loss', 0.0)
    week_str  = f'-${week_loss:.2f}' if week_loss > 0 else '+$0.00'
    entry_line = (f'ENTRY: 9:45–11:00am ET  |  25-30Δ ${SPREAD_WIDTH:.0f}-wide  |  '
                  f'Min credit ${MIN_CREDIT:.2f}  |  Stop {STOP_LOSS_PCT*100:.0f}%  |  '
                  f'Force close 3:45pm')
    system_line = f'SYSTEM: ✅ Active  |  Week P&L: {week_str}  |  Scan: every 60s'

    msg = (
        f'📊 **MORNING VITALS — {date_str}**\n\n'
        f'{spy_line}\n'
        f'{vix_line}\n'
        f'{ivr_line}\n'
        f'{macro_line}\n\n'
        f'{pos_lines}\n\n'
        f'{entry_line}\n'
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
        print(f'[{_t0.strftime("%H:%M ET")}] 0DTE-V2: outside market hours, skipping.')
        _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
        return

    now_et  = datetime.now(ET)
    now_str = now_et.strftime('%Y-%m-%d %H:%M ET')

    # ── 0. Morning vitals — fires once at 9:30 ET each trading day ────────────
    try:
        _send_morning_vitals()
    except Exception as e:
        print(f'  [morning_vitals] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'MORNING_VITALS_ERROR',
              'error': f'{type(e).__name__}: {e}'})

    print(f'\n[{now_str}] 0DTE-V2 scan  (ACTIVE={ACTIVE})…')

    pos_state = _load_positions()
    weekly    = _load_weekly()
    weekly    = _reset_weekly_if_needed(weekly)

    # ── 1. Monitor open positions ──────────────────────────────────────────────
    try:
        _monitor_positions(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [monitor] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'MONITOR_ERROR',
              'error': f'{type(e).__name__}: {e}'})

    # ── 2. Daily summary ───────────────────────────────────────────────────────
    try:
        _check_daily_summary(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [summary] ERROR — {type(e).__name__}: {e}')

    # ── 3. Evaluate entry ──────────────────────────────────────────────────────
    scan_result = 'no_signal'
    signal = None
    conds  = {}
    try:
        signal, conds = _evaluate_entry(pos_state, weekly, now_str)
    except Exception as e:
        print(f'  [entry eval] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'ENTRY_EVAL_ERROR',
              'error': f'{type(e).__name__}: {e}'})

    # ── 4. Execute entry ───────────────────────────────────────────────────────
    if signal:
        s = signal
        if not ACTIVE:
            scan_result = 'dormant_would_enter'
            print(
                f'  DORMANT MODE — entry skipped | '
                f'SPY {s["short_strike"]:.0f}/{s["long_strike"]:.0f}P  '
                f'exp={s["expiry"]}  credit=${s["credit"]:.4f}  '
                f'delta={s["short_delta"]:.3f}'
            )
        else:
            print(f'  ENTERING: SPY {s["short_strike"]:.0f}/{s["long_strike"]:.0f}P  '
                  f'exp={s["expiry"]}  credit=${s["credit"]:.4f}')
            filled = _attempt_entry(
                pos_state, weekly, now_str,
                s['short_sym'], s['long_sym'],
                s['short_strike'], s['long_strike'],
                s['expiry'], s['credit'],
                s['spy_px'], s['short_delta'],
            )
            scan_result = 'entry_filled' if filled else 'entry_not_filled'
    else:
        fails = [k for k, v in conds.items() if not v.get('passed')]
        if 'macro_event' in fails:
            print(f'  No entry — macro event day: {conds["macro_event"]["detail"]}')
        elif 'underwater_block' in fails:
            print('  No entry — blocked: existing position underwater')
        elif fails:
            print(f'  No entry — failed: {", ".join(fails)}')

    # ── 5. Log scan ────────────────────────────────────────────────────────────
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
        'conditions':     conds,
    }
    if signal and scan_result in ('dormant_would_enter', 'entry_filled', 'entry_not_filled'):
        log_entry['signal'] = {
            'short_symbol': signal['short_sym'],
            'long_symbol':  signal['long_sym'],
            'expiry':       str(signal['expiry']),
            'credit':       signal['credit'],
            'short_delta':  round(signal['short_delta'], 4) if signal['short_delta'] else None,
            'spy_px':       signal['spy_px'],
        }
    _log(log_entry)


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    print('=' * 64)
    print('  0DTE-V2 SPY PUT CREDIT SPREADS  |  25-30Δ / $3-wide')
    print(f'  ACTIVE = {ACTIVE}')
    if not ACTIVE:
        print('  *** DORMANT — scanning and logging, NO orders placed ***')
    print('  Entry: 9:45–11:00am ET  |  Force close: 3:45pm ET')
    print('  Alpaca v2 REST API  |  raw requests  |  no SDK')
    print('  Scan every 60s, 9:30–16:00 ET, Mon–Fri')
    print('=' * 64)

    _init_db()
    _init_files()
    _reconcile_on_startup()

    schedule.every(60).seconds.do(run_scan)

    if ACTIVE:
        _discord(
            f'✅ 0DTE-V2 system live | SPY 0DTE put spreads | '
            f'25-30Δ short / ${SPREAD_WIDTH:.0f}-wide / ${MIN_CREDIT:.2f} min credit | '
            f'{MAX_POSITIONS} positions max | Entry 9:45–11:00am | Force close 3:45pm'
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

                if since_s > 300:   # 5-minute watchdog — tighter than 7DTE's 10min, scans run every 60s here
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
        time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f'[FATAL] {type(e).__name__}: {e}')
        traceback.print_exc()
        raise
