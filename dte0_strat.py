#!/usr/bin/env python3
"""
dte0_strat.py — 0DTE SPY Put Credit Spread Strategy.

STANDALONE: no imports from any other strategy file.
DORMANT until ACTIVE = True.

Alpaca API: raw requests only (no SDK).
  POST /v2/orders  order_class='mleg'  for multi-leg spread orders.

Database tables (never touches credit_spread_positions or credit_spread_state):
  dte0_positions — open and closed positions with full entry/exit metadata
  dte0_state     — daily summary and morning vitals dedup flags

Key differences from credit_spread_strat.py (7DTE):
  Entry window  9:45–11:00 ET only (narrow — avoid afternoon gamma explosion)
  Force close   3:45pm ET same day (not 9:45am expiry morning)
  Target delta  0.15 (tighter than 0.20)
  Spread width  $2.00 (narrower than $5.00)
  Min credit    $0.10
  Stop loss     150% of credit (tighter than 200%)
  Max positions 1
  IVR threshold 20% (lower than 30%)
  No weekly loss limit
  Scan every 60 seconds (not 5 minutes)
  All Discord messages prefixed [0DTE]

Deploy as a second Railway worker (add to Procfile):
  dte0: python3 dte0_strat.py
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
ACTIVE = True    # Set to True to enable live order placement

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK  = os.getenv('DISCORD_WEBHOOK_URL')
ALPACA_KEY       = os.getenv('ALPACA_KEY')
ALPACA_SECRET    = os.getenv('ALPACA_SECRET')
PAPER_BASE_URL   = 'https://paper-api.alpaca.markets'
DATA_URL         = 'https://data.alpaca.markets'
DATABASE_URL     = os.getenv('DATABASE_URL')
ET               = pytz.timezone('US/Eastern')
_DIR             = os.path.dirname(os.path.abspath(__file__))

# Strategy parameters
TARGET_DELTA       = 0.15
DELTA_TOLERANCE    = 0.05
SPREAD_WIDTH       = 2.0
MIN_CREDIT         = 0.10
MAX_POSITIONS      = 1
PROFIT_TARGET_PCT  = 0.50
STOP_LOSS_PCT      = 1.50
ORDER_FILL_TIMEOUT = 120    # shorter than 7DTE — 0DTE moves fast
RISK_FREE_RATE     = 0.045
VIX_IVR_WINDOW     = 252
MIN_IVR            = 20.0
MAX_VIX            = 35.0
SMA_PERIOD         = 20

# Timing (ET)
ENTRY_HOUR_START, ENTRY_MIN_START  =  9, 45
ENTRY_HOUR_END,   ENTRY_MIN_END    = 11,  0
FORCE_CLOSE_HOUR, FORCE_CLOSE_MIN  = 15, 45
SUMMARY_HOUR,     SUMMARY_MIN      = 15, 50
VITALS_HOUR,      VITALS_MIN_START = 9,  30
VITALS_MIN_END                     =  9, 34

# ── MACRO EVENT CALENDAR (2026) ────────────────────────────────────────────────
# Update each January. NFP is always first Friday of the month — computed in code.
FOMC_DAYS = {
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
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-13', '2026-06-11', '2026-07-14', '2026-08-12',
    '2026-09-09', '2026-10-14', '2026-11-12', '2026-12-10',
}

GDP_DAYS = {
    '2026-01-29',
    '2026-04-29',
    '2026-07-30',
    '2026-10-29',
}


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


# ── DATABASE ───────────────────────────────────────────────────────────────────

_DB = None


def _get_db():
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
    """Create dte0_positions and dte0_state tables if they don't exist."""
    conn = _get_db()
    if conn is None:
        if DATABASE_URL and psycopg2 is None:
            print('[db] WARNING — DATABASE_URL set but psycopg2 not installed')
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dte0_positions (
                id             SERIAL PRIMARY KEY,
                short_symbol   TEXT NOT NULL,
                long_symbol    TEXT NOT NULL,
                short_strike   DOUBLE PRECISION,
                long_strike    DOUBLE PRECISION,
                expiration     TEXT,
                credit         DOUBLE PRECISION,
                max_risk       DOUBLE PRECISION,
                breakeven      DOUBLE PRECISION,
                profit_target  DOUBLE PRECISION,
                stop_loss_cost DOUBLE PRECISION,
                entry_time     TEXT,
                entry_order_id TEXT,
                short_delta    DOUBLE PRECISION,
                spy_entry_px   DOUBLE PRECISION,
                reconciled     BOOLEAN DEFAULT FALSE,
                note           TEXT,
                close_time     TEXT,
                close_reason   TEXT,
                realized_pnl   DOUBLE PRECISION
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dte0_state (
                id                  INTEGER PRIMARY KEY DEFAULT 1,
                daily_summary_sent  TEXT,
                morning_vitals_sent TEXT
            )
        """)
        conn.commit()
        print('[db] dte0 tables ready')
    except Exception as e:
        print(f'[db] _init_db failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass


def _db_load_active_position():
    """
    Return the single open dte0 position (close_time IS NULL) for today, or None.
    Returns dict with 'db_id' key added for later UPDATE on close.
    """
    conn = _get_db()
    if conn is None:
        return None
    try:
        today = datetime.now(ET).date().isoformat()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, short_symbol, long_symbol, short_strike, long_strike,
                   expiration, credit, max_risk, breakeven, profit_target,
                   stop_loss_cost, entry_time, entry_order_id, short_delta,
                   spy_entry_px, reconciled, note
            FROM dte0_positions
            WHERE close_time IS NULL AND expiration = %s
            ORDER BY id DESC LIMIT 1
        """, (today,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            'db_id':          row[0],
            'short_symbol':   row[1],
            'long_symbol':    row[2],
            'short_strike':   row[3],
            'long_strike':    row[4],
            'expiration':     row[5],
            'credit':         row[6],
            'max_risk':       row[7],
            'breakeven':      row[8],
            'profit_target':  row[9],
            'stop_loss_cost': row[10],
            'entry_time':     row[11],
            'entry_order_id': row[12],
            'short_delta':    row[13],
            'spy_entry_px':   row[14],
            'reconciled':     bool(row[15]),
            'note':           row[16],
        }
    except Exception as e:
        print(f'[db] _db_load_active_position failed: {e}')
        global _DB
        _DB = None
        return None


def _db_insert_position(pos):
    """Insert a new open position. Returns the new row id, or None on failure."""
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dte0_positions (
                short_symbol, long_symbol, short_strike, long_strike,
                expiration, credit, max_risk, breakeven, profit_target,
                stop_loss_cost, entry_time, entry_order_id, short_delta,
                spy_entry_px
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            pos.get('short_symbol'),    pos.get('long_symbol'),
            pos.get('short_strike'),    pos.get('long_strike'),
            pos.get('expiration'),      pos.get('credit'),
            pos.get('max_risk'),        pos.get('breakeven'),
            pos.get('profit_target'),   pos.get('stop_loss_cost'),
            pos.get('entry_time'),      pos.get('entry_order_id'),
            pos.get('short_delta'),     pos.get('spy_entry_px'),
        ))
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    except Exception as e:
        print(f'[db] _db_insert_position failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return None


def _db_close_position(db_id, close_time, close_reason, realized_pnl):
    """Fill in close fields on an existing dte0_positions row."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE dte0_positions
               SET close_time   = %s,
                   close_reason = %s,
                   realized_pnl = %s
             WHERE id = %s
        """, (close_time, close_reason, realized_pnl, db_id))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_close_position failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return False


def _db_mark_stale_open_expired(today_str):
    """
    On startup: mark any open positions with expiration < today as EXPIRED_UNTRACKED.
    0DTE options from a prior day expired at market close — realized_pnl is unknown.
    """
    conn = _get_db()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE dte0_positions
               SET close_time   = %s,
                   close_reason = 'EXPIRED_UNTRACKED',
                   realized_pnl = NULL
             WHERE close_time IS NULL AND expiration < %s
        """, (today_str + ' 16:00 ET (startup cleanup)', today_str))
        n = cur.rowcount
        conn.commit()
        if n:
            print(f'[db] Marked {n} stale open position(s) as EXPIRED_UNTRACKED')
    except Exception as e:
        print(f'[db] _db_mark_stale_open_expired failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass


def _db_get_today_closed(today_str):
    """Return list of closed position dicts for today (for daily summary)."""
    conn = _get_db()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT credit, realized_pnl, close_reason
            FROM dte0_positions
            WHERE expiration = %s AND close_time IS NOT NULL
              AND close_reason != 'EXPIRED_UNTRACKED'
            ORDER BY id
        """, (today_str,))
        rows = cur.fetchall()
        return [{'credit': r[0], 'realized_pnl': r[1], 'close_reason': r[2]}
                for r in rows]
    except Exception as e:
        print(f'[db] _db_get_today_closed failed: {e}')
        return []


def _db_read_state_flags():
    """Return (daily_summary_sent, morning_vitals_sent) or (None, None)."""
    conn = _get_db()
    if conn is None:
        return None, None
    try:
        cur = conn.cursor()
        cur.execute(
            'SELECT daily_summary_sent, morning_vitals_sent '
            'FROM dte0_state WHERE id = 1'
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)
    except Exception as e:
        print(f'[db] _db_read_state_flags failed: {e}')
        return None, None


def _db_write_state_flags(daily_str, vitals_str):
    """Upsert the dte0_state row."""
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dte0_state (id, daily_summary_sent, morning_vitals_sent)
            VALUES (1, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                daily_summary_sent  = EXCLUDED.daily_summary_sent,
                morning_vitals_sent = EXCLUDED.morning_vitals_sent
        """, (daily_str, vitals_str))
        conn.commit()
        return True
    except Exception as e:
        print(f'[db] _db_write_state_flags failed: {e}')
        try:
            conn.rollback()
        except Exception:
            pass
        global _DB
        _DB = None
        return False


# ── IN-MEMORY FALLBACK STATE ───────────────────────────────────────────────────
# Used when DB is unavailable. Lost on process restart, which is acceptable for
# a same-day 0DTE track: all positions expire by 4pm regardless.

_mem_position          = None   # active position dict, or None
_mem_daily_summary     = None   # date string if sent today
_mem_morning_vitals    = None   # date string if sent today
_mem_today_trades      = []     # closed trades today (for summary when DB is down)


def _load_active_position():
    """Load active position: DB primary, in-memory fallback."""
    if DATABASE_URL:
        result = _db_load_active_position()
        if result is not None or _get_db() is not None:
            return result
        print('[db] _load_active_position: DB unavailable, using in-memory')
    return _mem_position


def _save_position_opened(pos):
    """Persist a newly opened position. Returns pos dict (with db_id if DB worked)."""
    global _mem_position
    if DATABASE_URL:
        db_id = _db_insert_position(pos)
        if db_id is not None:
            pos['db_id'] = db_id
            _mem_position = pos
            return pos
        print('[db] _save_position_opened: DB write failed — tracking in memory only')
    _mem_position = pos
    return pos


def _save_position_closed(pos, close_time, close_reason, realized_pnl):
    """Persist position close. Clears in-memory position."""
    global _mem_position, _mem_today_trades
    trade = {**pos, 'close_time': close_time, 'close_reason': close_reason,
             'realized_pnl': realized_pnl}
    _mem_today_trades.append(trade)

    if DATABASE_URL and pos.get('db_id') is not None:
        if not _db_close_position(pos['db_id'], close_time, close_reason, realized_pnl):
            print('[db] _save_position_closed: DB update failed')
    _mem_position = None


def _load_state_flags():
    """Return (daily_summary_sent, morning_vitals_sent)."""
    if DATABASE_URL:
        d, v = _db_read_state_flags()
        if d is not None or v is not None or _get_db() is not None:
            return d, v
    return _mem_daily_summary, _mem_morning_vitals


def _save_state_flags(daily_str, vitals_str):
    global _mem_daily_summary, _mem_morning_vitals
    _mem_daily_summary  = daily_str
    _mem_morning_vitals = vitals_str
    if DATABASE_URL:
        _db_write_state_flags(daily_str, vitals_str)


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
    """Rate-limit-aware Discord post. ACTIVE-gated."""
    if not ACTIVE:
        return
    if not DISCORD_WEBHOOK:
        print(f'[Discord] {msg}')
        return
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK, json={'content': msg}, timeout=10)
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

_LOG_FILE = os.path.join(_DIR, 'dte0_log.json')
_LOG_MAX  = 500


def _log(entry):
    try:
        try:
            with open(_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append(entry)
        if len(log) > _LOG_MAX:
            log = log[-_LOG_MAX:]
        with open(_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        print(f'  [_log] write failed: {e}')


# ── MARKET DATA ────────────────────────────────────────────────────────────────

def _flatten_columns(df):
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
    Identical implementation to credit_spread_strat.py._spy_ivrank().
    Falls back to _vix_ivrank() if Alpaca options fetch fails.
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
                current_iv = _bs_iv_solve(S, K, T, (bid + ask) / 2)
            if current_iv is None:
                raise ValueError('IV not available; back-solve also failed')

    except Exception as e:
        print(f'  [ivrank] SPY IV fetch failed — using VIX proxy fallback: {e}')
        return _vix_ivrank()

    current_iv_pct = current_iv * 100

    try:
        df = yf.download('SPY', period=f'{VIX_IVR_WINDOW + 60}d',
                         interval='1d', progress=False, auto_adjust=True, timeout=10)
        if df.empty or len(df) < 20:
            raise ValueError('SPY OHLC data insufficient')
        df    = _flatten_columns(df)
        highs = df['High'].dropna().values.astype(float)
        lows  = df['Low'].dropna().values.astype(float)
        n     = min(len(highs), len(lows))
        log_hl   = np.log(highs[-n:] / lows[-n:])
        park_ann = np.sqrt(log_hl ** 2 / (4 * math.log(2))) * math.sqrt(252) * 100
        window   = park_ann[-VIX_IVR_WINDOW:] if len(park_ann) >= VIX_IVR_WINDOW else park_ann
        iv_low   = float(window.min())
        iv_high  = float(window.max())

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
    Uses actual wall-clock hours remaining until 4:00pm ET, divided by
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
            print(f'  [0dte chain] contracts fetch returned None after retries')
            _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR',
                  'reason': 'contracts fetch returned None'})
            return None
        contracts = r.json().get('option_contracts', [])
        if not contracts:
            print(f'  [0dte chain] no contracts returned for {expiry}')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry)})
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
                    print(f'  [0dte chain] snapshots batch failed (offset {batch_start})')
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
                print(f'  [0dte chain] snapshot batch error: {e}')

        if not chain:
            print('  [0dte chain] all snapshot batches returned empty')
            _log({'timestamp': now_str, 'event': 'CHAIN_EMPTY', 'expiry': str(expiry),
                  'reason': 'all batches empty'})
            return None

        return chain

    except Exception as e:
        print(f'  [0dte chain] fetch failed: {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'CHAIN_FETCH_ERROR',
              'reason': f'{type(e).__name__}: {e}'})
        return None


def _find_short_strike(chain, spy_px, vix, now_str):
    """
    Find OTM put with |delta| closest to TARGET_DELTA (0.15).

    Delta priority per contract:
      1. Alpaca greeks.delta  (most accurate)
      2. Per-contract IV via Black-Scholes with intraday T
      3. VIX / 100 via Black-Scholes (last resort)

    Rejects if best |delta| is > DELTA_TOLERANCE (0.05) from 0.15.
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
    """Find option symbol for the long leg (short_strike − SPREAD_WIDTH = $2 lower)."""
    target    = short_strike - SPREAD_WIDTH
    best_sym  = best_k = None
    best_diff = float('inf')
    for sym, data in chain.items():
        diff = abs(data['strike'] - target)
        if diff < best_diff:
            best_diff, best_sym, best_k = diff, sym, data['strike']
    return best_sym, best_k


def _spread_mid(chain, short_sym, long_sym):
    """Net credit = short_mid − long_mid. Returns None if degenerate."""
    try:
        s_mid = chain[short_sym]['mid']
        l_mid = chain[long_sym]['mid']
        if s_mid <= 0 or l_mid < 0:
            return None
        return round(s_mid - l_mid, 4)
    except (KeyError, TypeError):
        return None


def _current_cost_to_close(short_sym, long_sym):
    """
    Current debit to close = short_mid − long_mid.
    Returns 0.0 when both legs zero-priced (OTM, near expiry — valid max profit).
    Returns None on hard API failure.
    """
    ts = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    try:
        rs = _alpaca_get(
            f'{DATA_URL}/v1beta1/options/snapshots',
            headers=_data_headers(),
            params={'symbols': f'{short_sym},{long_sym}', 'feed': 'indicative'},
        )
        if rs is None:
            print(f'  [cost_to_close] API returned None ({short_sym}, {long_sym})')
            _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
                  'short_sym': short_sym, 'long_sym': long_sym, 'reason': 'api_none'})
            return None

        snaps = rs.json().get('snapshots', {})
        missing = [s for s in (short_sym, long_sym) if s not in snaps]
        if missing:
            keys_sample = list(snaps.keys())[:4]
            print(f'  [cost_to_close] symbols absent: {missing} (had: {keys_sample})')
            _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
                  'short_sym': short_sym, 'long_sym': long_sym,
                  'reason': 'symbols_absent', 'missing': missing})
            return None

        def _mid(sym):
            q   = snaps[sym].get('latestQuote', {})
            bid = float(q.get('bp') or 0)
            ask = float(q.get('ap') or 0)
            return (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0

        s_mid = _mid(short_sym)
        l_mid = _mid(long_sym)
        cost  = round(s_mid - l_mid, 4)
        if s_mid == 0.0 and l_mid == 0.0:
            print(f'  [cost_to_close] both legs zero-quoted → cost=0.00 (worthless)')
        return cost

    except Exception as e:
        print(f'  [cost_to_close] {type(e).__name__}: {e}')
        _log({'timestamp': ts, 'event': 'COST_TO_CLOSE_UNAVAILABLE',
              'short_sym': short_sym, 'long_sym': long_sym,
              'reason': f'{type(e).__name__}: {e}'})
        return None


# ── ORDER MANAGEMENT ───────────────────────────────────────────────────────────

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
            'limit_price':   str(round(credit, 2)),
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
    """Order to close the spread. ACTIVE-gated."""
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
        return r.json() if r else None
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


# ── ENTRY EXECUTION ────────────────────────────────────────────────────────────

def _attempt_entry(short_sym, long_sym, short_strike, long_strike,
                   expiry, credit, spy_px, short_delta, now_str):
    """
    Place opening order, poll for fill up to ORDER_FILL_TIMEOUT seconds.
    Returns filled position dict on success, None on failure.
    ACTIVE-gated.
    """
    if not ACTIVE:
        return None

    order_id = _place_open_order(short_sym, long_sym, credit)
    if not order_id:
        _log({'timestamp': now_str, 'event': 'ORDER_PLACE_FAILED',
              'short': short_sym, 'long': long_sym, 'credit': credit})
        print('  [entry] order placement failed')
        return None

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
            fill_credit = abs(raw_price)
            if raw_price < 0:
                print(f'  [entry] filled_avg_price was negative ({raw_price}) — using abs()')
            max_risk  = round((SPREAD_WIDTH - fill_credit) * 100, 2)
            breakeven = round(short_strike - fill_credit, 2)
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
                'entry_time':     now_str,
                'entry_order_id': order_id,
                'short_delta':    round(short_delta, 4) if short_delta else None,
                'spy_entry_px':   spy_px,
            }
            pos = _save_position_opened(pos)
            _log({'timestamp': now_str, 'event': 'ENTRY_FILLED', **pos})
            _discord(
                f'[0DTE] 🟢 SPREAD OPEN | '
                f'SPY {short_strike:.0f}/{long_strike:.0f}P  exp {expiry} | '
                f'Credit: ${fill_credit:.2f} | Max risk: ${max_risk:.2f} | '
                f'Breakeven: ${breakeven:.2f} | Delta: {short_delta:.3f}'
            )
            print(f'  [entry] FILLED ${fill_credit:.4f}  {short_sym} / {long_sym}')
            return pos

        if status == 'partially_filled':
            print(f'  [entry] partial fill on multi-leg order — cancelling {order_id}')
            _cancel_order(order_id)
            _discord(
                f'[0DTE] ⚠️ Partial fill on spread entry — cancelled. '
                f'Check Alpaca for any open legs needing manual close.'
            )
            return None

        if status in ('cancelled', 'expired', 'rejected', 'done_for_day'):
            print(f'  [entry] order {status} — no fill')
            _log({'timestamp': now_str, 'event': f'ORDER_{status.upper()}',
                  'order_id': order_id})
            return None

    print(f'  [entry] fill timeout ({ORDER_FILL_TIMEOUT}s) — cancelling {order_id}')
    _cancel_order(order_id)
    _log({'timestamp': now_str, 'event': 'ENTRY_FILL_TIMEOUT', 'order_id': order_id})
    return None


# ── POSITION MONITORING ────────────────────────────────────────────────────────

def _record_exit(pos, reason, close_cost, now_str):
    """Log exit to DB, Discord, and log file. Returns realized_pnl."""
    credit = pos['credit']
    pnl    = round((credit - close_cost) * 100, 2)
    label  = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'

    _save_position_closed(pos, now_str, reason, pnl)
    _log({'timestamp': now_str, 'event': 'EXIT', 'reason': reason,
          'label': label, 'credit': credit, 'close_cost': close_cost, 'pnl': pnl})

    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
    reason_labels = {
        'PROFIT_TARGET':   'profit target',
        'STOP_LOSS':       'stop loss',
        'FORCE_CLOSE':     'force close 3:45pm',
    }
    _discord(
        f'[0DTE] {"✅" if pnl >= 0 else "🔴"} SPREAD CLOSED '
        f'({reason_labels.get(reason, reason)}) | '
        f'{label} | '
        f'Credit: ${credit:.2f}  Close: ${close_cost:.4f} | '
        f'P&L: {pnl_str}'
    )
    return pnl


def _monitor_position(pos, now_str):
    """
    Check open position for profit target, stop loss, or 3:45pm force close.
    Returns True if position was closed (caller should stop using pos).
    ACTIVE-gated.
    """
    if not ACTIVE:
        return False

    if pos.get('reconciled') and pos.get('short_symbol') == 'UNKNOWN':
        print('  [monitor] Skipping untracked reconciled position — manual review required')
        return False

    short_sym = pos['short_symbol']
    long_sym  = pos['long_symbol']
    credit    = pos['credit']
    label     = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P {pos["expiration"]}'

    try:
        # ── 3:45pm force close ────────────────────────────────────────────────
        if _is_force_close_time():
            print(f'  {label}: FORCE CLOSE (3:45pm)')
            cost = _current_cost_to_close(short_sym, long_sym)
            if cost is None:
                cost = 0.0
                print(f'  {label}: cost unavailable for force close — using 0.0')
            ok = _place_close_order(short_sym, long_sym, order_type='market')
            if ok:
                _record_exit(pos, 'FORCE_CLOSE', cost, now_str)
                return True
            else:
                print(f'  {label}: FORCE CLOSE ORDER FAILED')
                _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                      'label': label, 'reason': 'FORCE_CLOSE'})
                _discord(
                    f'[0DTE] 🚨 FORCE CLOSE FAILED | {label} | '
                    f'Manual close in Alpaca required immediately — expires today.'
                )
            return False

        # ── Current spread value ──────────────────────────────────────────────
        cost = _current_cost_to_close(short_sym, long_sym)
        if cost is None:
            print(f'  {label}: value unavailable — skipping this cycle')
            _log({'timestamp': now_str, 'event': 'MONITOR_VALUE_UNAVAILABLE',
                  'label': label})
            return False

        print(f'  {label}: cost={cost:.4f}  credit={credit:.4f}  '
              f'tgt≤{pos["profit_target"]:.4f}  stop≥{pos["stop_loss_cost"]:.4f}')

        # ── Profit target: cost ≤ 50% of original credit ─────────────────────
        if cost <= pos['profit_target']:
            print(f'  {label}: PROFIT TARGET')
            ok = _place_close_order(short_sym, long_sym,
                                    order_type='limit',
                                    limit_price=pos['profit_target'])
            if ok:
                _record_exit(pos, 'PROFIT_TARGET', cost, now_str)
                return True
            else:
                print(f'  {label}: profit-target close failed — retrying next cycle')
                _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                      'label': label, 'reason': 'PROFIT_TARGET'})
                _discord(f'[0DTE] ⚠️ Close order failed | {label} | Profit target | Retrying next scan')
            return False

        # ── Stop loss: cost ≥ 150% of original credit — market order ─────────
        if cost >= pos['stop_loss_cost']:
            print(f'  {label}: STOP LOSS')
            _discord(f'[0DTE] 🚨 STOP TRIGGERED | {label} | cost={cost:.4f} ≥ stop={pos["stop_loss_cost"]:.4f}')
            ok = _place_close_order(short_sym, long_sym, order_type='market')
            if ok:
                _record_exit(pos, 'STOP_LOSS', cost, now_str)
                return True
            else:
                print(f'  {label}: stop-loss close failed — retrying next cycle')
                _log({'timestamp': now_str, 'event': 'CLOSE_ORDER_FAILED',
                      'label': label, 'reason': 'STOP_LOSS'})
                _discord(f'[0DTE] ⚠️ Close order failed | {label} | Stop loss | Retrying next scan')

    except Exception as e:
        print(f'  {label}: MONITOR ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'MONITOR_POSITION_ERROR',
              'label': label, 'error': f'{type(e).__name__}: {e}'})

    return False


# ── MACRO EVENT FILTER ─────────────────────────────────────────────────────────

def _macro_event_today():
    today     = datetime.now(ET).date()
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

def _check_entry_conditions(pos):
    """
    Gate cheapest-first. Returns (all_passed: bool, conditions: dict, spy_px, ivr, vix).
    pos is the current active position (or None).
    """
    conds    = {}
    spy_px   = ivr = vix = None

    def _c(name, passed, detail):
        conds[name] = {'passed': bool(passed), 'detail': str(detail)}

    macro = _macro_event_today()
    _c('macro_event',    macro is None,    macro or 'none')
    if macro:
        return False, conds, spy_px, ivr, vix

    _c('active',         ACTIVE,           'True' if ACTIVE else 'False — DORMANT')
    _c('entry_window',   _in_entry_window(), '9:45–11:00 ET')
    _c('position_limit', pos is None,       '0/1 open' if pos is None else '1/1 — position open')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, spy_px, ivr, vix

    # SMA filter
    above_sma, spy_close, sma_val = _above_sma20()
    spy_px = spy_close
    if above_sma is None:
        _c('spy_above_sma', False, 'data unavailable')
    else:
        _c('spy_above_sma', above_sma,
           f'SPY={spy_close:.2f}  SMA20={sma_val:.2f}  {"above" if above_sma else "BELOW"}')

    if not all(v['passed'] for v in conds.values()):
        return False, conds, spy_px, ivr, vix

    # IV rank + IV cap
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


# ── DAILY SUMMARY ──────────────────────────────────────────────────────────────

def _check_daily_summary():
    """Send once-daily summary at 3:40pm ET. ACTIVE-gated."""
    if not ACTIVE:
        return
    if not _is_summary_time():
        return

    today_str = datetime.now(ET).date().isoformat()
    daily_sent, vitals_sent = _load_state_flags()
    if daily_sent == today_str:
        return

    _save_state_flags(today_str, vitals_sent)

    closed_today = _db_get_today_closed(today_str) if DATABASE_URL else [
        t for t in _mem_today_trades
        if str(t.get('expiration', '')).startswith(today_str)
        and t.get('close_reason') not in (None, 'EXPIRED_UNTRACKED')
    ]

    n_trades  = len(closed_today)
    realized  = sum(t.get('realized_pnl', 0) or 0 for t in closed_today)
    wins      = sum(1 for t in closed_today if (t.get('realized_pnl') or 0) > 0)
    losses    = n_trades - wins

    reasons = {}
    for t in closed_today:
        r = t.get('close_reason', 'UNKNOWN')
        reasons[r] = reasons.get(r, 0) + 1

    reason_str = '  '.join(f'{r}: {n}' for r, n in reasons.items()) if reasons else 'none'
    pnl_str    = f'+${realized:.2f}' if realized >= 0 else f'-${abs(realized):.2f}'

    _discord(
        f'[0DTE] 📊 **0DTE Daily Summary**\n'
        f'Trades today: {n_trades}  (wins: {wins}  losses: {losses})\n'
        f'Realized P&L: {pnl_str}\n'
        f'Close reasons: {reason_str}'
    )
    _log({'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
          'event': 'DAILY_SUMMARY_SENT', 'date': today_str,
          'trades': n_trades, 'realized_pnl': realized})


# ── MORNING VITALS ─────────────────────────────────────────────────────────────

def _send_morning_vitals():
    """Post 0DTE market snapshot to Discord once per day at 9:30 ET. ACTIVE-gated."""
    if not ACTIVE:
        return
    if not _is_vitals_window():
        return

    today_str   = datetime.now(ET).date().isoformat()
    daily_sent, vitals_sent = _load_state_flags()
    if vitals_sent == today_str:
        return

    # Dedup: mark sent BEFORE fetching so a crash mid-fetch doesn't re-fire
    _save_state_flags(daily_sent, today_str)

    # Holiday check
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
        ivr_line = f'IVR:    {ivr:.1f}%  |  Min: {MIN_IVR:.0f}%         |  {ivr_status}'
    else:
        ivr_line = 'IVR:    N/A'

    macro = _macro_event_today()
    macro_line = (f'MACRO: {macro}  |  ⚠️ Blocked' if macro
                  else 'MACRO: None scheduled  |  ✅ Clear')

    entry_line = (f'ENTRY: 9:45–11:00am ET  |  '
                  f'15Δ ${SPREAD_WIDTH:.0f}-wide  |  Min credit ${MIN_CREDIT:.2f}  |  '
                  f'Stop {STOP_LOSS_PCT*100:.0f}%  |  Force close 3:45pm')

    # Open position from this morning (should be None at 9:30am)
    pos = _load_active_position()
    if pos:
        label     = f'SPY {pos["short_strike"]:.0f}/{pos["long_strike"]:.0f}P'
        cost      = _current_cost_to_close(pos['short_symbol'], pos['long_symbol'])
        cost_str  = f'${cost:.4f}' if cost is not None else 'N/A'
        pos_line  = f'OPEN POSITION: {label}  credit ${pos["credit"]:.2f}  current cost {cost_str}'
    else:
        pos_line = 'OPEN POSITION: none'

    _discord(
        f'[0DTE] 📊 **MORNING VITALS — {date_str}**\n\n'
        f'{spy_line}\n'
        f'{vix_line}\n'
        f'{ivr_line}\n'
        f'{macro_line}\n\n'
        f'{pos_line}\n\n'
        f'{entry_line}\n'
        f'SYSTEM: ✅ Active  |  Scan: every 60s'
    )
    print(f'  [morning_vitals] sent for {today_str}')
    _log({'timestamp': now_et.strftime('%Y-%m-%d %H:%M ET'),
          'event': 'MORNING_VITALS_SENT', 'date': today_str})


# ── STARTUP RECONCILIATION ─────────────────────────────────────────────────────

def _reconcile_on_startup():
    """
    Load DB state, cross-check Alpaca open SPY option legs, alert on mismatch.
    Since this is 0DTE, any open legs from a prior trading day have expired —
    we only expect legs matching today's date in their OCC symbol.
    Returns the active position dict (or None).
    """
    today_str = datetime.now(ET).date().isoformat()
    now_str   = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')

    # Clean up stale open rows from prior days
    _db_mark_stale_open_expired(today_str)

    pos = _load_active_position()

    try:
        r = _alpaca_get(f'{PAPER_BASE_URL}/v2/positions', headers=_headers())
        if r is None:
            raise RuntimeError('positions fetch returned None after retries')

        # 0DTE legs contain today's date in OCC symbol: SPY{YYMMDD}P...
        yy_mm_dd = datetime.now(ET).strftime('%y%m%d')
        option_legs = [
            p for p in r.json()
            if p.get('asset_class') == 'us_option'
            and str(p.get('symbol', '')).startswith(f'SPY{yy_mm_dd}')
        ]
        alpaca_legs    = len(option_legs)
        expected_legs  = 2 if pos is not None else 0

        if alpaca_legs == expected_legs:
            print(f'[reconcile] Alpaca 0DTE OK — {alpaca_legs} leg(s) / '
                  f'{"1 spread" if pos else "0 spreads"}')
        else:
            direction = (f'{alpaca_legs - expected_legs} untracked leg(s)' if alpaca_legs > expected_legs
                         else f'{expected_legs - alpaca_legs} extra state entry(ies)')
            msg = (
                f'[0DTE] ⚠️ OPTIONS MISMATCH on startup | '
                f'Alpaca: {alpaca_legs} 0DTE leg(s), '
                f'state: {expected_legs} expected ({1 if pos else 0} spread(s)). '
                f'{direction}. Manual review required.'
            )
            print(f'[reconcile] {msg}')
            _log({'timestamp': now_str, 'event': 'RECONCILE_MISMATCH',
                  'alpaca_legs': alpaca_legs, 'expected_legs': expected_legs,
                  'symbols': [p['symbol'] for p in option_legs]})
            _discord(msg)

            # Add blocker if Alpaca has untracked legs
            if alpaca_legs > expected_legs and pos is None:
                blocker = {
                    'short_symbol':   'UNKNOWN',
                    'long_symbol':    'UNKNOWN',
                    'short_strike':   0.0,
                    'long_strike':    0.0,
                    'expiration':     today_str,
                    'credit':         0.0,
                    'max_risk':       0.0,
                    'breakeven':      0.0,
                    'profit_target':  0.0,
                    'stop_loss_cost': 999.0,
                    'entry_time':     now_str,
                    'entry_order_id': None,
                    'short_delta':    None,
                    'spy_entry_px':   None,
                    'reconciled':     True,
                    'note':           'Untracked Alpaca position — manual close required',
                }
                pos = _save_position_opened(blocker)

    except Exception as e:
        print(f'[reconcile] Alpaca check failed (startup continues): {e}')
        _log({'timestamp': now_str, 'event': 'RECONCILE_API_ERROR', 'error': str(e)})

    if pos:
        if pos.get('reconciled'):
            print(f'[startup] Blocker placeholder active — new entries blocked')
        else:
            print(f'[startup] Resuming open position: '
                  f'SPY {pos.get("short_strike", 0):.0f}/{pos.get("long_strike", 0):.0f}P  '
                  f'entry={pos.get("entry_time", "?")}  credit=${pos.get("credit", 0):.2f}')
    else:
        print('[startup] No open 0DTE position — starting clean.')

    return pos


# ── MAIN SCAN ──────────────────────────────────────────────────────────────────

_last_scan_start:    datetime = None
_last_scan_duration: float    = 0.0


def run_scan():
    global _last_scan_start, _last_scan_duration
    _t0 = datetime.now(ET)
    _last_scan_start = _t0

    if not is_market_hours():
        print(f'[{_t0.strftime("%H:%M ET")}] 0DTE: outside market hours, skipping.')
        _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
        return

    now_et  = datetime.now(ET)
    now_str = now_et.strftime('%Y-%m-%d %H:%M ET')

    # ── 0. Morning vitals ─────────────────────────────────────────────────────
    try:
        _send_morning_vitals()
    except Exception as e:
        print(f'  [morning_vitals] ERROR — {type(e).__name__}: {e}')

    print(f'\n[{now_str}] 0DTE scan  (ACTIVE={ACTIVE})…')

    # ── 1. Load state ─────────────────────────────────────────────────────────
    pos = _load_active_position()

    # ── 2. Monitor open position ──────────────────────────────────────────────
    if pos is not None:
        try:
            closed = _monitor_position(pos, now_str)
            if closed:
                pos = None   # position gone; entry gate below will see no open pos
        except Exception as e:
            print(f'  [monitor] ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'MONITOR_ERROR',
                  'error': f'{type(e).__name__}: {e}'})

    # ── 3. Daily summary ──────────────────────────────────────────────────────
    try:
        _check_daily_summary()
    except Exception as e:
        print(f'  [summary] ERROR — {type(e).__name__}: {e}')

    # ── 4. Entry evaluation ───────────────────────────────────────────────────
    scan_result = 'no_signal'

    try:
        passed, conds, spy_px, ivr, vix = _check_entry_conditions(pos)
    except Exception as e:
        print(f'  [conditions] ERROR — {type(e).__name__}: {e}')
        _log({'timestamp': now_str, 'event': 'CONDITIONS_ERROR',
              'error': f'{type(e).__name__}: {e}'})
        _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
        return

    signal = None
    if passed:
        try:
            expiry = _find_today_expiration()
            if expiry is None:
                conds['expiry_found'] = {'passed': False, 'detail': 'no 0DTE expiry today'}
            else:
                conds['expiry_found'] = {'passed': True, 'detail': expiry.isoformat()}

                chain = _fetch_0dte_chain(expiry, now_str)
                if chain is None:
                    conds['chain_fetched'] = {'passed': False,
                                               'detail': 'chain unavailable'}
                else:
                    conds['chain_fetched'] = {'passed': True,
                                               'detail': f'{len(chain)} contracts'}

                    short_sym, short_strike, short_delta = _find_short_strike(
                        chain, spy_px, vix, now_str
                    )
                    if short_sym is None:
                        conds['short_strike'] = {'passed': False,
                                                  'detail': 'no suitable 15Δ put'}
                    else:
                        conds['short_strike'] = {'passed': True,
                                                  'detail': f'${short_strike:.0f}  delta={short_delta:.3f}'}

                        long_sym, long_strike = _find_long_symbol(chain, short_strike)
                        if long_sym is None:
                            conds['long_strike'] = {'passed': False, 'detail': 'not found'}
                        else:
                            conds['long_strike'] = {'passed': True, 'detail': f'${long_strike:.0f}'}

                            credit = _spread_mid(chain, short_sym, long_sym)
                            if credit is None or credit < MIN_CREDIT:
                                conds['min_credit'] = {
                                    'passed': False,
                                    'detail': (f'credit=${credit:.4f} < ${MIN_CREDIT}'
                                               if credit is not None else 'mid unavailable'),
                                }
                            else:
                                conds['min_credit'] = {'passed': True,
                                                        'detail': f'credit=${credit:.4f}'}
                                signal = {
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
                                }
        except Exception as e:
            print(f'  [entry eval] ERROR — {type(e).__name__}: {e}')
            _log({'timestamp': now_str, 'event': 'ENTRY_EVAL_ERROR',
                  'error': f'{type(e).__name__}: {e}'})

    # ── 5. Execute entry ──────────────────────────────────────────────────────
    if signal:
        s = signal
        if not ACTIVE:
            scan_result = 'dormant_would_enter'
            print(
                f'  DORMANT — entry skipped | '
                f'SPY {s["short_strike"]:.0f}/{s["long_strike"]:.0f}P  '
                f'exp={s["expiry"]}  credit=${s["credit"]:.4f}  '
                f'delta={s["short_delta"]:.3f}  IVR={s["ivr"]:.1f}%'
            )
        else:
            print(f'  ENTERING: SPY {s["short_strike"]:.0f}/{s["long_strike"]:.0f}P  '
                  f'exp={s["expiry"]}  credit=${s["credit"]:.4f}')
            new_pos = _attempt_entry(
                s['short_sym'], s['long_sym'],
                s['short_strike'], s['long_strike'],
                s['expiry'], s['credit'],
                s['spy_px'], s['short_delta'],
                now_str,
            )
            scan_result = 'entry_filled' if new_pos else 'entry_not_filled'
    else:
        if passed:
            pass   # signal evaluation failed at chain/credit level — logged above
        else:
            fails = [k for k, v in conds.items() if not v.get('passed')]
            if 'macro_event' in fails:
                print(f'  No entry — macro event day: {conds["macro_event"]["detail"]}')
            elif fails:
                print(f'  No entry — failed: {", ".join(fails)}')

    # ── 6. Log scan ───────────────────────────────────────────────────────────
    _last_scan_duration = (datetime.now(ET) - _t0).total_seconds()
    print(f'  [run_scan] completed in {_last_scan_duration:.1f}s')

    log_entry = {
        'timestamp':       now_str,
        'event':           'SCAN',
        'active':          ACTIVE,
        'scan_result':     scan_result,
        'open_position':   1 if pos is not None else 0,
        'scan_duration_s': round(_last_scan_duration, 1),
        'conditions':      conds,
    }
    if signal and scan_result in ('dormant_would_enter', 'entry_filled', 'entry_not_filled'):
        log_entry['signal'] = {
            'short_strike': signal['short_strike'],
            'long_strike':  signal['long_strike'],
            'expiry':       str(signal['expiry']),
            'credit':       signal['credit'],
            'short_delta':  round(signal['short_delta'], 4) if signal['short_delta'] else None,
            'spy_px':       signal['spy_px'],
            'ivr':          signal['ivr'],
        }
    _log(log_entry)


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    print('=' * 64)
    print('  0DTE SPY PUT CREDIT SPREADS  |  15Δ / $2-wide')
    print(f'  ACTIVE = {ACTIVE}')
    if not ACTIVE:
        print('  *** DORMANT — scanning and logging, NO orders placed ***')
    print('  Entry: 9:45–11:00am ET  |  Force close: 3:45pm ET')
    print('  Alpaca v2 REST API  |  raw requests  |  no SDK')
    print('  Scan every 60s, 9:30–16:00 ET, Mon–Fri')
    print('=' * 64)

    _init_db()
    _reconcile_on_startup()

    schedule.every(60).seconds.do(run_scan)

    if ACTIVE:
        _discord(
            f'[0DTE] ✅ 0DTE system live | SPY 0DTE put spreads | '
            f'15Δ short / $2-wide / ${MIN_CREDIT:.2f} min credit | '
            f'Entry 9:45–11:00am | Force close 3:45pm'
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
                      f'last_scan={_last_scan_start.strftime("%H:%M:%S")} | '
                      f'since={since_s:.0f}s | '
                      f'duration={_last_scan_duration:.1f}s')

        schedule.run_pending()
        time.sleep(10)


if __name__ == '__main__':
    main()
