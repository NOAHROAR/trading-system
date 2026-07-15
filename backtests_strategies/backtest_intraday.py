#!/usr/bin/env python3
"""
Intraday backtest: 4 signal types × 10 tickers × 15-min bars, 2023–2025.
Data: Alpaca historical API, cached to intraday_cache/{ticker}_15min.csv

Signals:
  vwap  – VWAP Mean Reversion  (±1.5% deviation + RSI + volume)
  orb   – Opening Range Breakout (first 30-min range + 1.5× volume)
  rs    – Relative Strength Momentum (vs SPY since open)
  vs    – Volume Spike Momentum (2.5× volume + 0.5% move, enter next bar)

Sizing: ATM 1-DTE options, Black-Scholes, $125 risk/trade
Limits: max 3 concurrent positions; no same-ticker re-entry while open
Results → intraday_backtest_results.json
"""

import json
import math
import os
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import norm
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'))

ALPACA_KEY    = os.getenv('ALPACA_KEY')
ALPACA_SECRET = os.getenv('ALPACA_SECRET')
DATA_URL      = 'https://data.alpaca.markets'
CACHE_DIR     = os.path.join(_DIR, 'intraday_cache')
RESULTS_FILE  = os.path.join(_DIR, 'intraday_backtest_results.json')

TICKERS    = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'AMD']
SIGNALS    = ['vwap', 'orb', 'rs', 'vs']
START_DATE = '2023-01-01'
END_DATE   = '2025-12-31'
ET         = ZoneInfo('America/New_York')

RISK_PER_TRADE = 125.0
RISK_FREE      = 0.05
MAX_POSITIONS  = 3
BARS_PER_DAY   = 26          # 9:30–16:00 in 15-min bars
RSI_LEN        = 14
VOL_MA_LEN     = 20
EOD_BAR        = '15:50'     # force-close at/after this time
NO_ENTRY_AFTER = '15:30'     # no new entries after this

# Signal parameters
VWAP_ENTRY  = 0.015          # ±1.5% VWAP deviation to enter
VWAP_EXIT   = 0.003          # ±0.3% → profit target
VWAP_STOP   = 2.0            # 2× entry deviation → stop

ORB_BARS    = 2              # first 2 bars define range (9:30–10:00)
ORB_VOL     = 1.5            # breakout needs 1.5× avg volume
ORB_TGT     = 1.5            # target = range_width × 1.5

RS_EDGE     = 0.015          # ticker must outperform SPY by ≥1.5%
RS_RSI_LONG = (50, 70)
RS_RSI_SHORT= (30, 50)
RS_HOLD     = 4              # hold max 4 bars
RS_REVERSAL = 0.01           # exit if RS reverses 1 pp

VS_VOL      = 2.5            # volume spike threshold
VS_MOVE     = 0.005          # spike bar must move ≥0.5%
VS_TGT      = 1.5            # target = 1.5× spike move


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def _bs(S, K, T, r, sig, typ):
    if T < 1e-9 or sig < 1e-9 or S <= 0 or K <= 0:
        iv = max(S - K, 0) if typ == 'call' else max(K - S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == 'call':
        return max(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 0.01)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.01)


def size_option(S, direction, rv30, bars_left):
    """ATM option, T scaled to bars remaining / (26*252). Returns (prem, contracts, K)."""
    T   = max(bars_left, 1) / (BARS_PER_DAY * 252)
    K   = round(S)
    sig = max(rv30, 0.05)
    typ = 'call' if direction == 'long' else 'put'
    prem = _bs(S, K, T, RISK_FREE, sig, typ)
    contracts = max(1, int(RISK_PER_TRADE / (prem * 100)))
    return prem, contracts, K


def reprice(pos, S, bars_left):
    T   = max(bars_left, 0) / (BARS_PER_DAY * 252)
    K   = pos['K']
    sig = max(pos['rv30'], 0.05)
    typ = 'call' if pos['direction'] == 'long' else 'put'
    return _bs(S, K, T, RISK_FREE, sig, typ)


# ── Data layer ─────────────────────────────────────────────────────────────────

def _hdrs():
    return {'APCA-API-KEY-ID': ALPACA_KEY, 'APCA-API-SECRET-KEY': ALPACA_SECRET}


def download_ticker(ticker):
    """Download + cache 15-min bars for ticker. Returns DataFrame with ET index."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{ticker}_15min.csv')

    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize(ET)
        else:
            df.index = df.index.tz_convert(ET)
        print(f'  {ticker:<6}: {len(df):>7,} bars  (cache)')
        return df

    print(f'  {ticker:<6}: downloading', end='', flush=True)
    bars, token = [], None
    params = dict(timeframe='15Min', feed='iex',
                  start=f'{START_DATE}T00:00:00Z',
                  end=f'{END_DATE}T23:59:59Z', limit=1000)
    while True:
        if token:
            params['page_token'] = token
        r = requests.get(f'{DATA_URL}/v2/stocks/{ticker}/bars',
                         headers=_hdrs(), params=params, timeout=30)
        r.raise_for_status()
        data  = r.json()
        bars += data.get('bars', [])
        token = data.get('next_page_token')
        print('.', end='', flush=True)
        if not token:
            break
        time.sleep(0.05)

    print(f'  {len(bars):,} bars')
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df['t'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert(ET)
    df = df.set_index('t').sort_index()
    df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low',
                       'c': 'close', 'v': 'volume', 'vw': 'vwap_bar'}, inplace=True)
    for col in ['n', 'vwap_bar']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Market hours only: 9:30–16:00 ET
    h, m = df.index.hour, df.index.minute
    df = df[((h == 9) & (m >= 30)) | ((h >= 10) & (h < 16))]
    df.to_csv(cache)
    return df


def add_indicators(df):
    """Adds vwap, rsi, vol_ma, ret_open in-place. Returns enriched copy."""
    df = df.copy()
    dates = df.index.date

    # Daily VWAP (reset each day)
    df['_d']   = dates
    df['_tp']  = (df['high'] + df['low'] + df['close']) / 3
    df['_tv']  = df['_tp'] * df['volume']
    df['vwap'] = (df.groupby('_d')['_tv'].cumsum() /
                  df.groupby('_d')['volume'].cumsum())
    df.drop(columns=['_d', '_tp', '_tv'], inplace=True)

    # RSI (Wilder EMA)
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1/RSI_LEN, adjust=False).mean()
    al    = loss.ewm(alpha=1/RSI_LEN, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    # Volume MA
    df['vol_ma'] = df['volume'].rolling(VOL_MA_LEN, min_periods=5).mean()

    # Return since open (daily reset)
    df['_d2']      = dates
    df['_dayopen'] = df.groupby('_d2')['open'].transform('first')
    df['ret_open'] = (df['close'] - df['_dayopen']) / df['_dayopen']
    df.drop(columns=['_d2', '_dayopen'], inplace=True)

    return df


def load_all():
    print('Loading data...')
    data = {}
    for t in TICKERS:
        df = download_ticker(t)
        if not df.empty:
            data[t] = add_indicators(df)
    print()
    return data


# ── Realized vol (30-day rolling from daily closes) ────────────────────────────

def daily_rv_table(ticker_df):
    daily = ticker_df.groupby(ticker_df.index.date)['close'].last()
    ret   = daily.pct_change().dropna()
    rv    = {}
    idx   = list(daily.index)
    for i, d in enumerate(idx):
        window = ret.iloc[max(0, i - 30):i]
        rv[d]  = float(window.std() * math.sqrt(252)) if len(window) >= 5 else 0.20
    return rv


# ── Entry / exit logic ─────────────────────────────────────────────────────────

def check_entry(sig, row, spy_row, orb, prev):
    """Returns 'long', 'short', or None."""
    close  = row.get('close',  np.nan)
    rsi    = row.get('rsi',    np.nan)
    vol    = row.get('volume', 0)
    vol_ma = row.get('vol_ma', np.nan)

    if pd.isna(close) or pd.isna(rsi) or pd.isna(vol_ma) or vol_ma == 0:
        return None

    if sig == 'vwap':
        vwap = row.get('vwap', np.nan)
        if pd.isna(vwap) or vwap == 0:
            return None
        dev = (close - vwap) / vwap
        if dev <= -VWAP_ENTRY and rsi < 35 and vol > vol_ma:
            return 'long'
        if dev >= VWAP_ENTRY and rsi > 65 and vol > vol_ma:
            return 'short'

    elif sig == 'orb':
        if orb is None:
            return None
        orb_h, orb_l = orb
        if close > orb_h and vol > vol_ma * ORB_VOL:
            return 'long'
        if close < orb_l and vol > vol_ma * ORB_VOL:
            return 'short'

    elif sig == 'rs':
        if spy_row is None:
            return None
        t_ret   = row.get('ret_open', 0)
        spy_ret = spy_row.get('ret_open', 0)
        rs      = t_ret - spy_ret
        if pd.isna(rs):
            return None
        if rs > RS_EDGE and RS_RSI_LONG[0] <= rsi <= RS_RSI_LONG[1] and vol > vol_ma:
            return 'long'
        if rs < -RS_EDGE and RS_RSI_SHORT[0] <= rsi <= RS_RSI_SHORT[1] and vol > vol_ma:
            return 'short'

    elif sig == 'vs':
        if prev is None:
            return None
        pv  = prev.get('volume', 0)
        pma = prev.get('vol_ma', np.nan)
        if pd.isna(pma) or pma == 0 or pv < VS_VOL * pma:
            return None
        po = prev.get('open', 1) or 1
        if abs(prev.get('close', 0) - po) / po < VS_MOVE:
            return None
        return 'long' if prev.get('close', 0) > po else 'short'

    return None


def check_exit(sig, pos, row, spy_row, bar_idx):
    """Returns exit reason string or None."""
    close = row.get('close', np.nan)
    if pd.isna(close):
        return None
    d = pos['direction']

    if sig == 'vwap':
        vwap = row.get('vwap', np.nan)
        if pd.isna(vwap) or vwap == 0:
            return None
        dev = (close - vwap) / vwap
        if d == 'long':
            if dev >= -VWAP_EXIT:
                return 'TARGET'
            if dev <= -(VWAP_STOP * VWAP_ENTRY):
                return 'STOP'
        else:
            if dev <= VWAP_EXIT:
                return 'TARGET'
            if dev >= VWAP_STOP * VWAP_ENTRY:
                return 'STOP'

    elif sig == 'orb':
        orb_h, orb_l = pos['orb_h'], pos['orb_l']
        if orb_h is None:
            return None
        width = orb_h - orb_l
        if d == 'long':
            if close >= orb_h + width * ORB_TGT:
                return 'TARGET'
            if close <= orb_l:
                return 'STOP'
        else:
            if close <= orb_l - width * ORB_TGT:
                return 'TARGET'
            if close >= orb_h:
                return 'STOP'

    elif sig == 'rs':
        if bar_idx - pos['entry_i'] >= RS_HOLD:
            return 'TIME'
        if spy_row is not None:
            rs_now   = row.get('ret_open', 0) - spy_row.get('ret_open', 0)
            rs_entry = pos['rs_entry_ticker'] - pos['rs_entry_spy']
            if d == 'long' and rs_now < rs_entry - RS_REVERSAL:
                return 'RS_REVERSE'
            if d == 'short' and rs_now > rs_entry + RS_REVERSAL:
                return 'RS_REVERSE'

    elif sig == 'vs':
        tgt, stp = pos['vs_target'], pos['vs_stop']
        if tgt is None or stp is None:
            return None
        if d == 'long':
            if close >= tgt:
                return 'TARGET'
            if close <= stp:
                return 'STOP'
        else:
            if close <= tgt:
                return 'TARGET'
            if close >= stp:
                return 'STOP'

    return None


def record(trades, pos, exit_ts, S, bars_left, reason):
    ep  = reprice(pos, S, bars_left)
    pnl = (ep - pos['entry_prem']) * pos['contracts'] * 100
    trades.append({
        'ticker':     pos['ticker'],
        'signal':     pos['signal'],
        'direction':  pos['direction'],
        'entry_ts':   str(pos['entry_ts']),
        'exit_ts':    str(exit_ts),
        'entry_px':   round(pos['entry_px'], 4),
        'exit_px':    round(S, 4),
        'entry_prem': round(pos['entry_prem'], 4),
        'exit_prem':  round(ep, 4),
        'contracts':  pos['contracts'],
        'K':          pos['K'],
        'reason':     reason,
        'pnl':        round(pnl, 2),
        'day':        pos['day'],
    })


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate(ticker_data, signal_type):
    trades   = []
    open_pos = {}   # ticker → position dict

    spy_df       = ticker_data['SPY']
    trading_days = sorted(set(spy_df.index.date))
    rv_tables    = {t: daily_rv_table(df) for t, df in ticker_data.items()}

    for day in trading_days:
        day_slices = {}
        for t in TICKERS:
            if t not in ticker_data:
                continue
            ds   = ticker_data[t]
            mask = ds.index.date == day
            if mask.any():
                day_slices[t] = ds[mask]

        spy_day = day_slices.get('SPY')
        if spy_day is None or spy_day.empty:
            continue

        timestamps = list(spy_day.index)
        n          = len(timestamps)

        # ORB ranges from first ORB_BARS bars of each ticker's day slice
        orb_ranges = {}
        for t, ds in day_slices.items():
            if len(ds) >= ORB_BARS:
                orb_ranges[t] = (ds['high'].iloc[:ORB_BARS].max(),
                                 ds['low'].iloc[:ORB_BARS].min())

        rv_today = {t: rv_tables[t].get(day, 0.20)
                    for t in TICKERS if t in ticker_data}

        for i, ts in enumerate(timestamps):
            bar_time  = ts.strftime('%H:%M')
            is_eod    = bar_time >= EOD_BAR
            bars_left = n - i - 1
            spy_row   = spy_day.loc[ts]

            # ── EOD force close ───────────────────────────────────────────────
            if is_eod:
                for t in list(open_pos.keys()):
                    pos = open_pos.pop(t)
                    ds  = day_slices.get(t)
                    S   = (ds.loc[ts]['close']
                           if ds is not None and ts in ds.index
                           else pos['entry_px'])
                    record(trades, pos, ts, S, 0, 'EOD')
                continue

            # ── Exits ─────────────────────────────────────────────────────────
            for t in list(open_pos.keys()):
                pos = open_pos[t]
                ds  = day_slices.get(t)
                if ds is None or ts not in ds.index:
                    continue
                row    = ds.loc[ts]
                reason = check_exit(signal_type, pos, row, spy_row, i)
                if reason:
                    record(trades, pos, ts, row['close'], bars_left, reason)
                    del open_pos[t]

            # ── Entries ───────────────────────────────────────────────────────
            if len(open_pos) >= MAX_POSITIONS or bar_time >= NO_ENTRY_AFTER:
                continue
            if signal_type == 'orb' and i < ORB_BARS:
                continue
            if signal_type == 'vs' and i < 1:
                continue

            prev_ts = timestamps[i - 1] if i >= 1 else None

            for t in TICKERS:
                if t in open_pos or len(open_pos) >= MAX_POSITIONS:
                    continue
                ds = day_slices.get(t)
                if ds is None or ts not in ds.index:
                    continue
                row  = ds.loc[ts]
                prev = (ds.loc[prev_ts]
                        if prev_ts is not None and prev_ts in ds.index
                        else None)
                orb  = orb_ranges.get(t)

                direction = check_entry(signal_type, row, spy_row, orb, prev)
                if direction is None:
                    continue

                S  = row['close']
                rv = rv_today.get(t, 0.20)
                prem, contracts, K = size_option(S, direction, rv, bars_left)

                # VS-specific: compute target/stop from spike bar
                if signal_type == 'vs' and prev is not None:
                    spike_move = abs(prev.get('close', S) - prev.get('open', S))
                    vs_target  = S + VS_TGT * spike_move if direction == 'long' else S - VS_TGT * spike_move
                    vs_stop    = prev.get('low', S) if direction == 'long' else prev.get('high', S)
                else:
                    vs_target = vs_stop = None

                open_pos[t] = {
                    'ticker':          t,
                    'signal':          signal_type,
                    'direction':       direction,
                    'entry_ts':        ts,
                    'entry_px':        S,
                    'entry_prem':      prem,
                    'contracts':       contracts,
                    'K':               K,
                    'rv30':            rv,
                    'entry_i':         i,
                    'day':             str(day),
                    'vwap_entry':      float(row.get('vwap', np.nan) or np.nan),
                    'orb_h':           orb[0] if orb else None,
                    'orb_l':           orb[1] if orb else None,
                    'rs_entry_ticker': float(row.get('ret_open', 0) or 0),
                    'rs_entry_spy':    float(spy_row.get('ret_open', 0) or 0),
                    'vs_target':       vs_target,
                    'vs_stop':         vs_stop,
                }

        # Day ended — force-close anything still open (belt+suspenders)
        for t in list(open_pos.keys()):
            pos = open_pos.pop(t)
            ds  = day_slices.get(t)
            last_ts = timestamps[-1] if timestamps else pos['entry_ts']
            S = (ds.loc[last_ts]['close']
                 if ds is not None and last_ts in ds.index
                 else pos['entry_px'])
            record(trades, pos, last_ts, S, 0, 'EOD')

    return trades


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_pnl': 0, 'sharpe': 0, 'max_consec_loss': 0,
                'trades_per_active_day': 0}
    pnls  = [t['pnl'] for t in trades]
    wins  = sum(1 for p in pnls if p > 0)
    # Daily P&L for Sharpe
    daily = defaultdict(float)
    for t in trades:
        daily[t['day']] += t['pnl']
    dv    = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * math.sqrt(252)) if np.std(dv) > 0 else 0
    # Max consecutive losses
    mx = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        mx  = max(mx, cur)
    active_days = len(daily)
    return {
        'label':                label,
        'n':                    len(trades),
        'win_rate':             round(wins / len(trades) * 100, 1),
        'total_pnl':            round(sum(pnls), 2),
        'avg_pnl':              round(np.mean(pnls), 2),
        'sharpe':               round(sharpe, 2),
        'max_consec_loss':      mx,
        'trades_per_active_day': round(len(trades) / active_days, 2) if active_days else 0,
    }


def by_ticker(trades):
    d = defaultdict(list)
    for t in trades:
        d[t['ticker']].append(t['pnl'])
    out = {}
    for tk, pnls in d.items():
        w = sum(1 for p in pnls if p > 0)
        out[tk] = {'n': len(pnls), 'total': round(sum(pnls), 2),
                   'win_rate': round(w / len(pnls) * 100, 1)}
    return out


def by_year(trades):
    d = defaultdict(list)
    for t in trades:
        d[t['day'][:4]].append(t['pnl'])
    out = {}
    for yr, pnls in sorted(d.items()):
        w = sum(1 for p in pnls if p > 0)
        out[yr] = {'n': len(pnls), 'total': round(sum(pnls), 2),
                   'win_rate': round(w / len(pnls) * 100, 1)}
    return out


# ── Report ─────────────────────────────────────────────────────────────────────

W = 78

def div(title=''):
    print()
    print('═' * W)
    if title:
        print(f'  {title}')
        print('═' * W)


def print_report(results):
    div('INTRADAY BACKTEST  2023–2025  |  4 Signals  |  10 Tickers  |  15-min bars')

    # ── Comparison table ──────────────────────────────────────────────────────
    div('SIGNAL COMPARISON')
    print(f'  {"Signal":<8} {"Trades":>7} {"Win%":>6} {"Total P&L":>11}'
          f' {"Avg/Trade":>10} {"Sharpe":>8} {"MaxLoss":>8} {"Trades/Day":>11}')
    print('  ' + '─' * 72)
    for sig in SIGNALS:
        s = results[sig]['stats']
        if s['n'] == 0:
            print(f'  {sig:<8}  (no trades)')
            continue
        print(f'  {sig:<8} {s["n"]:>7,} {s["win_rate"]:>5.1f}%'
              f' ${s["total_pnl"]:>10,.0f} ${s["avg_pnl"]:>9,.2f}'
              f' {s["sharpe"]:>8.2f} {s["max_consec_loss"]:>8}'
              f' {s["trades_per_active_day"]:>11.2f}')

    # ── Per-signal detail ─────────────────────────────────────────────────────
    for sig in SIGNALS:
        s       = results[sig]['stats']
        trades  = results[sig]['trades']
        bt      = results[sig]['by_ticker']
        by_yr   = results[sig]['by_year']

        div(f'{sig.upper()}  —  {s["n"]} trades  |  '
            f'${s["total_pnl"]:,.0f} P&L  |  Sharpe {s["sharpe"]:.2f}')

        if not trades:
            print('  (no trades)')
            continue

        # Per-ticker
        print(f'  {"Ticker":<8} {"N":>5} {"Win%":>6} {"Total P&L":>11}')
        print('  ' + '─' * 33)
        for tk, ts in sorted(bt.items(), key=lambda x: -x[1]['total']):
            print(f'  {tk:<8} {ts["n"]:>5} {ts["win_rate"]:>5.1f}%'
                  f' ${ts["total"]:>10,.0f}')
        best_t  = max(bt, key=lambda x: bt[x]['total'])
        worst_t = min(bt, key=lambda x: bt[x]['total'])
        print(f'\n  Best ticker: {best_t} (${bt[best_t]["total"]:,.0f})'
              f'   Worst: {worst_t} (${bt[worst_t]["total"]:,.0f})')

        # Per-year
        print(f'\n  {"Year":<6} {"N":>5} {"Win%":>6} {"Total P&L":>11}')
        print('  ' + '─' * 31)
        for yr, ys in by_yr.items():
            print(f'  {yr:<6} {ys["n"]:>5} {ys["win_rate"]:>5.1f}%'
                  f' ${ys["total"]:>10,.0f}')
        if by_yr:
            best_y  = max(by_yr, key=lambda x: by_yr[x]['total'])
            worst_y = min(by_yr, key=lambda x: by_yr[x]['total'])
            print(f'\n  Best year: {best_y} (${by_yr[best_y]["total"]:,.0f})'
                  f'   Worst: {worst_y} (${by_yr[worst_y]["total"]:,.0f})')

        # Exit reasons
        ec = Counter(t['reason'] for t in trades)
        print(f'\n  Exit reasons:')
        for reason, cnt in sorted(ec.items(), key=lambda x: -x[1]):
            print(f'    {reason:<15} {cnt:>5}  ({cnt/len(trades)*100:.1f}%)')

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ticker_data = load_all()

    if 'SPY' not in ticker_data:
        print('SPY data missing — check credentials.')
        return

    total_days = len(set(ticker_data['SPY'].index.date))
    print(f'Trading days in dataset: {total_days}')
    print()
    print('Running simulations...')

    results = {}
    for sig in SIGNALS:
        print(f'  {sig.upper():<6} ', end='', flush=True)
        trades = simulate(ticker_data, sig)
        s      = compute_stats(trades, sig)
        results[sig] = {
            'stats':     s,
            'trades':    trades,
            'by_ticker': by_ticker(trades),
            'by_year':   by_year(trades),
        }
        print(f'{s["n"]:>5} trades  ${s["total_pnl"]:>10,.0f}  '
              f'win {s["win_rate"]:.1f}%  Sharpe {s["sharpe"]:.2f}')

    print_report(results)

    # Save (cap trades list to 1000/signal to keep file manageable)
    save = {}
    for sig, data in results.items():
        save[sig] = {
            'stats':     data['stats'],
            'by_ticker': data['by_ticker'],
            'by_year':   data['by_year'],
            'trades':    data['trades'][:1000],
        }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f'Results saved → {RESULTS_FILE}')


if __name__ == '__main__':
    main()
