#!/usr/bin/env python3
"""
Regime-switching backtest: buy-side (SPY + QQQ options) by default,
auto-switches to sell-side (SPY short put credit spread) when ALL of:
  • SPY IV rank > 40  AND  QQQ IV rank > 40
  • SPY within 3 % of EMA20  AND  QQQ within 3 % of EMA20
  • VIX 25–35  (elevated but not panic)

Two switcher variants are compared:
  switcher_d1  – switch immediately when conditions change (original)
  switcher_d5  – require 5 consecutive days before confirming any switch

Buy-side  : exact same indicators/exits as backtest.py  (daily bars)
Sell-side : exact same spread/exits as backtest_sellside.py (BS pricing)
Both sized to $125 max risk / $5 k simulated account.

Results → switcher_backtest_results.json
"""

import json
import math
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

_DIR         = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_DIR, 'switcher_backtest_results.json')
YEARS        = [2020, 2021, 2022, 2023, 2024, 2025]
RISK_FREE      = 0.05
RISK_PER_TRADE = 125.0
ACCOUNT_SIZE   = 5_000.0

# ── Regime trigger (ALL must be true on BOTH tickers) ─────────────────────────
SELL_IV_MIN   = 40
SELL_EMA_BAND = 0.03
SELL_VIX_LOW  = 25
SELL_VIX_HIGH = 35

# ── Buy-side (backtest.py defaults) ───────────────────────────────────────────
BUY_VIX_LOW   = 15
BUY_VIX_HIGH  = 35
BUY_IV_MAX    = 50
BUY_DTE       = 7
BUY_STOP      = -0.45
BUY_TAKE_HALF = 0.50
BUY_FULL      = 1.00
BUY_MAX_HOLD  = 7

# ── Sell-side (backtest_sellside.py) ─────────────────────────────────────────
SELL_SHORT_OTM   = 0.05
SELL_LONG_OTM    = 0.10
SELL_HOLD_BARS   = 21
SELL_TAKE_PROFIT = 0.50
SELL_STOP_MULT   = 2.0


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def _bs(S, K, T, r, sig, typ='call'):
    if T < 1e-6 or sig < 1e-6 or S <= 0 or K <= 0:
        iv = max(S - K, 0) if typ == 'call' else max(K - S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == 'call':
        return max(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 0.01)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.01)


def _bs_put(S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6:
        return max(K - S, 0.0001)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.0001)


# ── Indicators + data loading ─────────────────────────────────────────────────

def _add_indicators(df):
    df = df.copy()
    c, v = df['Close'], df['Volume']
    df['ema20'] = c.ewm(span=20, adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    d  = c.diff()
    ag = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['rsi']   = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26
    df['mhist'] = macd - macd.ewm(span=9, adjust=False).mean()
    df['volma'] = v.rolling(20).mean()
    r, m = df['rsi'], df['mhist']
    df['rsi_up3']    = (r > r.shift(1)) & (r.shift(1) > r.shift(2))
    df['rsi_dn3']    = (r < r.shift(1)) & (r.shift(1) < r.shift(2))
    df['mhist_bull'] = (m > 0) & (m > m.shift(1))
    df['mhist_bear'] = (m < 0) & (m < m.shift(1))
    lr = np.log(c / c.shift(1))
    df['rv30'] = lr.rolling(30).std() * math.sqrt(252)
    return df


def _iv_rank_series(rv_series):
    rv_arr  = rv_series.to_numpy(dtype=float)
    iv_rank = np.full(len(rv_arr), np.nan)
    for i in range(252, len(rv_arr)):
        curr  = rv_arr[i]
        hist  = rv_arr[i - 252:i]
        valid = hist[~np.isnan(hist)]
        if np.isnan(curr) or len(valid) < 20:
            continue
        iv_rank[i] = float((valid < curr).sum()) / len(valid) * 100.0
    return pd.Series(iv_rank, index=rv_series.index)


def load_data():
    print('Fetching SPY, QQQ, VIX (2019–2025) …', end=' ', flush=True)
    spy_raw = yf.download('SPY',  start='2019-01-01', end='2026-01-01',
                          auto_adjust=True, progress=False)
    qqq_raw = yf.download('QQQ',  start='2019-01-01', end='2026-01-01',
                          auto_adjust=True, progress=False)
    vix_raw = yf.download('^VIX', start='2019-01-01', end='2026-01-01',
                          auto_adjust=True, progress=False)
    print('done')

    def clean(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    spy_raw, qqq_raw, vix_raw = clean(spy_raw), clean(qqq_raw), clean(vix_raw)
    spy = _add_indicators(spy_raw)
    qqq = _add_indicators(qqq_raw)

    print('Computing IV ranks …', end=' ', flush=True)
    spy['iv_rank'] = _iv_rank_series(spy['rv30'])
    qqq['iv_rank'] = _iv_rank_series(qqq['rv30'])
    print('done\n')

    combined = pd.DataFrame(index=spy.index)
    for col in ['Close', 'Volume', 'ema20', 'ema21', 'ema50', 'rsi',
                'mhist_bull', 'mhist_bear', 'rsi_up3', 'rsi_dn3',
                'volma', 'rv30', 'iv_rank']:
        combined[f'spy_{col}'] = spy[col]
    for col in ['Close', 'Volume', 'ema20', 'ema21', 'rsi',
                'mhist_bull', 'mhist_bear', 'rsi_up3', 'rsi_dn3',
                'volma', 'rv30', 'iv_rank']:
        combined[f'qqq_{col}'] = qqq[col].reindex(spy.index).ffill()
    combined['vix'] = vix_raw['Close'].reindex(spy.index).ffill()
    return combined


# ── Regime detection ──────────────────────────────────────────────────────────

def _detect_regime(row):
    vix = row.get('vix')
    if pd.isna(vix) or not (SELL_VIX_LOW <= vix <= SELL_VIX_HIGH):
        return 'buy'
    for pfx in ('spy', 'qqq'):
        iv    = row.get(f'{pfx}_iv_rank')
        close = row.get(f'{pfx}_Close')
        ema20 = row.get(f'{pfx}_ema20')
        if pd.isna(iv) or iv <= SELL_IV_MIN:
            return 'buy'
        if pd.isna(close) or pd.isna(ema20) or ema20 <= 0:
            return 'buy'
        if abs(close - ema20) / ema20 > SELL_EMA_BAND:
            return 'buy'
    return 'sell'


def _switch_reason(row, new_regime):
    if new_regime == 'sell':
        return (f"SPY IV={row.get('spy_iv_rank', 0):.0f}% "
                f"QQQ IV={row.get('qqq_iv_rank', 0):.0f}% "
                f"VIX={row.get('vix', 0):.1f} — all conditions met")
    vix     = row.get('vix', 0)
    spy_iv  = row.get('spy_iv_rank', 0)
    qqq_iv  = row.get('qqq_iv_rank', 0)
    spy_ema_pct = (abs(row.get('spy_Close', 0) - row.get('spy_ema20', 1))
                   / max(row.get('spy_ema20', 1), 1e-9) * 100)
    qqq_ema_pct = (abs(row.get('qqq_Close', 0) - row.get('qqq_ema20', 1))
                   / max(row.get('qqq_ema20', 1), 1e-9) * 100)
    reasons = []
    if not (SELL_VIX_LOW <= vix <= SELL_VIX_HIGH):
        reasons.append(f'VIX={vix:.1f} (outside 25–35)')
    if spy_iv <= SELL_IV_MIN:
        reasons.append(f'SPY IV={spy_iv:.0f}% (≤{SELL_IV_MIN})')
    if qqq_iv <= SELL_IV_MIN:
        reasons.append(f'QQQ IV={qqq_iv:.0f}% (≤{SELL_IV_MIN})')
    if spy_ema_pct > SELL_EMA_BAND * 100:
        reasons.append(f'SPY off EMA20 by {spy_ema_pct:.1f}%')
    if qqq_ema_pct > SELL_EMA_BAND * 100:
        reasons.append(f'QQQ off EMA20 by {qqq_ema_pct:.1f}%')
    return '; '.join(reasons) if reasons else 'unknown'


# ── Buy-side helpers ──────────────────────────────────────────────────────────

def _buy_signal(row, spy_above, pfx):
    close   = row.get(f'{pfx}_Close')
    ema21   = row.get(f'{pfx}_ema21')
    rsi_val = row.get(f'{pfx}_rsi')
    mh_bull = row.get(f'{pfx}_mhist_bull', False)
    mh_bear = row.get(f'{pfx}_mhist_bear', False)
    ru3     = row.get(f'{pfx}_rsi_up3', False)
    rd3     = row.get(f'{pfx}_rsi_dn3', False)
    volume  = row.get(f'{pfx}_Volume')
    volma   = row.get(f'{pfx}_volma')
    if any(pd.isna(x) for x in [close, ema21, rsi_val, volume, volma]):
        return None
    if volma <= 0:
        return None
    vol_ok = volume > volma
    if spy_above:
        bull = sum([close > ema21, (30 <= rsi_val <= 50) and bool(ru3),
                    bool(mh_bull), vol_ok])
        if bull >= 3:
            return 'bullish'
    else:
        bear = sum([close < ema21, (50 <= rsi_val <= 70) and bool(rd3),
                    bool(mh_bear), vol_ok])
        if bear >= 3:
            return 'bearish'
    return None


def _update_buy_pos(pos, cur_close):
    pos['bars_held'] += 1
    bars  = pos['bars_held']
    K, e0, sigma = pos['K'], pos['entry_prem'], pos['sigma']
    typ   = 'call' if pos['direction'] == 'bullish' else 'put'
    T     = max((BUY_DTE - bars) / 365.0, 0.001)
    cur   = _bs(float(cur_close), K, T, RISK_FREE, sigma, typ)
    pnl   = (cur - e0) / e0
    if bars >= BUY_MAX_HOLD:
        return 'TIME_EXIT', pnl, pnl * RISK_PER_TRADE
    if pnl <= BUY_STOP:
        return 'STOP', pnl, pnl * RISK_PER_TRADE
    if pnl >= BUY_FULL:
        return 'FULL_TARGET', pnl, pnl * RISK_PER_TRADE
    if not pos['partial_taken'] and pnl >= BUY_TAKE_HALF:
        pos['partial_taken'] = True
    if pos['partial_taken'] and pnl < BUY_TAKE_HALF * 0.5:
        avg_pnl = (BUY_TAKE_HALF + pnl) / 2
        return 'TRAIL_EXIT', avg_pnl, avg_pnl * RISK_PER_TRADE
    return None


# ── Sell-side helpers ─────────────────────────────────────────────────────────

def _try_sell_entry(row):
    iv, close = row.get('spy_iv_rank'), row.get('spy_Close')
    rv, ema20 = row.get('spy_rv30'),    row.get('spy_ema20')
    vix       = row.get('vix')
    if any(pd.isna(x) for x in [iv, close, rv, ema20, vix]):
        return None
    if iv <= SELL_IV_MIN or rv <= 0:
        return None
    if not (SELL_VIX_LOW <= vix <= SELL_VIX_HIGH):
        return None
    if abs(close - ema20) / ema20 > SELL_EMA_BAND:
        return None
    K_short  = close * (1 - SELL_SHORT_OTM)
    K_long   = close * (1 - SELL_LONG_OTM)
    T        = SELL_HOLD_BARS / 252.0
    net_prem = _bs_put(close, K_short, T, RISK_FREE, rv) - _bs_put(close, K_long, T, RISK_FREE, rv)
    if net_prem < 0.001:
        return None
    max_loss_shr = (K_short - K_long) - net_prem
    if max_loss_shr <= 0:
        return None
    return {
        'entry_close': float(close), 'K_short': K_short, 'K_long': K_long,
        'net_premium': net_prem, 'frac_contracts': RISK_PER_TRADE / (max_loss_shr * 100.0),
        'sigma_entry': float(rv), 'bars_held': 0,
    }


def _update_sell_pos(pos, close, rv):
    pos['bars_held'] += 1
    bars, K_short, K_long = pos['bars_held'], pos['K_short'], pos['K_long']
    net_prem, frac_c      = pos['net_premium'], pos['frac_contracts']
    cur_rv = (rv if (rv is not None and not np.isnan(rv) and rv > 0) else pos['sigma_entry'])

    if bars >= SELL_HOLD_BARS:
        if close >= K_short:
            spread_val = 0.0
        elif close >= K_long:
            spread_val = K_short - close
        else:
            spread_val = K_short - K_long
        return 'EXPIRY', (net_prem - spread_val) * frac_c * 100.0

    T_rem      = (SELL_HOLD_BARS - bars) / 252.0
    spread_val = (_bs_put(close, K_short, T_rem, RISK_FREE, cur_rv)
                - _bs_put(close, K_long,  T_rem, RISK_FREE, cur_rv))
    pnl_shr = net_prem - spread_val
    if pnl_shr >= SELL_TAKE_PROFIT * net_prem:
        return 'TAKE_PROFIT', pnl_shr * frac_c * 100.0
    if pnl_shr <= -SELL_STOP_MULT * net_prem:
        return 'STOP_LOSS', pnl_shr * frac_c * 100.0
    return None


# ── Core simulation loop ──────────────────────────────────────────────────────

def run_simulation(df, mode='switcher', dwell=1):
    """
    mode  : 'switcher' | 'buy_only' | 'sell_only'
    dwell : minimum consecutive days with new-regime conditions before
            a switch is confirmed. 1 = switch immediately (original).
    Returns: (trades, regime_log, switch_log)
    """
    trades, open_buy, open_sell = [], {}, None
    cur_regime, regime_start    = None, None
    pending_count               = 0   # days building toward a dwell-confirmed switch
    switch_log, regime_log      = [], []

    year_data = df[(df.index.year >= 2020) & (df.index.year <= 2025)]

    for dt, row in year_data.iterrows():
        year = dt.year

        # ── Raw conditions for today ──────────────────────────────────────────
        if mode == 'buy_only':
            raw_regime = 'buy'
        elif mode == 'sell_only':
            raw_regime = 'sell'
        else:
            raw_regime = _detect_regime(row)

        # ── Apply dwell filter (switcher mode only) ───────────────────────────
        if mode == 'switcher':
            if cur_regime is None:
                cur_regime    = raw_regime
                regime_start  = dt
                pending_count = 0
            elif raw_regime == cur_regime:
                # Conditions align with current regime — reset any pending streak
                pending_count = 0
            else:
                # Conditions differ — build confirmation streak
                pending_count += 1
                if pending_count >= dwell:
                    switch_log.append({
                        'switch_date':          str(dt.date()),
                        'from_regime':          cur_regime,
                        'to_regime':            raw_regime,
                        'prev_regime_days':     (dt - regime_start).days,
                        'confirmed_after_days': pending_count,
                        'trigger':              _switch_reason(row, raw_regime),
                    })
                    cur_regime    = raw_regime
                    regime_start  = dt
                    pending_count = 0
                # pending_count < dwell: stay in cur_regime, do NOT update regime_start
        else:
            # buy_only / sell_only: fixed, no dwell needed
            if cur_regime is None:
                cur_regime   = raw_regime
                regime_start = dt

        regime_log.append({'date': str(dt.date()), 'regime': cur_regime, 'year': year})

        # ── Update / exit open buy positions (always, regardless of regime) ───
        for tkr in list(open_buy.keys()):
            pos   = open_buy[tkr]
            pfx   = 'spy' if tkr == 'SPY' else 'qqq'
            close = row.get(f'{pfx}_Close')
            if pd.isna(close):
                continue
            result = _update_buy_pos(pos, float(close))
            if result is None:
                continue
            reason, pnl_pct, pnl_dollar = result
            trades.append({
                'mode': mode, 'side': 'buy', 'ticker': tkr,
                'direction': pos['direction'],
                'entry_date': str(pos['entry_date'].date()), 'exit_date': str(dt.date()),
                'bars_held': pos['bars_held'], 'exit_reason': reason,
                'entry_regime': pos['entry_regime'],
                'pnl_pct': round(pnl_pct * 100, 2), 'pnl_dollar': round(pnl_dollar, 2),
                'result': 'WIN' if pnl_dollar > 0 else 'LOSS', 'year': year,
            })
            del open_buy[tkr]

        # ── Update / exit open sell position (always, regardless of regime) ───
        if open_sell is not None:
            spy_close = row.get('spy_Close')
            spy_rv    = row.get('spy_rv30')
            if not pd.isna(spy_close):
                result = _update_sell_pos(
                    open_sell, float(spy_close),
                    None if pd.isna(spy_rv) else float(spy_rv))
                if result is not None:
                    reason, pnl_dollar = result
                    trades.append({
                        'mode': mode, 'side': 'sell', 'ticker': 'SPY',
                        'direction': 'short_put_spread',
                        'entry_date': str(open_sell['entry_date'].date()),
                        'exit_date': str(dt.date()),
                        'bars_held': open_sell['bars_held'], 'exit_reason': reason,
                        'entry_regime': open_sell['entry_regime'],
                        'net_premium': round(open_sell['net_premium'], 4),
                        'K_short': round(open_sell['K_short'], 2),
                        'K_long':  round(open_sell['K_long'],  2),
                        'pnl_dollar': round(pnl_dollar, 2),
                        'pnl_pct': round(pnl_dollar / RISK_PER_TRADE * 100, 2),
                        'result': 'WIN' if pnl_dollar > 0 else 'LOSS', 'year': year,
                    })
                    open_sell = None

        # ── Enter new positions per current (confirmed) regime ────────────────
        if cur_regime == 'buy' and mode != 'sell_only':
            vix_val, spy_close, spy_ema50 = (
                row.get('vix'), row.get('spy_Close'), row.get('spy_ema50'))
            if not any(pd.isna(x) for x in [vix_val, spy_close, spy_ema50]):
                if BUY_VIX_LOW <= vix_val <= BUY_VIX_HIGH:
                    spy_above = float(spy_close) > float(spy_ema50)
                    for tkr, pfx in [('SPY', 'spy'), ('QQQ', 'qqq')]:
                        if tkr in open_buy:
                            continue
                        iv_rank = row.get(f'{pfx}_iv_rank')
                        rv_val  = row.get(f'{pfx}_rv30')
                        if pd.isna(iv_rank) or iv_rank > BUY_IV_MAX:
                            continue
                        if pd.isna(rv_val) or rv_val <= 0:
                            continue
                        direction = _buy_signal(row, spy_above, pfx)
                        if direction is None:
                            continue
                        close = float(row[f'{pfx}_Close'])
                        open_buy[tkr] = {
                            'ticker': tkr, 'direction': direction,
                            'entry_date': dt, 'entry_close': close,
                            'K': close,
                            'entry_prem': _bs(close, close, BUY_DTE / 365.0, RISK_FREE,
                                             float(rv_val),
                                             'call' if direction == 'bullish' else 'put'),
                            'sigma': float(rv_val), 'bars_held': 0,
                            'partial_taken': False, 'entry_regime': cur_regime,
                        }

        elif cur_regime == 'sell' and mode != 'buy_only':
            if open_sell is None:
                new_pos = _try_sell_entry(row)
                if new_pos is not None:
                    new_pos['entry_date']   = dt
                    new_pos['entry_regime'] = cur_regime
                    open_sell = new_pos

    # ── Force-close anything open at end of 2025 ─────────────────────────────
    last_row, last_dt = year_data.iloc[-1], year_data.index[-1]

    for tkr, pos in list(open_buy.items()):
        pfx   = 'spy' if tkr == 'SPY' else 'qqq'
        close = float(last_row[f'{pfx}_Close'])
        T     = max((BUY_DTE - pos['bars_held']) / 365.0, 0.001)
        typ   = 'call' if pos['direction'] == 'bullish' else 'put'
        cur   = _bs(close, pos['K'], T, RISK_FREE, pos['sigma'], typ)
        pnl   = (cur - pos['entry_prem']) / pos['entry_prem']
        trades.append({
            'mode': mode, 'side': 'buy', 'ticker': tkr,
            'direction': pos['direction'],
            'entry_date': str(pos['entry_date'].date()), 'exit_date': str(last_dt.date()),
            'bars_held': pos['bars_held'], 'exit_reason': 'YEAR_END',
            'entry_regime': pos['entry_regime'],
            'pnl_pct': round(pnl * 100, 2), 'pnl_dollar': round(pnl * RISK_PER_TRADE, 2),
            'result': 'WIN' if pnl > 0 else 'LOSS', 'year': last_dt.year,
        })

    if open_sell is not None:
        close = float(last_row['spy_Close'])
        K_short, K_long = open_sell['K_short'], open_sell['K_long']
        net_prem, frac_c = open_sell['net_premium'], open_sell['frac_contracts']
        spread_val = (0.0 if close >= K_short
                      else K_short - close if close >= K_long
                      else K_short - K_long)
        pnl_dollar = (net_prem - spread_val) * frac_c * 100.0
        trades.append({
            'mode': mode, 'side': 'sell', 'ticker': 'SPY',
            'direction': 'short_put_spread',
            'entry_date': str(open_sell['entry_date'].date()), 'exit_date': str(last_dt.date()),
            'bars_held': open_sell['bars_held'], 'exit_reason': 'YEAR_END',
            'entry_regime': open_sell['entry_regime'],
            'net_premium': round(net_prem, 4),
            'K_short': round(K_short, 2), 'K_long': round(K_long, 2),
            'pnl_dollar': round(pnl_dollar, 2),
            'pnl_pct': round(pnl_dollar / RISK_PER_TRADE * 100, 2),
            'result': 'WIN' if pnl_dollar > 0 else 'LOSS', 'year': last_dt.year,
        })

    return trades, regime_log, switch_log


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(trades):
    if not trades:
        return {'taken': 0, 'wins': 0, 'losses': 0, 'win_pct': 0.0,
                'pnl_dollar': 0.0, 'max_dd_pct': 0.0, 'sharpe': 0.0}
    dolls = [t['pnl_dollar'] for t in trades]
    wins  = [d for d in dolls if d > 0]
    eq    = ACCOUNT_SIZE + np.cumsum([0.0] + dolls)
    peak  = np.maximum.accumulate(eq)
    arr   = np.array(dolls)
    std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 1e-9
    return {
        'taken':      len(trades),
        'wins':       len(wins),
        'losses':     len(trades) - len(wins),
        'win_pct':    len(wins) / len(trades) * 100.0,
        'pnl_dollar': float(sum(dolls)),
        'max_dd_pct': float(((eq - peak) / peak * 100.0).min()),
        'sharpe':     float(arr.mean() / std_r) * math.sqrt(len(arr)) if std_r > 1e-9 else 0.0,
    }


def _regime_counts(regime_log, year=None):
    subset = [r for r in regime_log if year is None or r['year'] == year]
    buy   = sum(1 for r in subset if r['regime'] == 'buy')
    sell  = sum(1 for r in subset if r['regime'] == 'sell')
    total = buy + sell
    return buy, sell, total


# ── Reference data from prior backtests ───────────────────────────────────────

def _load_reference():
    buy_ref, sell_ref = {}, {}
    try:
        with open(os.path.join(_DIR, 'backtest_results.json')) as f:
            br = json.load(f)
        for yr in YEARS:
            s = br['years'][str(yr)]['stats']
            pnl = (s['wins'] * s['avg_winner_pct'] / 100 * RISK_PER_TRADE +
                   s['losses'] * s['avg_loser_pct']  / 100 * RISK_PER_TRADE)
            buy_ref[yr] = {'taken': s['total_trades'],
                           'win_pct': round(s['win_rate'] * 100, 1),
                           'pnl_dollar': round(pnl, 0), 'sharpe': s['sharpe']}
    except Exception:
        pass
    try:
        with open(os.path.join(_DIR, 'sellside_backtest_results.json')) as f:
            sr = json.load(f)
        for yr in YEARS:
            s = sr['years'][str(yr)]['stats']
            sell_ref[yr] = {'taken': s['taken'], 'win_pct': round(s['win_pct'], 1),
                            'pnl_dollar': round(s['total_pnl'], 0), 'sharpe': s['sharpe']}
    except Exception:
        pass
    return buy_ref, sell_ref


# ── Report ────────────────────────────────────────────────────────────────────

def _s(v):
    return '+' if v >= 0 else ''


def _fmt_stat(s):
    p = s['pnl_dollar']
    return (f'{s["taken"]:>3} {s["win_pct"]:>5.1f}% '
            f'{_s(p)}${abs(p):>6.0f} {s["sharpe"]:>+6.2f}')


def print_report(sw_d1_trades, sw_d1_regime, sw_d1_switches,
                 sw_d5_trades, sw_d5_regime, sw_d5_switches,
                 bu_trades, se_trades):

    W = 116
    bar = '═' * W

    # ── SECTION 1: Regime calendar, d1 vs d5 ─────────────────────────────────
    print(bar)
    print('  SECTION 1 — REGIME CALENDAR  (buy/sell day counts per year)')
    print(bar)
    print(f'  {"Year":<6}  {"──── SWITCHER-D1 (immediate) ────":^34}  '
          f'{"──── SWITCHER-D5 (5-day dwell) ────":^36}')
    print(f'  {"":6}  {"Buy days":>9} {"Sell days":>10} {"Sell%":>6}  '
          f'{"Buy days":>9} {"Sell days":>10} {"Sell%":>7}  {"Sell days saved":>15}')
    print('  ' + '─' * (W - 2))

    for yr in YEARS:
        b1, s1, t1 = _regime_counts(sw_d1_regime, yr)
        b5, s5, t5 = _regime_counts(sw_d5_regime, yr)
        saved = s1 - s5
        print(f'  {yr:<6}  {b1:>9} {s1:>10} {s1/t1*100:>5.1f}%  '
              f'{b5:>9} {s5:>10} {s5/t5*100:>6.1f}%  {saved:>+15}')

    b1_all, s1_all, t1 = _regime_counts(sw_d1_regime)
    b5_all, s5_all, t5 = _regime_counts(sw_d5_regime)
    print('  ' + '─' * (W - 2))
    print(f'  {"ALL":<6}  {b1_all:>9} {s1_all:>10} {s1_all/t1*100:>5.1f}%  '
          f'{b5_all:>9} {s5_all:>10} {s5_all/t5*100:>6.1f}%  {s1_all-s5_all:>+15}')
    print()

    # ── SECTION 2: Year-by-year performance ───────────────────────────────────
    print(bar)
    print('  SECTION 2 — YEAR-BY-YEAR PERFORMANCE')
    print('  ⚠  Buy-only / Sell-only are re-simulated from daily bars in this script.')
    print('     Do not compare dollar figures directly to backtest.py (hourly) or')
    print('     backtest_sellside.py (VIX 18–35). Reference numbers shown separately.')
    print(bar)

    col_hdr = 'Trades Win%   P&L$  Sharpe'
    print(f'  {"Year":<6}  {"BUY-ONLY":^28}  {"SELL-ONLY":^28}  '
          f'{"SWITCHER-D1":^28}  {"SWITCHER-D5":^28}')
    print(f'  {"":6}  {col_hdr:^28}  {col_hdr:^28}  {col_hdr:^28}  {col_hdr:^28}')
    print('  ' + '─' * (W - 2))

    d1_all = _stats(sw_d1_trades)
    d5_all = _stats(sw_d5_trades)
    bu_all = _stats(bu_trades)
    se_all = _stats(se_trades)

    for yr in YEARS:
        bu = _stats([t for t in bu_trades    if t['year'] == yr])
        se = _stats([t for t in se_trades    if t['year'] == yr])
        d1 = _stats([t for t in sw_d1_trades if t['year'] == yr])
        d5 = _stats([t for t in sw_d5_trades if t['year'] == yr])
        best = max(bu['pnl_dollar'], se['pnl_dollar'],
                   d1['pnl_dollar'], d5['pnl_dollar'])
        def star(s):
            return '★' if abs(s['pnl_dollar'] - best) < 0.01 else ' '
        print(f'  {yr:<6}  {star(bu)}{_fmt_stat(bu)}  '
              f'{star(se)}{_fmt_stat(se)}  '
              f'{star(d1)}{_fmt_stat(d1)}  '
              f'{star(d5)}{_fmt_stat(d5)}')

    print('  ' + '─' * (W - 2))
    print(f'  {"ALL":<6}  {" "}{_fmt_stat(bu_all)}  '
          f'{" "}{_fmt_stat(se_all)}  '
          f'{" "}{_fmt_stat(d1_all)}  '
          f'{" "}{_fmt_stat(d5_all)}')
    print()
    print('  ★ = best dollar P&L for that year across all four strategies')
    print()

    # Reference from original files
    buy_ref, sell_ref = _load_reference()
    if buy_ref or sell_ref:
        print('  Reference from original backtest files (different VIX/hourly parameters):')
        print(f'  {"Year":<6}  {"Orig Buy Win%":>14}  {"Orig Buy P&L$*":>15}'
              f'  {"Orig Sell Win%":>15}  {"Orig Sell P&L$":>14}')
        print('  ' + '─' * 70)
        for yr in YEARS:
            b = buy_ref.get(yr, {}); s = sell_ref.get(yr, {})
            bw = f'{b.get("win_pct","n/a"):>5}%' if b else '  n/a'
            bp = (f'{_s(b["pnl_dollar"])}${abs(b["pnl_dollar"]):>6.0f}'
                  if b else '    n/a')
            sw = f'{s.get("win_pct","n/a"):>5}%' if s else '  n/a'
            sp = (f'{_s(s["pnl_dollar"])}${abs(s["pnl_dollar"]):>6.0f}'
                  if s else '    n/a')
            print(f'  {yr:<6}  {bw:>14}  {bp:>15}  {sw:>15}  {sp:>14}')
        print('  * Buy P&L approximated: avg_win% × wins + avg_loss% × losses × $125')
        print()

    # ── SECTION 3: Switch logs ────────────────────────────────────────────────
    print(bar)
    print('  SECTION 3 — SWITCH LOG COMPARISON')
    print(bar)
    print()
    print(f'  D1 (immediate):  {len(sw_d1_switches):>3} switches over 6 years  '
          f'({len(sw_d1_switches)/6:.1f}/year)  →  '
          f'{"STABLE" if len(sw_d1_switches) <= 6 else "MODERATE" if len(sw_d1_switches) <= 18 else "NOISY"}')
    print(f'  D5 (5-day dwell):{len(sw_d5_switches):>3} switches over 6 years  '
          f'({len(sw_d5_switches)/6:.1f}/year)  →  '
          f'{"STABLE" if len(sw_d5_switches) <= 6 else "MODERATE" if len(sw_d5_switches) <= 18 else "NOISY"}')
    print(f'  Noise reduction: {len(sw_d1_switches) - len(sw_d5_switches)} fewer switches '
          f'({(1 - len(sw_d5_switches)/max(len(sw_d1_switches),1))*100:.0f}% reduction)')
    print()

    # D1 switch log — compact summary by year
    print('  ── D1 switch log (summarised by year) ──────────────────────────────')
    for yr in YEARS:
        yr_sw = [s for s in sw_d1_switches
                 if int(s['switch_date'][:4]) == yr
                 or (int(s['switch_date'][:4]) == yr - 1
                     and s['to_regime'] == 'sell')]
        to_sell = [s for s in sw_d1_switches
                   if s['to_regime'] == 'sell' and int(s['switch_date'][:4]) == yr]
        to_buy  = [s for s in sw_d1_switches
                   if s['to_regime'] == 'buy'  and int(s['switch_date'][:4]) == yr]
        sell_windows = [s['confirmed_after_days'] if 'confirmed_after_days' in s
                        else 1 for s in to_sell]
        avg_sell_dur = (sum(sell_windows) / len(sell_windows)
                        if sell_windows else 0)
        b_days, s_days, _ = _regime_counts(sw_d1_regime, yr)
        print(f'    {yr}: →sell {len(to_sell):>2}×  →buy {len(to_buy):>2}×  '
              f'sell-days {s_days:>3}  buy-days {b_days:>3}')

    print()

    # D5 switch log — full detail (should be short)
    if not sw_d5_switches:
        print('  ── D5 switch log: NO SWITCHES — system stayed in buy-side mode all 6 years ──')
        print('     The 5-day dwell requirement was never met. Every sell-regime window')
        print('     lasted fewer than 5 consecutive days before conditions broke.')
    else:
        print(f'  ── D5 switch log (full, {len(sw_d5_switches)} switches) ─────────────────────────')
        print(f'  {"#":>3}  {"Date":>12}  {"From→To":>12}  '
              f'{"Prev dur":>9}  {"Confirmed":>10}  Trigger')
        print('  ' + '─' * 105)
        for i, sw in enumerate(sw_d5_switches, 1):
            arrow = f'{sw["from_regime"].upper()[:3]}→{sw["to_regime"].upper()[:4]}'
            conf  = sw.get('confirmed_after_days', '?')
            print(f'  {i:>3}  {sw["switch_date"]:>12}  {arrow:>12}  '
                  f'{sw["prev_regime_days"]:>7}d  '
                  f'{str(conf):>8}d  {sw["trigger"]}')

        # Sell-side windows in d5
        sell_enters = [s for s in sw_d5_switches if s['to_regime'] == 'sell']
        buy_returns = [s for s in sw_d5_switches if s['to_regime'] == 'buy']
        print()
        if sell_enters:
            print('  Sell-side windows confirmed under D5:')
            for e in sell_enters:
                # find matching return to buy
                ret = next((r for r in buy_returns
                            if r['switch_date'] > e['switch_date']), None)
                end   = ret['switch_date'] if ret else '2025-12-31'
                start = e['switch_date']
                dur   = (pd.to_datetime(end) - pd.to_datetime(start)).days
                print(f'    {start} → {end}  ({dur} calendar days in sell mode)')
    print()

    # ── SECTION 4: Five questions, D1 vs D5 ──────────────────────────────────
    print(bar)
    print('  SECTION 4 — FIVE QUESTIONS  (answered for both switcher variants)')
    print(bar)

    def q_row(label, v1, v5, unit='$', fmt='.0f'):
        sign1 = _s(v1) if unit == '$' else ''
        sign5 = _s(v5) if unit == '$' else ''
        pre1  = f'{sign1}{unit}' if unit == '$' else ''
        pre5  = f'{sign5}{unit}' if unit == '$' else ''
        suf   = unit if unit != '$' else ''
        print(f'     {label:<30}  D1: {pre1}{abs(v1):{fmt}}{suf}   '
              f'D5: {pre5}{abs(v5):{fmt}}{suf}')

    print()
    print('  Q1. Did either switcher beat buy-only in total 6-year dollar P&L?')
    print()
    bu_pnl = bu_all['pnl_dollar']
    d1_pnl = d1_all['pnl_dollar']
    d5_pnl = d5_all['pnl_dollar']
    q_row('Buy-only P&L',    bu_pnl, bu_pnl)
    q_row('Switcher P&L',    d1_pnl, d5_pnl)
    q_row('Delta vs buy-only', d1_pnl - bu_pnl, d5_pnl - bu_pnl)
    print()
    for sw_pnl, name in [(d1_pnl,'D1'),(d5_pnl,'D5')]:
        delta = sw_pnl - bu_pnl
        if delta > 0:
            print(f'     {name}: YES — switcher beats buy-only by ${delta:+.0f}')
        elif abs(delta) < 10:
            print(f'     {name}: ESSENTIALLY TIED (delta = ${delta:+.0f})')
        else:
            print(f'     {name}: NO — buy-only beats switcher by ${abs(delta):.0f}')
    print()

    print('  Q2. Did either switcher beat sell-only?')
    print()
    se_pnl = se_all['pnl_dollar']
    q_row('Sell-only P&L', se_pnl, se_pnl)
    q_row('Switcher P&L',  d1_pnl, d5_pnl)
    for sw_pnl, name in [(d1_pnl,'D1'),(d5_pnl,'D5')]:
        if sw_pnl > se_pnl:
            print(f'     {name}: YES — switcher beats sell-only by ${sw_pnl - se_pnl:.0f}')
        else:
            print(f'     {name}: NO — sell-only beats switcher by ${se_pnl - sw_pnl:.0f}')
    print()

    print('  Q3. In 2022, did the dwell filter limit damage better than D1?')
    print()
    bu22 = _stats([t for t in bu_trades    if t['year'] == 2022])
    se22 = _stats([t for t in se_trades    if t['year'] == 2022])
    d1_22 = _stats([t for t in sw_d1_trades if t['year'] == 2022])
    d5_22 = _stats([t for t in sw_d5_trades if t['year'] == 2022])
    print(f'     Buy-only 2022 : {_s(bu22["pnl_dollar"])}${abs(bu22["pnl_dollar"]):.0f}'
          f'  ({bu22["taken"]} trades, {bu22["win_pct"]:.1f}% win)')
    print(f'     Sell-only 2022: {_s(se22["pnl_dollar"])}${abs(se22["pnl_dollar"]):.0f}'
          f'  ({se22["taken"]} trades, {se22["win_pct"]:.1f}% win)')
    print(f'     Switcher  D1  : {_s(d1_22["pnl_dollar"])}${abs(d1_22["pnl_dollar"]):.0f}'
          f'  ({d1_22["taken"]} trades, {d1_22["win_pct"]:.1f}% win)')
    print(f'     Switcher  D5  : {_s(d5_22["pnl_dollar"])}${abs(d5_22["pnl_dollar"]):.0f}'
          f'  ({d5_22["taken"]} trades, {d5_22["win_pct"]:.1f}% win)')
    best22 = max(bu22['pnl_dollar'], se22['pnl_dollar'],
                 d1_22['pnl_dollar'], d5_22['pnl_dollar'])
    winner22 = ('Buy-only' if abs(bu22['pnl_dollar'] - best22) < 0.01 else
                'Sell-only' if abs(se22['pnl_dollar'] - best22) < 0.01 else
                'Switcher-D1' if abs(d1_22['pnl_dollar'] - best22) < 0.01 else
                'Switcher-D5')
    print(f'     Best 2022 result: {winner22} (${best22:.0f})')
    if d5_22['pnl_dollar'] > d1_22['pnl_dollar']:
        print(f'     → D5 improved on D1 in 2022 by ${d5_22["pnl_dollar"] - d1_22["pnl_dollar"]:.0f}')
    elif abs(d5_22['pnl_dollar'] - d1_22['pnl_dollar']) < 5:
        print(f'     → D5 and D1 were essentially equal in 2022')
    else:
        print(f'     → D1 actually beat D5 in 2022 by ${d1_22["pnl_dollar"] - d5_22["pnl_dollar"]:.0f}')
    print()

    print('  Q4. How many total switches — D1 vs D5? Which is operationally usable?')
    print()
    n1, n5 = len(sw_d1_switches), len(sw_d5_switches)
    print(f'     D1: {n1} switches ({n1/6:.1f}/year)  — '
          f'{"NOISY" if n1 > 18 else "MODERATE" if n1 > 6 else "STABLE"}')
    print(f'     D5: {n5} switches ({n5/6:.1f}/year)  — '
          f'{"NOISY" if n5 > 18 else "MODERATE" if n5 > 6 else "STABLE"}')
    print()
    if n5 == 0:
        print('     D5 never switched at all. The 5-day dwell requirement completely')
        print('     filters out the noisy regime flips. In practice, the system would')
        print('     have traded purely as a buy-side strategy for all 6 years.')
        print('     This means D5 ≈ buy-only in behavior.')
    elif n5 <= 4:
        print(f'     D5 switched only {n5} times in 6 years — less than once per year.')
        print('     This is operationally clean: each switch is a deliberate, sustained')
        print('     regime change, not a day-by-day flip on VIX oscillation.')
    else:
        print(f'     D5 reduced switches by {n1-n5} ({(1-n5/max(n1,1))*100:.0f}%).')
        print('     Still moderate noise — may need higher dwell (7–10 days) for stability.')
    print()

    print('  Q5. Longest continuous sell-side period under D5 (vs D1)?')
    print()

    def longest_sell_run(regime_log):
        max_run, cur_run, max_start, cur_start, max_end = 0, 0, None, None, None
        for r in regime_log:
            if r['regime'] == 'sell':
                if cur_run == 0:
                    cur_start = r['date']
                cur_run += 1
                if cur_run > max_run:
                    max_run, max_start, max_end = cur_run, cur_start, r['date']
            else:
                cur_run = 0
        return max_run, max_start, max_end

    d1_run, d1_start, d1_end = longest_sell_run(sw_d1_regime)
    d5_run, d5_start, d5_end = longest_sell_run(sw_d5_regime)

    print(f'     D1 longest sell run: {d1_run} consecutive trading days'
          + (f'  ({d1_start} → {d1_end})' if d1_start else ''))
    print(f'     D5 longest sell run: {d5_run} consecutive trading days'
          + (f'  ({d5_start} → {d5_end})' if d5_start else ''))
    print()
    if d5_run == 0:
        print('     D5 never accumulated a sell window. With a 21-bar hold on the')
        print('     sell-side spread, no confirmed sell period was long enough to')
        print('     open AND exit a position within the confirmed window.')
        print('     Any spreads entered at the start of a sell regime will carry')
        print('     forward into the next buy regime — exactly what the transition')
        print('     rule allows, but it means sell-side P&L in D5 is driven almost')
        print('     entirely by positions entered at the START of confirmed sell periods.')
    else:
        print(f'     D5 run of {d5_run} days = {d5_run/21:.1f}× a 21-DTE spread hold.')
        if d5_run >= 21:
            print('     A single sell-side position could open and close within this window.')
        else:
            print('     A 21-bar spread would still be running when the window closed.')
    print()

    # ── SECTION 5: Caveats ────────────────────────────────────────────────────
    print(bar)
    print('  SECTION 5 — HONEST CAVEATS')
    print(bar)
    print()
    print('  1. DWELL DOES NOT FIX THE FUNDAMENTAL PROBLEM')
    print('     The sell-side signal (VIX 25–35 + both IV ranks >40 + both within EMA20 ±3%)')
    print('     was too restrictive to generate sustained windows in this data. Dwell=5')
    print('     reduces the noise but mainly by converting a noisy switcher into something')
    print('     that behaves almost identically to buy-only. If the sell-side window never')
    print('     stays open for 5+ days, the dwell filter simply never activates sell-side.')
    print('     You are not capturing sell-side premium; you are just running buy-side.')
    print()
    print('  2. DURING THE DWELL WINDOW, BUY-SIDE ENTRIES CONTINUE')
    print('     For days 1–4 of a potential regime switch, cur_regime stays "buy", so')
    print('     new buy-side trades can be entered that will then run into the sell window.')
    print('     This creates a partial overlap between strategies that is difficult to')
    print('     manage cleanly in live trading.')
    print()
    print('  3. IN-SAMPLE PARAMETER SELECTION')
    print('     Both the dwell=5 and the VIX 25–35 / IV rank 40 thresholds were designed')
    print('     knowing the 2020–2025 outcome. Out-of-sample the regime filter may be')
    print('     active too frequently or not enough depending on the macro environment.')
    print()
    print('  4. DAILY-ONLY BUY-SIDE vs HOURLY (backtest.py)')
    print('     Buy-only trade counts here are much lower than backtest.py because this')
    print('     script uses only daily bars. Dollar P&L figures are not directly comparable.')
    print()
    print('  5. NO TRANSACTION COSTS')
    print('     BS mid-market pricing. Real options bid/ask (~$0.05–$0.15 on SPY ATM)')
    print('     would cost 8–16% of the $125 risk budget per round-trip.')
    print()
    print('  Do NOT deploy based on this backtest alone.')
    print()


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(sw_d1_trades, sw_d5_trades, bu_trades, se_trades,
                 sw_d1_switches, sw_d5_switches):
    out = {
        'generated':   datetime.now().isoformat(),
        'description': 'Regime-switching backtest: D1 (immediate) vs D5 (5-day dwell)',
        'switcher_d1': {
            'overall':  _stats(sw_d1_trades),
            'switches': len(sw_d1_switches),
            'switch_log': sw_d1_switches,
            'years': {str(yr): _stats([t for t in sw_d1_trades if t['year'] == yr])
                      for yr in YEARS},
        },
        'switcher_d5': {
            'overall':  _stats(sw_d5_trades),
            'switches': len(sw_d5_switches),
            'switch_log': sw_d5_switches,
            'years': {str(yr): _stats([t for t in sw_d5_trades if t['year'] == yr])
                      for yr in YEARS},
        },
        'buy_only':  {
            'overall': _stats(bu_trades),
            'years': {str(yr): _stats([t for t in bu_trades if t['year'] == yr])
                      for yr in YEARS},
        },
        'sell_only': {
            'overall': _stats(se_trades),
            'years': {str(yr): _stats([t for t in se_trades if t['year'] == yr])
                      for yr in YEARS},
        },
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'  Results saved → {os.path.basename(RESULTS_FILE)}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    W = 116
    print('=' * W)
    print('  REGIME-SWITCHING BACKTEST  |  SPY + QQQ  |  2020–2025')
    print(f'  Sell trigger : BOTH IV rank >{SELL_IV_MIN} | BOTH within {SELL_EMA_BAND*100:.0f}% EMA20 | VIX {SELL_VIX_LOW}–{SELL_VIX_HIGH}')
    print(f'  Buy-side     : ATM options, 7 DTE, −45% stop / +50% partial / +100% full / 7-day exit')
    print(f'  Sell-side    : Short {SELL_SHORT_OTM*100:.0f}% OTM put / Long {SELL_LONG_OTM*100:.0f}% OTM put, 21-bar hold, BS pricing')
    print(f'  Sizing       : ${RISK_PER_TRADE:.0f} max risk / trade | ${ACCOUNT_SIZE:,.0f} simulated account')
    print(f'  Variants     : D1 = switch immediately | D5 = require 5 consecutive days first')
    print('=' * W)
    print()

    df = load_data()

    print('Running simulations …', flush=True)
    print('  [1/4] switcher D1 …', end=' ', flush=True)
    sw_d1_trades, sw_d1_regime, sw_d1_switches = run_simulation(df, 'switcher', dwell=1)
    print(f'done  ({len(sw_d1_trades)} trades, {len(sw_d1_switches)} switches)')

    print('  [2/4] switcher D5 …', end=' ', flush=True)
    sw_d5_trades, sw_d5_regime, sw_d5_switches = run_simulation(df, 'switcher', dwell=5)
    print(f'done  ({len(sw_d5_trades)} trades, {len(sw_d5_switches)} switches)')

    print('  [3/4] buy_only    …', end=' ', flush=True)
    bu_trades, _, _ = run_simulation(df, 'buy_only')
    print(f'done  ({len(bu_trades)} trades)')

    print('  [4/4] sell_only   …', end=' ', flush=True)
    se_trades, _, _ = run_simulation(df, 'sell_only')
    print(f'done  ({len(se_trades)} trades)')
    print()

    print_report(sw_d1_trades, sw_d1_regime, sw_d1_switches,
                 sw_d5_trades, sw_d5_regime, sw_d5_switches,
                 bu_trades, se_trades)

    save_results(sw_d1_trades, sw_d5_trades, bu_trades, se_trades,
                 sw_d1_switches, sw_d5_switches)


if __name__ == '__main__':
    main()
