#!/usr/bin/env python3
"""
backtest_0dte.py — 0DTE SPY put credit spread, standalone research track.

Intraday simulation on daily OHLC data:
  Entry    9:45am ET → OPEN price,      T = 6.25h / (6.5h × 252) yr remaining
  Stop     ~noon     → daily LOW price, T = 4.00h / (6.5h × 252) yr remaining
  Close    3:45pm ET → CLOSE price,     T = 0.25h / (6.5h × 252) ≈ intrinsic

Stop check fires before force-close (assumes the worst intraday move precedes
the close on a bad day — conservative ordering of OHLC events).

Pricing: sigma = VIX × 0.85 (same as backtest_credit_spread_v2.py).
IVR:     252-day VIX percentile (threshold 20% vs 30% for the 7DTE track).
No weekly loss limit — isolated research track, not production.

DO NOT COMMIT.
"""

import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_DELTA  = 0.15
DELTA_TOL     = 0.05
SPREAD_WIDTH  = 2.0
MIN_CREDIT    = 0.10
PROFIT_TARGET = 0.50
STOP_LOSS     = 1.50
MIN_IVR       = 20.0
MAX_VIX       = 35.0
SMA_PERIOD    = 20
RISK_FREE     = 0.045
VIX_WINDOW    = 252
MAX_POSITIONS = 1          # 1 trade per day max; 0DTE closes same day so never stacks

# Intraday time fractions (hours remaining / (6.5 trading hours × 252 days))
_YR_HOURS     = 6.5 * 252
T_ENTRY       = 6.25 / _YR_HOURS   # 9:45am ET  —  6.25h remaining
T_NOON        = 4.00 / _YR_HOURS   # ~noon       —  4.00h remaining (stop check)
T_CLOSE       = 0.25 / _YR_HOURS   # 3:45pm ET  —  0.25h remaining (force close)

HOLD_H_STOP   = 6.50 - 4.00        # 9:45am → noon  ≈ 2.25 h elapsed from entry
HOLD_H_EOD    = 6.25 - 0.25        # 9:45am → 3:45pm = 6.00 h elapsed from entry

# ── Macro dates (2020–2026, embedded to avoid cross-file imports) ──────────────
FOMC_DAYS = {
    '2020-01-28','2020-01-29','2020-03-03','2020-03-15',
    '2020-04-28','2020-04-29','2020-06-09','2020-06-10',
    '2020-07-28','2020-07-29','2020-09-15','2020-09-16',
    '2020-11-04','2020-11-05','2020-12-15','2020-12-16',
    '2021-01-26','2021-01-27','2021-03-16','2021-03-17',
    '2021-04-27','2021-04-28','2021-06-15','2021-06-16',
    '2021-07-27','2021-07-28','2021-09-21','2021-09-22',
    '2021-11-02','2021-11-03','2021-12-14','2021-12-15',
    '2022-01-25','2022-01-26','2022-03-15','2022-03-16',
    '2022-05-03','2022-05-04','2022-06-14','2022-06-15',
    '2022-07-26','2022-07-27','2022-09-20','2022-09-21',
    '2022-11-01','2022-11-02','2022-12-13','2022-12-14',
    '2023-01-31','2023-02-01','2023-03-21','2023-03-22',
    '2023-05-02','2023-05-03','2023-06-13','2023-06-14',
    '2023-07-25','2023-07-26','2023-09-19','2023-09-20',
    '2023-10-31','2023-11-01','2023-12-12','2023-12-13',
    '2024-01-30','2024-01-31','2024-03-19','2024-03-20',
    '2024-04-30','2024-05-01','2024-06-11','2024-06-12',
    '2024-07-30','2024-07-31','2024-09-17','2024-09-18',
    '2024-11-06','2024-11-07','2024-12-17','2024-12-18',
    '2025-01-28','2025-01-29','2025-03-18','2025-03-19',
    '2025-05-06','2025-05-07','2025-06-17','2025-06-18',
    '2025-07-29','2025-07-30','2025-09-16','2025-09-17',
    '2025-10-28','2025-10-29','2025-12-09','2025-12-10',
    '2026-01-28','2026-01-29','2026-03-18','2026-03-19',
    '2026-04-29','2026-04-30','2026-06-10','2026-06-11',
    '2026-07-29','2026-07-30','2026-09-16','2026-09-17',
    '2026-11-04','2026-11-05','2026-12-09','2026-12-10',
}

CPI_DAYS = {
    '2020-01-14','2020-02-13','2020-03-11','2020-04-10',
    '2020-05-12','2020-06-10','2020-07-14','2020-08-12',
    '2020-09-11','2020-10-13','2020-11-12','2020-12-10',
    '2021-01-13','2021-02-10','2021-03-10','2021-04-13',
    '2021-05-12','2021-06-10','2021-07-13','2021-08-11',
    '2021-09-14','2021-10-13','2021-11-10','2021-12-10',
    '2022-01-12','2022-02-10','2022-03-10','2022-04-12',
    '2022-05-11','2022-06-11','2022-07-13','2022-08-10',
    '2022-09-13','2022-10-13','2022-11-10','2022-12-13',
    '2023-01-12','2023-02-14','2023-03-14','2023-04-12',
    '2023-05-10','2023-06-13','2023-07-12','2023-08-10',
    '2023-09-13','2023-10-12','2023-11-14','2023-12-12',
    '2024-01-11','2024-02-13','2024-03-12','2024-04-10',
    '2024-05-15','2024-06-12','2024-07-11','2024-08-14',
    '2024-09-11','2024-10-10','2024-11-13','2024-12-11',
    '2025-01-15','2025-02-12','2025-03-12','2025-04-10',
    '2025-05-13','2025-06-11','2025-07-15','2025-08-13',
    '2025-09-10','2025-10-14','2025-11-12','2025-12-10',
    '2026-01-14','2026-02-11','2026-03-11','2026-04-10',
    '2026-05-13','2026-06-11','2026-07-14','2026-08-12',
    '2026-09-09','2026-10-14','2026-11-12','2026-12-10',
}

GDP_DAYS = {
    '2020-01-30','2020-04-29','2020-07-30','2020-10-29',
    '2021-01-28','2021-04-29','2021-07-29','2021-10-28',
    '2022-01-27','2022-04-28','2022-07-28','2022-10-27',
    '2023-01-26','2023-04-27','2023-07-27','2023-10-26',
    '2024-01-25','2024-04-25','2024-07-25','2024-10-30',
    '2025-01-29','2025-04-30','2025-07-30','2025-10-29',
    '2026-01-29','2026-04-29','2026-07-30','2026-10-29',
}


def is_macro_day(d: date):
    s = d.isoformat()
    if s in FOMC_DAYS: return 'FOMC'
    if s in CPI_DAYS:  return 'CPI'
    if s in GDP_DAYS:  return 'GDP'
    first     = d.replace(day=1)
    first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
    if d == first_fri: return 'NFP'
    return None


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_put_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float(max(K - S, 0))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def find_short_put_strike(S, T, sigma):
    """Strike where put |delta| ≈ TARGET_DELTA. Returns (K, delta) or (None, None)."""
    if sigma <= 0 or T <= 0 or S <= 0:
        return None, None
    d1_target = norm.ppf(1.0 - TARGET_DELTA)       # N^{-1}(0.85) ≈ +1.036
    log_SK    = d1_target * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma**2) * T
    K         = round(S * math.exp(-log_SK))
    if K <= 0 or K >= S:
        return None, None
    d1    = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta = norm.cdf(d1) - 1.0                      # put delta (negative)
    if abs(abs(delta) - TARGET_DELTA) > DELTA_TOL:
        return None, None
    return float(K), float(delta)


def put_spread_credit(S, short_K, long_K, T, sigma):
    """Current value of the short put spread (cost to close = credit received at entry)."""
    return round(bs_put_price(S, short_K, T, sigma) - bs_put_price(S, long_K, T, sigma), 4)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Download SPY OHLC + VIX. Returns daily rows from 2020-06-30 onward."""
    print('Loading SPY (OHLC), ^VIX from yfinance (2019-07-01 → 2026-07-05)...')
    raw = yf.download(
        ['SPY', '^VIX'],
        start='2019-07-01', end='2026-07-05',
        auto_adjust=True, progress=True,
    )

    spy_close = raw['Close']['SPY'].ffill()
    spy_open  = raw['Open']['SPY'].ffill()
    spy_high  = raw['High']['SPY'].ffill()
    spy_low   = raw['Low']['SPY'].ffill()
    vix       = raw['Close']['^VIX'].ffill()

    sma_spy = spy_close.rolling(SMA_PERIOD).mean()

    # IVR: 252-day VIX percentile (identical to backtest_credit_spread_v2.py)
    vix_arr = vix.values.astype(float)
    n       = len(vix_arr)
    ivr_arr = np.full(n, np.nan)
    for i in range(VIX_WINDOW, n):
        w          = vix_arr[i - VIX_WINDOW:i]
        ivr_arr[i] = float((w < vix_arr[i]).sum()) / VIX_WINDOW * 100.0
    ivr = pd.Series(ivr_arr, index=vix.index)

    df = pd.DataFrame({
        'spy_close': spy_close,
        'spy_open':  spy_open,
        'spy_high':  spy_high,
        'spy_low':   spy_low,
        'vix':       vix,
        'sma_spy':   sma_spy,
        'ivr':       ivr,
    })
    df = df[(df.index >= '2020-01-01') & (df.index < '2026-07-05')].dropna()
    print(f'  {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})')
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_0dte(df: pd.DataFrame) -> dict:
    """
    0DTE SPY put credit spread simulation.

    One entry attempt per day in the 9:45–11:00am window (approximated via OPEN
    price). Position always closes same day — either on a stop, at profit, or
    force-closed at 3:45pm. Max one concurrent position (trivially satisfied
    since all positions expire same day).
    """
    trades    = []
    daily_pnl = []
    equity    = 0.0
    peak      = 0.0
    max_dd    = 0.0

    blocked = {k: 0 for k in ('macro', 'sma', 'ivr', 'vix', 'min_credit', 'delta_miss')}

    rows = df.to_dict('records')
    idx  = [d.date() for d in df.index]

    for i, d in enumerate(idx):
        row = rows[i]
        pnl = 0.0

        if d.weekday() < 5:
            S_open  = float(row['spy_open'])
            S_low   = float(row['spy_low'])
            S_close = float(row['spy_close'])
            sigma   = max(float(row['vix']) / 100.0 * 0.85, 0.05)
            sma_val = float(row['sma_spy'])
            vix_val = float(row['vix'])
            ivr_val = float(row['ivr'])

            can_enter = True
            macro     = is_macro_day(d)
            if macro:
                blocked['macro'] += 1;   can_enter = False
            elif S_open <= sma_val:
                blocked['sma']   += 1;   can_enter = False
            elif ivr_val < MIN_IVR:
                blocked['ivr']   += 1;   can_enter = False
            elif vix_val >= MAX_VIX:
                blocked['vix']   += 1;   can_enter = False

            if can_enter:
                short_K, _ = find_short_put_strike(S_open, T_ENTRY, sigma)
                if short_K is None:
                    blocked['delta_miss'] += 1
                else:
                    long_K = short_K - SPREAD_WIDTH
                    credit = put_spread_credit(S_open, short_K, long_K, T_ENTRY, sigma)

                    if credit < MIN_CREDIT:
                        blocked['min_credit'] += 1
                    else:
                        stop_threshold   = round(credit * STOP_LOSS,     4)
                        profit_threshold = round(credit * PROFIT_TARGET,  4)

                        # Stop check: cost of spread at daily LOW around noon.
                        # Using T_NOON (4h remaining) because the LOW could occur
                        # mid-session; T_NOON is more realistic than using T≈0.
                        cost_at_low   = put_spread_credit(S_low,   short_K, long_K, T_NOON,  sigma)
                        # Force-close cost: spread at CLOSE, T≈0 (essentially intrinsic)
                        cost_at_close = put_spread_credit(S_close, short_K, long_K, T_CLOSE, sigma)

                        if cost_at_low >= stop_threshold:
                            reason     = 'STOP'
                            close_cost = stop_threshold     # exit exactly at stop level
                            held_hours = HOLD_H_STOP
                        elif cost_at_close <= profit_threshold:
                            reason     = 'PROFIT'
                            close_cost = cost_at_close
                            held_hours = HOLD_H_EOD
                        else:
                            reason     = 'FORCE_CLOSE'
                            close_cost = cost_at_close
                            held_hours = HOLD_H_EOD

                        pnl = round((credit - close_cost) * 100, 2)
                        trades.append(dict(
                            date=d, year=d.year,
                            short_K=short_K, long_K=long_K,
                            credit=credit, close_cost=round(close_cost, 4),
                            pnl=pnl, reason=reason,
                            held_hours=held_hours,
                            spy_open=S_open, spy_low=S_low, spy_close=S_close,
                            vix=vix_val, ivr=round(ivr_val, 1),
                        ))

        equity  += pnl
        peak     = max(peak, equity)
        max_dd   = max(max_dd, peak - equity)
        daily_pnl.append(pnl)

    return dict(
        trades=trades, blocked=blocked,
        daily_pnl=daily_pnl, max_dd=max_dd, dates=idx,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_stats(res: dict, label: str):
    trades    = res['trades']
    blocked   = res['blocked']
    daily_pnl = res['daily_pnl']
    max_dd    = res['max_dd']

    W = 64
    print(f'\n{"="*W}')
    print(f'  {label}')
    print(f'{"="*W}')

    if not trades:
        print('  No trades executed.')
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['pnl'] > 0]
    loss = df_t[df_t['pnl'] <= 0]

    total_pnl  = df_t['pnl'].sum()
    win_rate   = len(wins) / len(df_t) * 100
    avg_win    = wins['pnl'].mean() if len(wins) else 0.0
    avg_loss   = loss['pnl'].mean() if len(loss) else 0.0
    dpnl       = pd.Series(daily_pnl, dtype=float)
    sharpe     = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    stop_mask  = df_t['reason'] == 'STOP'
    stop_pct   = stop_mask.sum() / len(df_t) * 100
    avg_credit = df_t['credit'].mean()
    avg_held   = df_t['held_hours'].mean()

    # How often the 15-delta put is NOT above SPY at close (spread has some cost)
    ate_premium = (df_t['close_cost'] == 0.0).sum() / len(df_t) * 100

    total_days    = len(daily_pnl)
    days_in_trade = len(df_t)
    trade_freq    = days_in_trade / total_days * 100

    yearly = df_t.groupby('year').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
        stop_pct=('reason', lambda x: (x == 'STOP').mean() * 100),
        avg_credit=('credit', 'mean'),
    )

    print(f'  Total trades:     {len(df_t):<6} ({days_in_trade} of {total_days} trading days, {trade_freq:.0f}%)')
    print(f'  Win rate:         {win_rate:.1f}%')
    print(f'  Stop rate:        {stop_pct:.1f}%')
    print(f'  Expired worthless:{ate_premium:.1f}%')
    print(f'  Avg hold:         {avg_held:.1f}h  (stop≈{HOLD_H_STOP:.1f}h  profit/close≈{HOLD_H_EOD:.1f}h)')
    print(f'  Avg credit:       ${avg_credit:.4f}  (${avg_credit*100:.2f}/contract)')
    print(f'  Avg win:          ${avg_win:.2f}')
    print(f'  Avg loss:         ${avg_loss:.2f}')
    print(f'  Total P&L:        ${total_pnl:,.2f}')
    print(f'  Sharpe ratio:     {sharpe:.2f}')
    print(f'  Max drawdown:     ${max_dd:,.2f}')

    print(f'\n  {"Year":<6} {"Trades":>7} {"WR%":>6} {"Stop%":>6} '
          f'{"AvgCr":>7} {"P&L":>11}')
    print(f'  {"-"*47}')
    for yr, row in yearly.iterrows():
        print(f'  {yr:<6} {int(row["trades"]):>7} {row["wr"]:>5.1f}% '
              f'{row["stop_pct"]:>5.1f}% '
              f'${row["avg_credit"]:>5.3f}  ${row["pnl"]:>9,.2f}')

    print(f'\n  Exit reasons:')
    for reason, cnt in df_t['reason'].value_counts().items():
        avg = df_t[df_t['reason'] == reason]['pnl'].mean()
        print(f'    {reason:<14} {cnt:>4} ({cnt/len(df_t)*100:4.1f}%)  avg ${avg:+.2f}')

    total_blocked = sum(blocked.values())
    if total_blocked:
        print(f'\n  Entry filter blocks ({total_blocked:,} total):')
        for k, v in blocked.items():
            if v:
                print(f'    {k:<14} {v:>6,}  ({v/total_blocked*100:4.1f}%)')


def print_period_sharpe(res: dict, label: str, n_periods: int = 4):
    """Split trading history into n equal periods; report per-period metrics."""
    dates     = res['dates']
    daily_pnl = res['daily_pnl']
    n_days    = len(dates)
    chunk     = n_days // n_periods

    if not res['trades']:
        print(f'\n  {label}: No trades — skipping period breakdown.')
        return

    df_t = pd.DataFrame(res['trades'])

    print(f'\n  {label} — Sharpe / P&L by sub-period:')
    print(f'  {"Period":<26}  {"Days":>4}  {"Trades":>6}  '
          f'{"Stop%":>6}  {"P&L":>10}  {"Sharpe":>7}')
    print(f'  {"-"*70}')

    flag_lines = []

    for p in range(n_periods):
        start   = p * chunk
        end     = (p + 1) * chunk if p < n_periods - 1 else n_days
        d_start = dates[start]
        d_end   = dates[end - 1]

        sl       = pd.Series(daily_pnl[start:end], dtype=float)
        pnl_p    = sl.sum()
        sharpe_p = (sl.mean() / sl.std() * math.sqrt(252)) if sl.std() > 0 else 0.0

        mask  = (df_t['date'] >= d_start) & (df_t['date'] <= d_end)
        sub   = df_t[mask]
        n_t   = len(sub)
        stop_p = (sub['reason'] == 'STOP').sum() / n_t * 100 if n_t > 0 else 0.0

        period_str = f'P{p+1} {d_start} → {d_end}'
        print(f'  {period_str:<26}  {end-start:>4}  {n_t:>6}  '
              f'{stop_p:>5.1f}%  ${pnl_p:>9,.2f}  {sharpe_p:>7.2f}')

        # Concentration flags
        total_pnl_all  = pd.Series(daily_pnl, dtype=float).sum()
        total_trades   = len(df_t)
        pnl_share      = abs(pnl_p / total_pnl_all) if total_pnl_all != 0 else 0
        trade_share    = n_t / total_trades if total_trades > 0 else 0
        if pnl_share > 0.60 and total_pnl_all > 0:
            flag_lines.append(f'  ⚠  P{p+1}: {pnl_share*100:.0f}% of total P&L concentrated here')
        if trade_share > 0.60:
            flag_lines.append(f'  ⚠  P{p+1}: {trade_share*100:.0f}% of all trades concentrated here')

    # Sharpe dispersion check
    sharpes = []
    for p in range(n_periods):
        start = p * chunk
        end   = (p + 1) * chunk if p < n_periods - 1 else n_days
        sl    = pd.Series(daily_pnl[start:end], dtype=float)
        sharpes.append((sl.mean() / sl.std() * math.sqrt(252)) if sl.std() > 0 else 0.0)
    dispersion = max(sharpes) - min(sharpes)
    if dispersion > 1.5:
        flag_lines.append(f'  ⚠  Sharpe dispersion {dispersion:.2f} across periods (>1.5 threshold)')

    if flag_lines:
        print()
        for fl in flag_lines:
            print(fl)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df  = load_data()
    res = run_0dte(df)
    print_stats(res,        '0DTE SPY PUT CREDIT SPREAD (15Δ, $2-wide, 50%PT / 150%SL)')
    print_period_sharpe(res, '0DTE SPY')


if __name__ == '__main__':
    main()
