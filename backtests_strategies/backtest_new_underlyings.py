#!/usr/bin/env python3
"""
backtest_new_underlyings.py
Tests TLT, XLU, SLV, GDX as put credit spread candidates alongside a SPY baseline.

Methodology matches backtest_credit_spread_v2.py with one deliberate difference:
sigma is Parkinson realized vol (21-day rolling, annualized) rather than VIX*0.85,
because VIX is not a valid implied vol proxy for these underlyings.
IVR is the 252-day percentile rank of each ticker's own Parkinson vol.
VIX is still used as the MAX_VIX market-stress gate (same as v2).

DO NOT COMMIT.
"""

import math
import os
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_credit_spread_v2 import FOMC_DAYS, CPI_DAYS, GDP_DAYS, is_macro_day  # noqa: F401

warnings.filterwarnings('ignore')

# ── GLOBAL CONSTANTS ──────────────────────────────────────────────────────────
TARGET_DELTA  = 0.20
DELTA_TOL     = 0.05
STOP_LOSS     = 2.00
PROFIT_TARGET = 0.50
SMA_PERIOD    = 20
RISK_FREE     = 0.045
VIX_WINDOW    = 252
PARK_WINDOW   = 21
MAX_POSITIONS = 2

# ── PER-TICKER PARAMETERS ─────────────────────────────────────────────────────
TICKER_PARAMS = {
    'SPY': dict(spread_width=5.0, min_credit=0.25, min_ivr=30.0, max_vix=35.0),
    'TLT': dict(spread_width=5.0, min_credit=0.15, min_ivr=20.0, max_vix=35.0),
    'XLU': dict(spread_width=2.0, min_credit=0.15, min_ivr=20.0, max_vix=35.0),
    'SLV': dict(spread_width=2.0, min_credit=0.15, min_ivr=25.0, max_vix=35.0),
    'GDX': dict(spread_width=5.0, min_credit=0.25, min_ivr=30.0, max_vix=35.0),
}

ALL_TICKERS = ['SPY', 'TLT', 'XLU', 'SLV', 'GDX']
NEW_TICKERS = ['TLT', 'XLU', 'SLV', 'GDX']


# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────

def bs_put_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float(max(K - S, 0))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def find_short_strike(S, T, sigma):
    if sigma <= 0 or T <= 0 or S <= 0:
        return None, None
    d1_target = norm.ppf(1.0 - TARGET_DELTA)
    log_SK    = d1_target * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma**2) * T
    K_raw     = S * math.exp(-log_SK)
    K         = round(K_raw)
    if K <= 0:
        return None, None
    d1    = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta = norm.cdf(d1) - 1.0
    if abs(abs(delta) - TARGET_DELTA) > DELTA_TOL:
        return None, None
    return float(K), float(delta)


def spread_credit(S, short_K, long_K, T, sigma):
    return round(bs_put_price(S, short_K, T, sigma) - bs_put_price(S, long_K, T, sigma), 4)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = PARK_WINDOW) -> pd.Series:
    """Annualized Parkinson vol over a rolling window."""
    hl_sq = (np.log(high / low) ** 2) / (4.0 * math.log(2))
    return (hl_sq.rolling(window).mean() * 252) ** 0.5


def _vol_rank(sigma: pd.Series, window: int = VIX_WINDOW) -> pd.Series:
    """252-day percentile rank of Parkinson vol — IVR equivalent for non-SPY tickers."""
    arr      = sigma.values.astype(float)
    n        = len(arr)
    rank_arr = np.full(n, np.nan)
    for i in range(window, n):
        w            = arr[i - window:i]
        rank_arr[i]  = float((w < arr[i]).sum()) / window * 100.0
    return pd.Series(rank_arr, index=sigma.index)


def load_data() -> pd.DataFrame:
    download_tickers = ALL_TICKERS + ['^VIX']
    print(f'Loading {", ".join(ALL_TICKERS)}, ^VIX from yfinance (2019-07-01 → 2026-07-05)...')
    raw = yf.download(
        download_tickers,
        start='2019-07-01', end='2026-07-05',
        auto_adjust=True, progress=True,
    )

    close_all = raw['Close'].copy()
    high_all  = raw['High'].copy()
    low_all   = raw['Low'].copy()
    close_all.columns = [str(c) for c in close_all.columns]
    high_all.columns  = [str(c) for c in high_all.columns]
    low_all.columns   = [str(c) for c in low_all.columns]

    cols = {'vix': close_all['^VIX'].ffill()}

    for t in ALL_TICKERS:
        c       = close_all[t].ffill()
        h       = high_all[t].ffill()
        lo      = low_all[t].ffill()
        park    = _parkinson_vol(h, lo).clip(lower=0.05)
        ivr     = _vol_rank(park)

        cols[f'close_{t}'] = c
        cols[f'sma_{t}']   = c.rolling(SMA_PERIOD).mean()
        cols[f'sigma_{t}'] = park
        cols[f'ivr_{t}']   = ivr

    df = pd.DataFrame(cols)
    df = df[(df.index >= '2020-01-01') & (df.index < '2026-07-05')].dropna()
    print(f'  {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})')
    return df


# ── POSITION ──────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ('ticker', 'entry_date', 'expiry', 'short_K', 'long_K',
                 'credit', 'profit_tgt', 'stop_cost')

    def __init__(self, ticker, entry_date, expiry, short_K, long_K, credit):
        self.ticker      = ticker
        self.entry_date  = entry_date
        self.expiry      = expiry
        self.short_K     = short_K
        self.long_K      = long_K
        self.credit      = credit
        self.profit_tgt  = round(credit * PROFIT_TARGET, 4)
        self.stop_cost   = round(credit * STOP_LOSS, 4)

    def cost_to_close(self, S, T, sigma):
        return spread_credit(S, self.short_K, self.long_K, T, sigma)


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def next_expiry(d: date) -> date:
    best, best_diff = None, 999
    for delta in range(4, 13):
        c    = d + timedelta(days=delta)
        if c.weekday() == 4:
            diff = abs(delta - 7)
            if diff < best_diff:
                best_diff, best = diff, c
    if best is None:
        best = d + timedelta(days=7)
        while best.weekday() >= 5:
            best += timedelta(days=1)
    return best


def run_backtest(df: pd.DataFrame, use_tickers: list) -> dict:
    trades    = []
    open_pos  = []
    daily_pnl = []
    equity    = 0.0
    peak      = 0.0
    max_dd    = 0.0

    blocked = {k: 0 for k in (
        'macro', 'sma', 'ivr', 'vix', 'min_credit',
        'position_limit', 'underwater', 'delta_miss',
    )}

    rows = df.to_dict('records')
    idx  = [d.date() for d in df.index]

    for i, d in enumerate(idx):
        row = rows[i]

        # ── 1. Monitor + close positions ────────────────────────────────────
        day_pnl = 0.0
        keep    = []
        for pos in open_pos:
            S     = float(row[f'close_{pos.ticker}'])
            sigma = float(row[f'sigma_{pos.ticker}'])
            T     = max((pos.expiry - d).days / 365.0, 0.0)
            cost  = pos.cost_to_close(S, T, sigma)

            if d >= pos.expiry:
                cost   = max(pos.short_K - S, 0) - max(pos.long_K - S, 0)
                reason = 'EXPIRY'
            elif cost <= pos.profit_tgt:
                reason = 'PROFIT'
            elif cost >= pos.stop_cost:
                reason = 'STOP'
            else:
                keep.append(pos)
                continue

            pnl      = round((pos.credit - cost) * 100, 2)
            day_pnl += pnl
            trades.append(dict(
                ticker=pos.ticker, entry=pos.entry_date, exit=d,
                expiry=pos.expiry, short_K=pos.short_K, long_K=pos.long_K,
                credit=pos.credit, close_cost=round(cost, 4), pnl=pnl,
                reason=reason, held=(d - pos.entry_date).days,
                year=d.year,
            ))

        open_pos = keep

        # ── 2. Entry evaluation ─────────────────────────────────────────────
        if d.weekday() < 5:
            for ticker in use_tickers:
                if len(open_pos) >= MAX_POSITIONS:
                    blocked['position_limit'] += 1
                    continue

                params  = TICKER_PARAMS[ticker]
                S       = float(row[f'close_{ticker}'])
                sma_val = float(row[f'sma_{ticker}'])
                sigma   = float(row[f'sigma_{ticker}'])
                ivr_val = float(row[f'ivr_{ticker}'])
                vix_val = float(row['vix'])

                if is_macro_day(d):
                    blocked['macro'] += 1
                    continue

                if S <= sma_val:
                    blocked['sma'] += 1
                    continue

                if ivr_val < params['min_ivr']:
                    blocked['ivr'] += 1
                    continue

                if vix_val >= params['max_vix']:
                    blocked['vix'] += 1
                    continue

                if len(open_pos) == 1:
                    p    = open_pos[0]
                    pS   = float(row[f'close_{p.ticker}'])
                    pSig = float(row[f'sigma_{p.ticker}'])
                    pT   = max((p.expiry - d).days / 365.0, 0.0)
                    if p.cost_to_close(pS, pT, pSig) > p.credit:
                        blocked['underwater'] += 1
                        continue

                expiry = next_expiry(d)

                if any(p.ticker == ticker and p.expiry == expiry for p in open_pos):
                    continue

                T       = max((expiry - d).days / 365.0, 1 / 365)
                short_K, _ = find_short_strike(S, T, sigma)
                if short_K is None:
                    blocked['delta_miss'] += 1
                    continue

                long_K = short_K - params['spread_width']
                if long_K <= 0:
                    continue

                credit = spread_credit(S, short_K, long_K, T, sigma)
                if credit < params['min_credit']:
                    blocked['min_credit'] += 1
                    continue

                open_pos.append(Position(ticker, d, expiry, short_K, long_K, credit))

        # ── 3. Daily equity tracking ─────────────────────────────────────────
        unrealized = 0.0
        for pos in open_pos:
            S     = float(row[f'close_{pos.ticker}'])
            sigma = float(row[f'sigma_{pos.ticker}'])
            T     = max((pos.expiry - d).days / 365.0, 0.0)
            unrealized += (pos.credit - pos.cost_to_close(S, T, sigma)) * 100

        equity  += day_pnl
        total_eq = equity + unrealized
        peak     = max(peak, total_eq)
        max_dd   = max(max_dd, peak - total_eq)
        daily_pnl.append(day_pnl)

    # Force-close anything still open at end of data
    last_row = rows[-1]
    last_d   = idx[-1]
    for pos in open_pos:
        S     = float(last_row[f'close_{pos.ticker}'])
        sigma = float(last_row[f'sigma_{pos.ticker}'])
        T     = max((pos.expiry - last_d).days / 365.0, 0.0)
        cost  = pos.cost_to_close(S, T, sigma)
        pnl   = round((pos.credit - cost) * 100, 2)
        trades.append(dict(
            ticker=pos.ticker, entry=pos.entry_date, exit=last_d,
            expiry=pos.expiry, short_K=pos.short_K, long_K=pos.long_K,
            credit=pos.credit, close_cost=round(cost, 4), pnl=pnl,
            reason='OPEN_AT_END', held=(last_d - pos.entry_date).days,
            year=last_d.year,
        ))

    return dict(trades=trades, blocked=blocked, daily_pnl=daily_pnl,
                max_dd=max_dd, dates=idx)


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _metrics(res: dict) -> dict:
    df_t = pd.DataFrame(res['trades'])
    if df_t.empty:
        return dict(n=0, wr=0.0, pnl=0.0, dd=res['max_dd'], sharpe=0.0, stop_pct=0.0)
    wins     = (df_t['pnl'] > 0).sum()
    wr       = wins / len(df_t) * 100
    total    = df_t['pnl'].sum()
    dpnl     = pd.Series(res['daily_pnl'], dtype=float)
    sharpe   = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    stop_pct = (df_t['reason'] == 'STOP').sum() / len(df_t) * 100
    return dict(n=len(df_t), wr=wr, pnl=total, dd=res['max_dd'],
                sharpe=sharpe, stop_pct=stop_pct)


def print_stats(res: dict, label: str):
    trades    = res['trades']
    blocked   = res['blocked']
    daily_pnl = res['daily_pnl']
    max_dd    = res['max_dd']

    if not trades:
        print(f'\n{label}: No trades executed.')
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['pnl'] > 0]
    loss = df_t[df_t['pnl'] <= 0]

    total_pnl  = df_t['pnl'].sum()
    win_rate   = len(wins) / len(df_t) * 100
    avg_win    = wins['pnl'].mean() if len(wins) else 0.0
    avg_loss   = loss['pnl'].mean() if len(loss) else 0.0
    avg_credit = df_t['credit'].mean()
    avg_held   = df_t['held'].mean()

    dpnl   = pd.Series(daily_pnl, dtype=float)
    sharpe = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0

    yearly = (df_t.groupby('year')['pnl']
              .agg(pnl='sum', trades='count')
              .rename(columns={'pnl': 'P&L', 'trades': 'N'}))

    W = 62
    print(f'\n{"="*W}')
    print(f'  {label}')
    print(f'{"="*W}')
    print(f'  Total trades:    {len(df_t):<6}  (wins {len(wins)}  /  losses {len(loss)})')
    print(f'  Win rate:        {win_rate:.1f}%')
    print(f'  Total P&L:       ${total_pnl:>10,.2f}')
    print(f'  Avg win:         ${avg_win:>10.2f}')
    print(f'  Avg loss:        ${avg_loss:>10.2f}')
    print(f'  Avg credit:      ${avg_credit:>10.4f}')
    print(f'  Avg hold (days): {avg_held:>10.1f}')
    print(f'  Sharpe ratio:    {sharpe:>10.2f}')
    print(f'  Max drawdown:    ${max_dd:>10,.2f}')

    print(f'\n  {"Year":<6} {"P&L":>10} {"Trades":>8}')
    print(f'  {"-"*24}')
    for yr, row in yearly.iterrows():
        print(f'  {yr:<6} ${row["P&L"]:>9,.2f} {int(row["N"]):>8}')

    best_yr  = yearly['P&L'].idxmax()
    worst_yr = yearly['P&L'].idxmin()
    print(f'\n  Best year:  {best_yr}  (${yearly.loc[best_yr,  "P&L"]:,.2f})')
    print(f'  Worst year: {worst_yr}  (${yearly.loc[worst_yr, "P&L"]:,.2f})')

    print(f'\n  Exit reasons:')
    for reason, cnt in df_t['reason'].value_counts().items():
        avg = df_t[df_t['reason'] == reason]['pnl'].mean()
        print(f'    {reason:<16} {cnt:>4} ({cnt/len(df_t)*100:4.1f}%)  avg ${avg:+.2f}')

    total_blocked = sum(blocked.values())
    if total_blocked:
        print(f'\n  Entry filter blocks (total {total_blocked:,}):')
        for k, v in blocked.items():
            if v:
                print(f'    {k:<16} {v:>6,}  ({v/total_blocked*100:4.1f}%)')


def print_period_sharpe(res: dict, label: str, n_periods: int = 4):
    """Split the backtest date range into n equal periods and report Sharpe per slice."""
    dates     = res['dates']
    daily_pnl = res['daily_pnl']
    n_days    = len(dates)

    if n_days < n_periods:
        return

    chunk = n_days // n_periods

    print(f'\n  {label} — Sharpe by sub-period:')
    print(f'  {"Period":<23}  {"Days":>5}  {"P&L":>10}  {"Sharpe":>7}')
    print(f'  {"-"*50}')

    for p in range(n_periods):
        start   = p * chunk
        end     = (p + 1) * chunk if p < n_periods - 1 else n_days
        d_start = dates[start]
        d_end   = dates[end - 1]
        sl      = pd.Series(daily_pnl[start:end], dtype=float)
        pnl_p   = sl.sum()
        sharpe_p = (sl.mean() / sl.std() * math.sqrt(252)) if sl.std() > 0 else 0.0
        period_str = f'{d_start} to {d_end}'
        print(f'  {period_str:<23}  {end-start:>5}  ${pnl_p:>9,.2f}  {sharpe_p:>7.2f}')


def print_combined_comparison(res_spy: dict, res_solo: dict, res_comb: dict, ticker: str):
    """Three-column table: SPY-only | ticker-only | SPY+ticker combined."""
    ms  = _metrics(res_spy)
    mn  = _metrics(res_solo)
    mc  = _metrics(res_comb)

    col_spy  = 'SPY-only'
    col_solo = f'{ticker}-only'
    col_comb = f'SPY+{ticker}'

    W = 74
    print(f'\n{"="*W}')
    print(f'  COMBINED: {col_spy}  vs  {col_solo}  vs  {col_comb}')
    print(f'{"="*W}')
    print(f'  {"Metric":<22} {col_spy:>14} {col_solo:>14} {col_comb:>14}')
    print(f'  {"-"*64}')
    print(f'  {"Total trades":<22} {ms["n"]:>14} {mn["n"]:>14} {mc["n"]:>14}')
    print(f'  {"Win rate":<22} {ms["wr"]:>13.1f}% {mn["wr"]:>13.1f}% {mc["wr"]:>13.1f}%')
    print(f'  {"Stop-loss rate":<22} {ms["stop_pct"]:>13.1f}% {mn["stop_pct"]:>13.1f}% {mc["stop_pct"]:>13.1f}%')
    print(f'  {"Total P&L":<22} ${ms["pnl"]:>13,.2f} ${mn["pnl"]:>13,.2f} ${mc["pnl"]:>13,.2f}')
    print(f'  {"Max drawdown":<22} ${ms["dd"]:>13,.2f} ${mn["dd"]:>13,.2f} ${mc["dd"]:>13,.2f}')
    print(f'  {"Sharpe ratio":<22} {ms["sharpe"]:>14.2f} {mn["sharpe"]:>14.2f} {mc["sharpe"]:>14.2f}')
    print(f'\n  vs SPY-only:')
    print(f'  {"P&L delta":<22} {"—":>14} ${mn["pnl"]-ms["pnl"]:>+13,.2f} ${mc["pnl"]-ms["pnl"]:>+13,.2f}')
    print(f'  {"Drawdown delta":<22} {"—":>14} ${mn["dd"]-ms["dd"]:>+13,.2f} ${mc["dd"]-ms["dd"]:>+13,.2f}')
    print(f'  {"Sharpe delta":<22} {"—":>14} {mn["sharpe"]-ms["sharpe"]:>+14.2f} {mc["sharpe"]-ms["sharpe"]:>+14.2f}')

    # Year-by-year delta vs SPY
    t_spy  = pd.DataFrame(res_spy['trades'])
    t_solo = pd.DataFrame(res_solo['trades'])
    t_comb = pd.DataFrame(res_comb['trades'])
    spy_yr  = t_spy.groupby('year')['pnl'].sum()  if not t_spy.empty  else pd.Series(dtype=float)
    solo_yr = t_solo.groupby('year')['pnl'].sum() if not t_solo.empty else pd.Series(dtype=float)
    comb_yr = t_comb.groupby('year')['pnl'].sum() if not t_comb.empty else pd.Series(dtype=float)
    all_yrs = sorted(set(spy_yr.index) | set(solo_yr.index) | set(comb_yr.index))

    print(f'\n  {"Year":<6} {col_spy:>12} {col_solo:>12} {col_comb:>12}')
    print(f'  {"-"*44}')
    for yr in all_yrs:
        s  = spy_yr.get(yr, 0.0)
        n_ = solo_yr.get(yr, 0.0)
        c  = comb_yr.get(yr, 0.0)
        print(f'  {yr:<6} ${s:>11,.2f} ${n_:>11,.2f} ${c:>11,.2f}')


def print_summary_table(results: list, res_spy: dict):
    """
    Rank all 4 new tickers by standalone Sharpe.
    results: list of (ticker, res_standalone)
    """
    ranked  = sorted(results, key=lambda x: _metrics(x[1])['sharpe'], reverse=True)
    spy_m   = _metrics(res_spy)
    SPY_STOP_BASELINE = 24.1

    W = 74
    print(f'\n{"="*W}')
    print('  STANDALONE RANKING — new tickers by Sharpe ratio')
    print(f'{"="*W}')
    print(f'  {"Rank":<5} {"Ticker":<7} {"Trades":>7} {"Win%":>6} '
          f'{"Stop%":>6} {"P&L":>11} {"MaxDD":>11} {"Sharpe":>8}')
    print(f'  {"-"*65}')

    for rank, (ticker, res) in enumerate(ranked, 1):
        m    = _metrics(res)
        flag = '  ⚠' if m['stop_pct'] > SPY_STOP_BASELINE else ''
        print(f'  {rank:<5} {ticker:<7} {m["n"]:>7} {m["wr"]:>5.1f}% '
              f'{m["stop_pct"]:>5.1f}%{flag} ${m["pnl"]:>10,.2f} '
              f'${m["dd"]:>10,.2f} {m["sharpe"]:>8.2f}')

    print(f'\n  SPY baseline (Parkinson): '
          f'{spy_m["n"]} trades  '
          f'WR {spy_m["wr"]:.1f}%  '
          f'Stop {spy_m["stop_pct"]:.1f}%  '
          f'P&L ${spy_m["pnl"]:,.2f}  '
          f'DD ${spy_m["dd"]:,.2f}  '
          f'Sharpe {spy_m["sharpe"]:.2f}')
    print(f'  ⚠ = stop-loss rate exceeds SPY baseline of {SPY_STOP_BASELINE}%')

    # Sharpe delta vs SPY baseline for each ticker
    print(f'\n  Sharpe delta vs SPY baseline:')
    for ticker, res in ranked:
        m = _metrics(res)
        sign = '+' if m['sharpe'] >= spy_m['sharpe'] else ''
        print(f'    {ticker:<5}  {sign}{m["sharpe"] - spy_m["sharpe"]:+.2f}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    # SPY baseline (Parkinson vol — slightly different from v2's VIX*0.85)
    print('\nRunning SPY baseline (Parkinson vol)...')
    res_spy = run_backtest(df, ['SPY'])
    print_stats(res_spy, 'SPY ONLY — Parkinson vol baseline')
    print_period_sharpe(res_spy, 'SPY ONLY')

    standalone_results = []

    for ticker in NEW_TICKERS:
        print(f'\n{"─"*62}')
        print(f'  Running {ticker}...')
        print(f'{"─"*62}')

        res_solo = run_backtest(df, [ticker])
        print_stats(res_solo, f'{ticker} STANDALONE')
        print_period_sharpe(res_solo, ticker)

        res_comb = run_backtest(df, ['SPY', ticker])
        print_combined_comparison(res_spy, res_solo, res_comb, ticker)

        standalone_results.append((ticker, res_solo))

    print_summary_table(standalone_results, res_spy)


if __name__ == '__main__':
    main()
