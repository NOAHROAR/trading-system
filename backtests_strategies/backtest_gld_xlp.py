#!/usr/bin/env python3
"""
backtest_gld_xlp.py — Standalone backtest: SPY / GLD / XLP put credit spreads.
Tests current credit_spread_strat.py parameters against alternative underlyings.
DO NOT COMMIT — analysis only.

IV method: per-ticker 30-day Parkinson realized vol as both current IV and
252-day historical window (mirrors live _spy_ivrank() Parkinson approach).
"""

import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Strategy parameters (mirror credit_spread_strat.py exactly) ───────────────
DTE_MIN        = 6
DTE_MAX        = 8
TARGET_DELTA   = 0.20
DELTA_TOL      = 0.05
SPREAD_WIDTH   = 5.0
MIN_CREDIT     = 0.25
MAX_POSITIONS  = 2       # shared across all tickers
MAX_PER_TICKER = 1       # max concurrent positions per ticker
PROFIT_TARGET  = 0.50
STOP_LOSS      = 2.00
MIN_IVR        = 30.0
MAX_VIX        = 35.0
SMA_PERIOD     = 20
RISK_FREE      = 0.045
IVR_WINDOW     = 252
PARK_WINDOW    = 30      # days for Parkinson rolling window

SPY_STOP_RATE  = 24.1    # SPY baseline stop-loss hit rate — flag if exceeded

# ── Macro event dates (2020-2026, same as credit_spread_strat.py) ─────────────
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
    '2022-05-11','2022-06-10','2022-07-13','2022-08-10',
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


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def bs_put_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float(max(K - S, 0))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def find_short_strike(S, T, sigma):
    """Invert BS d1 to find put strike at TARGET_DELTA. Returns (K, delta) or (None, None)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return None, None
    d1_target = norm.ppf(1.0 - TARGET_DELTA)
    log_SK    = d1_target * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma**2) * T
    K         = round(S * math.exp(-log_SK))
    d1        = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta     = norm.cdf(d1) - 1.0
    if abs(abs(delta) - TARGET_DELTA) > DELTA_TOL:
        return None, None
    return float(K), float(delta)


def spread_credit(S, short_K, long_K, T, sigma):
    return round(bs_put_price(S, short_K, T, sigma) - bs_put_price(S, long_K, T, sigma), 4)


# ── Data loading ───────────────────────────────────────────────────────────────

def parkinson_ann(high: pd.Series, low: pd.Series, window=PARK_WINDOW) -> pd.Series:
    """Annualised Parkinson vol (%) over a rolling window."""
    log_hl_sq = np.log(high / low) ** 2
    park_var  = log_hl_sq.rolling(window).mean() / (4 * math.log(2))
    return np.sqrt(park_var * 252) * 100


def ivrank_series(park_vol: pd.Series, window=IVR_WINDOW) -> pd.Series:
    """Rolling 252-day IV rank: (current - min) / (max - min) * 100."""
    vals   = park_vol.values.astype(float)
    result = np.full(len(vals), np.nan)
    for i in range(window, len(vals)):
        w     = vals[i - window: i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < 10:
            continue
        lo, hi = valid.min(), valid.max()
        result[i] = ((vals[i] - lo) / (hi - lo) * 100) if hi > lo else 50.0
    return pd.Series(result, index=park_vol.index)


def load_data():
    print('Loading SPY, GLD, XLP, ^VIX from yfinance (2019-07-01 → 2026-07-04)...')
    raw = yf.download(
        ['SPY', 'GLD', 'XLP', '^VIX'],
        start='2019-07-01', end='2026-07-05',
        auto_adjust=True, progress=True,
    )

    # MultiIndex: (field, ticker) → access by raw['Close']['SPY']
    price_df = {}
    for t in ['SPY', 'GLD', 'XLP']:
        df = pd.DataFrame({
            'Close': raw['Close'][t].ffill(),
            'High':  raw['High'][t].ffill(),
            'Low':   raw['Low'][t].ffill(),
        }).dropna()
        df['SMA20']    = df['Close'].rolling(SMA_PERIOD).mean()
        df['park_vol'] = parkinson_ann(df['High'], df['Low'])
        df['ivr']      = ivrank_series(df['park_vol'])
        price_df[t] = df

    vix = raw['Close']['^VIX'].ffill()

    # Align to common trading days (2020-01-01 onward)
    common = price_df['SPY'].index
    for t in ['GLD', 'XLP']:
        common = common.intersection(price_df[t].index)
    common = common[common >= pd.Timestamp('2020-01-01')]

    for t in ['SPY', 'GLD', 'XLP']:
        price_df[t] = price_df[t].loc[common].copy()
    vix = vix.reindex(common).ffill()

    print(f'  {len(common)} trading days ({common[0].date()} → {common[-1].date()})')

    # Avg vol summary
    for t in ['SPY', 'GLD', 'XLP']:
        pv = price_df[t]['park_vol'].dropna()
        cl = price_df[t]['Close']
        print(f'  {t}  price range ${cl.min():.0f}–${cl.max():.0f}'
              f'  park_vol {pv.mean():.1f}% avg  ({pv.min():.1f}–{pv.max():.1f}%)')

    return price_df, vix


# ── Position ───────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ('ticker','entry_date','expiry','short_K','long_K',
                 'credit','profit_tgt','stop_cost')

    def __init__(self, ticker, entry_date, expiry, short_K, long_K, credit):
        self.ticker     = ticker
        self.entry_date = entry_date
        self.expiry     = expiry
        self.short_K    = short_K
        self.long_K     = long_K
        self.credit     = credit
        self.profit_tgt = round(credit * PROFIT_TARGET, 4)
        self.stop_cost  = round(credit * STOP_LOSS, 4)

    def cost_to_close(self, S, T, sigma):
        return spread_credit(S, self.short_K, self.long_K, T, sigma)


# ── Next Friday expiry ─────────────────────────────────────────────────────────

def next_expiry(d: date) -> date:
    """Nearest Friday in the 6-8 DTE window (5-11 day search, closest to 7)."""
    best, best_diff = None, 999
    for delta in range(DTE_MIN - 1, DTE_MAX + 5):
        c = d + timedelta(days=delta)
        if c.weekday() == 4:    # Friday
            diff = abs(delta - 7)
            if diff < best_diff:
                best_diff, best = diff, c
    if best is None:
        best = d + timedelta(days=7)
        while best.weekday() >= 5:
            best += timedelta(days=1)
    return best


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(price_df: dict, vix: pd.Series, use_tickers: list) -> dict:
    trades   = []
    open_pos = []
    daily_pnl = []
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0

    blocked = {t: {k: 0 for k in (
        'macro', 'sma', 'ivr', 'vix', 'min_credit',
        'position_limit', 'underwater', 'delta_miss',
    )} for t in use_tickers}

    idx = price_df[use_tickers[0]].index
    for t in use_tickers[1:]:
        idx = idx.intersection(price_df[t].index)
    idx = idx.sort_values()

    for dt in idx:
        d       = dt.date()
        vix_val = float(vix.get(dt, float('nan')))

        # ── Monitor + close ───────────────────────────────────────────────────
        day_pnl = 0.0
        keep    = []
        for pos in open_pos:
            row   = price_df[pos.ticker].loc[dt]
            S     = float(row['Close'])
            sigma = max(float(row['park_vol']) / 100.0, 0.03)
            T     = max((pos.expiry - d).days / 365.0, 0.0)

            if d >= pos.expiry:
                cost   = max(pos.short_K - S, 0.0) - max(pos.long_K - S, 0.0)
                reason = 'EXPIRY'
            elif pos.cost_to_close(S, T, sigma) <= pos.profit_tgt:
                cost   = pos.cost_to_close(S, T, sigma)
                reason = 'PROFIT'
            elif pos.cost_to_close(S, T, sigma) >= pos.stop_cost:
                cost   = pos.cost_to_close(S, T, sigma)
                reason = 'STOP'
            else:
                keep.append(pos)
                continue

            cost    = round(cost, 4)
            pnl     = round((pos.credit - cost) * 100, 2)
            day_pnl += pnl
            trades.append(dict(
                ticker=pos.ticker, entry=pos.entry_date, exit=d,
                expiry=pos.expiry, short_K=pos.short_K, long_K=pos.long_K,
                credit=pos.credit, close_cost=cost, pnl=pnl,
                reason=reason, held=(d - pos.entry_date).days, year=d.year,
            ))

        open_pos = keep

        # ── Entry evaluation ──────────────────────────────────────────────────
        if d.weekday() < 5:
            macro = is_macro_day(d)

            # Underwater check: any open position with current cost > original credit
            any_underwater = False
            for pos in open_pos:
                pr   = price_df[pos.ticker].loc[dt]
                pS   = float(pr['Close'])
                psig = max(float(pr['park_vol']) / 100.0, 0.03)
                pT   = max((pos.expiry - d).days / 365.0, 0.0)
                if pos.cost_to_close(pS, pT, psig) > pos.credit:
                    any_underwater = True
                    break

            for ticker in use_tickers:
                row   = price_df[ticker].loc[dt]
                S     = float(row['Close'])
                sma   = row.get('SMA20', float('nan'))
                ivr   = row.get('ivr',   float('nan'))
                sigma = row.get('park_vol', float('nan'))

                # Global position limit
                if len(open_pos) >= MAX_POSITIONS:
                    blocked[ticker]['position_limit'] += 1
                    continue

                # Per-ticker limit
                if any(p.ticker == ticker for p in open_pos):
                    continue   # already have one; not a blocked entry

                # Macro
                if macro:
                    blocked[ticker]['macro'] += 1
                    continue

                # SMA above
                if math.isnan(float(sma)) or S <= float(sma):
                    blocked[ticker]['sma'] += 1
                    continue

                # IVR
                if math.isnan(float(ivr)) or float(ivr) < MIN_IVR:
                    blocked[ticker]['ivr'] += 1
                    continue

                # VIX cap
                if math.isnan(vix_val) or vix_val >= MAX_VIX:
                    blocked[ticker]['vix'] += 1
                    continue

                # Underwater gate
                if any_underwater:
                    blocked[ticker]['underwater'] += 1
                    continue

                sigma_dec = max(float(sigma) / 100.0, 0.03)
                expiry    = next_expiry(d)

                if any(p.ticker == ticker and p.expiry == expiry for p in open_pos):
                    continue

                T = max((expiry - d).days / 365.0, 1 / 365)
                short_K, _ = find_short_strike(S, T, sigma_dec)
                if short_K is None:
                    blocked[ticker]['delta_miss'] += 1
                    continue

                long_K = short_K - SPREAD_WIDTH
                credit = spread_credit(S, short_K, long_K, T, sigma_dec)
                if credit < MIN_CREDIT:
                    blocked[ticker]['min_credit'] += 1
                    continue

                open_pos.append(Position(ticker, d, expiry, short_K, long_K, credit))

        # ── Equity tracking ───────────────────────────────────────────────────
        unrealized = 0.0
        for pos in open_pos:
            pr   = price_df[pos.ticker].loc[dt]
            pS   = float(pr['Close'])
            psig = max(float(pr['park_vol']) / 100.0, 0.03)
            pT   = max((pos.expiry - d).days / 365.0, 0.0)
            unrealized += (pos.credit - pos.cost_to_close(pS, pT, psig)) * 100

        equity  += day_pnl
        total_eq = equity + unrealized
        peak     = max(peak, total_eq)
        max_dd   = max(max_dd, peak - total_eq)
        daily_pnl.append(day_pnl)

    # Force-close anything still open
    last_dt = idx[-1]
    last_d  = last_dt.date()
    for pos in open_pos:
        pr   = price_df[pos.ticker].loc[last_dt]
        S    = float(pr['Close'])
        sig  = max(float(pr['park_vol']) / 100.0, 0.03)
        T    = max((pos.expiry - last_d).days / 365.0, 0.0)
        cost = round(pos.cost_to_close(S, T, sig), 4)
        pnl  = round((pos.credit - cost) * 100, 2)
        trades.append(dict(
            ticker=pos.ticker, entry=pos.entry_date, exit=last_d,
            expiry=pos.expiry, short_K=pos.short_K, long_K=pos.long_K,
            credit=pos.credit, close_cost=cost, pnl=pnl,
            reason='OPEN_AT_END', held=(last_d - pos.entry_date).days, year=last_d.year,
        ))

    return dict(trades=trades, blocked=blocked, daily_pnl=daily_pnl, max_dd=max_dd)


# ── Reporting ──────────────────────────────────────────────────────────────────

def _metrics(res):
    df_t = pd.DataFrame(res['trades'])
    if df_t.empty:
        return None
    wins  = df_t[df_t['pnl'] > 0]
    loss  = df_t[df_t['pnl'] <= 0]
    stops = df_t[df_t['reason'] == 'STOP']
    dpnl  = pd.Series(res['daily_pnl'], dtype=float)
    sh    = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    return dict(
        n=len(df_t), wr=len(wins)/len(df_t)*100,
        pnl=df_t['pnl'].sum(),
        avg_win=wins['pnl'].mean() if len(wins) else 0.0,
        avg_loss=loss['pnl'].mean() if len(loss) else 0.0,
        avg_credit=df_t['credit'].mean(),
        stop_pct=len(stops)/len(df_t)*100,
        sharpe=sh, max_dd=res['max_dd'],
    )


def print_ticker_stats(res, ticker_label):
    df_t      = pd.DataFrame(res['trades'])
    max_dd    = res['max_dd']
    daily_pnl = res['daily_pnl']

    # Filter to this ticker's trades if multi-ticker result
    if 'ticker' in df_t.columns and ticker_label in df_t['ticker'].values:
        df_t = df_t[df_t['ticker'] == ticker_label].copy()

    if df_t.empty:
        print(f'\n  {ticker_label}: No trades executed.')
        return

    wins  = df_t[df_t['pnl'] > 0]
    loss  = df_t[df_t['pnl'] <= 0]
    stops = df_t[df_t['reason'] == 'STOP']
    dpnl  = pd.Series(daily_pnl, dtype=float)
    sh    = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    stop_pct = len(stops) / len(df_t) * 100

    flag = f'  ⚠️  STOP RATE {stop_pct:.1f}% > SPY {SPY_STOP_RATE}% — QQQ-killer pattern!' \
           if stop_pct > SPY_STOP_RATE else ''

    yearly = df_t.groupby('year')['pnl'].agg(pnl='sum', trades='count')
    W = 62

    print(f'\n{"=" * W}')
    print(f'  {ticker_label}')
    print(f'{"=" * W}')
    print(f'  Total trades:    {len(df_t):<6}  (wins {len(wins)}  /  losses {len(loss)})')
    print(f'  Win rate:        {len(wins)/len(df_t)*100:.1f}%')
    print(f'  Total P&L:       ${df_t["pnl"].sum():>10,.2f}')
    print(f'  Avg win:         ${wins["pnl"].mean() if len(wins) else 0:>10.2f}')
    print(f'  Avg loss:        ${loss["pnl"].mean() if len(loss) else 0:>10.2f}')
    print(f'  Avg credit:      ${df_t["credit"].mean():>10.4f}')
    print(f'  Sharpe ratio:    {sh:>10.2f}')
    print(f'  Max drawdown:    ${max_dd:>10,.2f}')
    print(f'  Stop-loss rate:  {stop_pct:>9.1f}%{flag}')

    print(f'\n  {"Year":<6} {"P&L":>10} {"Trades":>8}')
    print(f'  {"-" * 24}')
    for yr, row in yearly.iterrows():
        print(f'  {yr:<6} ${row["pnl"]:>9,.2f} {int(row["trades"]):>8}')

    print(f'\n  Exit reasons:')
    for reason, cnt in df_t['reason'].value_counts().items():
        avg = df_t[df_t['reason'] == reason]['pnl'].mean()
        print(f'    {reason:<16} {cnt:>4} ({cnt/len(df_t)*100:4.1f}%)  avg ${avg:+.2f}')


def print_blocks(res, use_tickers):
    blocked = res['blocked']
    print(f'\n  Entry filter blocks:')
    for ticker in use_tickers:
        b = blocked.get(ticker, {})
        total = sum(b.values())
        if total == 0:
            print(f'    [{ticker}] no blocks recorded')
            continue
        print(f'    [{ticker}] total {total:,}')
        for k, v in b.items():
            if v:
                print(f'      {k:<16} {v:>6,}  ({v/total*100:4.1f}%)')


def print_combined_comparison(label_a, res_a, label_b, res_b, spy_pnl, spy_dd, spy_sharpe):
    def m(res):
        df = pd.DataFrame(res['trades'])
        if df.empty:
            return 0, 0.0, 0.0, 0.0, 0.0
        dpnl = pd.Series(res['daily_pnl'], dtype=float)
        sh   = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
        wr   = (df['pnl'] > 0).mean() * 100
        return len(df), wr, df['pnl'].sum(), res['max_dd'], sh

    nA, wrA, pnlA, ddA, shA = m(res_a)
    nB, wrB, pnlB, ddB, shB = m(res_b)

    W = 70
    print(f'\n{"=" * W}')
    print(f'  COMBINED COMPARISON  —  {label_b}  vs  {label_a}  vs  SPY-only')
    print(f'{"=" * W}')
    print(f'  {"Metric":<18} {"SPY-only":>14} {label_a:>18} {label_b:>18}')
    print(f'  {"-" * 68}')
    print(f'  {"Trades":<18} {"—":>14} {nA:>18,} {nB:>18,}')
    print(f'  {"Win rate":<18} {"—":>14} {wrA:>17.1f}% {wrB:>17.1f}%')
    print(f'  {"Total P&L":<18} ${spy_pnl:>13,.2f} ${pnlA:>17,.2f} ${pnlB:>17,.2f}')
    print(f'  {"Max drawdown":<18} ${spy_dd:>13,.2f} ${ddA:>17,.2f} ${ddB:>17,.2f}')
    print(f'  {"Sharpe":<18} {spy_sharpe:>14.2f} {shA:>18.2f} {shB:>18.2f}')
    delta_pnl_a  = pnlA  - spy_pnl
    delta_pnl_b  = pnlB  - spy_pnl
    delta_dd_a   = ddA   - spy_dd
    delta_dd_b   = ddB   - spy_dd
    delta_sh_a   = shA   - spy_sharpe
    delta_sh_b   = shB   - spy_sharpe
    print(f'  {"P&L vs SPY-only":<18} {"baseline":>14} {delta_pnl_a:>+17,.2f} {delta_pnl_b:>+17,.2f}')
    print(f'  {"DD vs SPY-only":<18} {"baseline":>14} {delta_dd_a:>+17,.2f} {delta_dd_b:>+17,.2f}')
    print(f'  {"Sharpe vs SPY":<18} {"baseline":>14} {delta_sh_a:>+17.2f} {delta_sh_b:>+17.2f}')


def print_correlations(price_df):
    print(f'\n{"=" * 62}')
    print(f'  DAILY RETURN CORRELATIONS (2020–2026)')
    print(f'{"=" * 62}')
    rets = {}
    for t in ['SPY', 'GLD', 'XLP']:
        rets[t] = price_df[t]['Close'].pct_change().dropna()
    df_r   = pd.DataFrame(rets).dropna()
    corr   = df_r.corr()
    pairs  = [('SPY', 'GLD'), ('SPY', 'XLP'), ('GLD', 'XLP')]
    for a, b in pairs:
        r = corr.loc[a, b]
        note = 'low correlation — diversification benefit' if abs(r) < 0.30 else \
               'moderate correlation' if abs(r) < 0.60 else 'high correlation'
        print(f'  {a} vs {b}:  r = {r:+.3f}  ({note})')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    price_df, vix = load_data()

    print('\nRunning individual backtests...')
    res_spy = run_backtest(price_df, vix, ['SPY'])
    res_gld = run_backtest(price_df, vix, ['GLD'])
    res_xlp = run_backtest(price_df, vix, ['XLP'])

    print('Running combined backtests...')
    res_spy_gld     = run_backtest(price_df, vix, ['SPY', 'GLD'])
    res_spy_xlp     = run_backtest(price_df, vix, ['SPY', 'XLP'])
    res_spy_gld_xlp = run_backtest(price_df, vix, ['SPY', 'GLD', 'XLP'])

    # ── Individual stats ─────────────────────────────────────────────────────
    for res, label in [(res_spy, 'SPY ONLY'), (res_gld, 'GLD ONLY'), (res_xlp, 'XLP ONLY')]:
        print_ticker_stats(res, label.split()[0])
        print_blocks(res, [label.split()[0]])

    # ── Combined stats ───────────────────────────────────────────────────────
    spy_m = _metrics(res_spy)

    for res, label in [(res_spy_gld, 'SPY+GLD'), (res_spy_xlp, 'SPY+XLP'),
                       (res_spy_gld_xlp, 'SPY+GLD+XLP')]:
        tickers = label.split('+')
        W = 62
        print(f'\n{"=" * W}')
        print(f'  {label} COMBINED')
        print(f'{"=" * W}')
        df_all = pd.DataFrame(res['trades'])
        if not df_all.empty:
            by_t = df_all.groupby('ticker')['pnl'].agg(pnl='sum', n='count', wr=lambda x: (x > 0).mean() * 100)
            for t in tickers:
                if t in by_t.index:
                    r = by_t.loc[t]
                    stops_t = df_all[(df_all['ticker'] == t) & (df_all['reason'] == 'STOP')]
                    sr = len(stops_t) / r['n'] * 100
                    flag = f'  ⚠️  STOP {sr:.1f}% > {SPY_STOP_RATE}%' if sr > SPY_STOP_RATE else ''
                    print(f'  [{t}]  {int(r["n"])} trades  WR {r["wr"]:.1f}%  P&L ${r["pnl"]:,.2f}'
                          f'  stop-loss {sr:.1f}%{flag}')
        print_blocks(res, tickers)

    # ── Comparison tables ────────────────────────────────────────────────────
    spy_m   = _metrics(res_spy)
    spy_pnl = spy_m['pnl']
    spy_dd  = spy_m['max_dd']
    spy_sh  = spy_m['sharpe']

    print_combined_comparison('SPY-only', res_spy,
                              'SPY+GLD',   res_spy_gld,
                              spy_pnl, spy_dd, spy_sh)
    print_combined_comparison('SPY-only', res_spy,
                              'SPY+XLP',   res_spy_xlp,
                              spy_pnl, spy_dd, spy_sh)
    print_combined_comparison('SPY-only', res_spy,
                              'SPY+GLD+XLP', res_spy_gld_xlp,
                              spy_pnl, spy_dd, spy_sh)

    # ── Stop-loss flag summary ───────────────────────────────────────────────
    print(f'\n{"=" * 62}')
    print(f'  STOP-LOSS RATE SUMMARY  (flag threshold: {SPY_STOP_RATE}%)')
    print(f'{"=" * 62}')
    for res, label in [(res_spy, 'SPY'), (res_gld, 'GLD'), (res_xlp, 'XLP')]:
        df_t = pd.DataFrame(res['trades'])
        if df_t.empty:
            print(f'  {label:<6}  no trades')
            continue
        sr   = (df_t['reason'] == 'STOP').mean() * 100
        flag = '  ⚠️  EXCEEDS SPY BASELINE — QQQ-killer pattern' if sr > SPY_STOP_RATE else '  ✅'
        print(f'  {label:<6}  stop-loss rate: {sr:5.1f}%{flag}')

    # ── Correlations ────────────────────────────────────────────────────────
    print_correlations(price_df)


if __name__ == '__main__':
    main()
