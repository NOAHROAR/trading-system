#!/usr/bin/env python3
"""
backtest_xlp_spreads.py — XLP spread-width sensitivity analysis.
Tests $2/$3/$5 spreads against XLP's lower price level.
DO NOT COMMIT — analysis only.

Builds on backtest_gld_xlp.py (same data, same engine, parameterised spread width).
"""

import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Shared strategy parameters (same as credit_spread_strat.py) ───────────────
DTE_MIN        = 6
DTE_MAX        = 8
TARGET_DELTA   = 0.20
DELTA_TOL      = 0.05
PROFIT_TARGET  = 0.50
STOP_LOSS      = 2.00
MAX_POSITIONS  = 2
MIN_IVR        = 30.0
MAX_VIX        = 35.0
SMA_PERIOD     = 20
RISK_FREE      = 0.045
IVR_WINDOW     = 252
PARK_WINDOW    = 30

SPY_STOP_RATE  = 24.1    # flag threshold
SPY_MIN_TRADES = 50      # flag threshold

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    {'label': 'A ($2 / $0.15)', 'spread': 2.0, 'min_credit': 0.15},
    {'label': 'B ($3 / $0.20)', 'spread': 3.0, 'min_credit': 0.20},
    {'label': 'C ($5 / $0.25)', 'spread': 5.0, 'min_credit': 0.25},  # baseline
]

# ── Macro event dates (2020-2026) ─────────────────────────────────────────────
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


# ── Data ───────────────────────────────────────────────────────────────────────

def parkinson_ann(high, low, window=PARK_WINDOW):
    log_hl_sq = np.log(high / low) ** 2
    park_var  = log_hl_sq.rolling(window).mean() / (4 * math.log(2))
    return np.sqrt(park_var * 252) * 100


def ivrank_series(park_vol, window=IVR_WINDOW):
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
    print('Loading SPY, XLP, ^VIX from yfinance (2019-07-01 → 2026-07-04)...')
    raw = yf.download(
        ['SPY', 'XLP', '^VIX'],
        start='2019-07-01', end='2026-07-05',
        auto_adjust=True, progress=True,
    )
    price_df = {}
    for t in ['SPY', 'XLP']:
        df = pd.DataFrame({
            'Close': raw['Close'][t].ffill(),
            'High':  raw['High'][t].ffill(),
            'Low':   raw['Low'][t].ffill(),
        }).dropna()
        df['SMA20']    = df['Close'].rolling(SMA_PERIOD).mean()
        df['park_vol'] = parkinson_ann(df['High'], df['Low'])
        df['ivr']      = ivrank_series(df['park_vol'])
        price_df[t] = df

    vix    = raw['Close']['^VIX'].ffill()
    common = price_df['SPY'].index.intersection(price_df['XLP'].index)
    common = common[common >= pd.Timestamp('2020-01-01')]
    for t in ['SPY', 'XLP']:
        price_df[t] = price_df[t].loc[common].copy()
    vix = vix.reindex(common).ffill()

    print(f'  {len(common)} trading days '
          f'({common[0].date()} → {common[-1].date()})')
    for t in ['SPY', 'XLP']:
        cl = price_df[t]['Close']
        pv = price_df[t]['park_vol'].dropna()
        print(f'  {t}  ${cl.min():.0f}–${cl.max():.0f}'
              f'  vol {pv.mean():.1f}% avg ({pv.min():.1f}–{pv.max():.1f}%)')
    return price_df, vix


# ── Position ───────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ('ticker','entry_date','expiry','short_K','long_K',
                 'credit','profit_tgt','stop_cost','spread_w')

    def __init__(self, ticker, entry_date, expiry, short_K, long_K, credit, spread_w):
        self.ticker     = ticker
        self.entry_date = entry_date
        self.expiry     = expiry
        self.short_K    = short_K
        self.long_K     = long_K
        self.credit     = credit
        self.spread_w   = spread_w
        self.profit_tgt = round(credit * PROFIT_TARGET, 4)
        self.stop_cost  = round(credit * STOP_LOSS,     4)

    def cost_to_close(self, S, T, sigma):
        return spread_credit(S, self.short_K, self.long_K, T, sigma)


def next_expiry(d: date) -> date:
    best, best_diff = None, 999
    for delta in range(DTE_MIN - 1, DTE_MAX + 5):
        c = d + timedelta(days=delta)
        if c.weekday() == 4:
            diff = abs(delta - 7)
            if diff < best_diff:
                best_diff, best = diff, c
    if best is None:
        best = d + timedelta(days=7)
        while best.weekday() >= 5:
            best += timedelta(days=1)
    return best


# ── Backtest engine (parameterised per-ticker spread/credit) ───────────────────

def run_backtest(price_df, vix, use_tickers, ticker_params):
    """
    ticker_params: dict  e.g. {'XLP': {'spread': 2.0, 'min_credit': 0.15},
                                'SPY': {'spread': 5.0, 'min_credit': 0.25}}
    """
    trades    = []
    open_pos  = []
    daily_pnl = []
    equity    = 0.0
    peak      = 0.0
    max_dd    = 0.0

    blocked = {t: {k: 0 for k in (
        'macro','sma','ivr','vix','min_credit','position_limit','underwater','delta_miss',
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
            else:
                cost = pos.cost_to_close(S, T, sigma)
                if cost <= pos.profit_tgt:
                    reason = 'PROFIT'
                elif cost >= pos.stop_cost:
                    reason = 'STOP'
                else:
                    keep.append(pos)
                    continue

            cost    = round(max(0.0, cost), 4)
            pnl     = round((pos.credit - cost) * 100, 2)
            day_pnl += pnl
            trades.append(dict(
                ticker=pos.ticker, entry=pos.entry_date, exit=d,
                expiry=pos.expiry, short_K=pos.short_K, long_K=pos.long_K,
                credit=pos.credit, spread_w=pos.spread_w,
                close_cost=cost, pnl=pnl, reason=reason,
                held=(d - pos.entry_date).days, year=d.year,
            ))

        open_pos = keep

        # ── Entry evaluation ──────────────────────────────────────────────────
        if d.weekday() < 5:
            macro = is_macro_day(d)

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
                tp         = ticker_params[ticker]
                sw         = tp['spread']
                min_credit = tp['min_credit']

                row   = price_df[ticker].loc[dt]
                S     = float(row['Close'])
                sma   = row.get('SMA20', float('nan'))
                ivr   = row.get('ivr',   float('nan'))
                sigma = row.get('park_vol', float('nan'))

                if len(open_pos) >= MAX_POSITIONS:
                    blocked[ticker]['position_limit'] += 1
                    continue
                if any(p.ticker == ticker for p in open_pos):
                    continue
                if macro:
                    blocked[ticker]['macro'] += 1
                    continue
                if math.isnan(float(sma)) or S <= float(sma):
                    blocked[ticker]['sma'] += 1
                    continue
                if math.isnan(float(ivr)) or float(ivr) < MIN_IVR:
                    blocked[ticker]['ivr'] += 1
                    continue
                if math.isnan(vix_val) or vix_val >= MAX_VIX:
                    blocked[ticker]['vix'] += 1
                    continue
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

                long_K = short_K - sw
                credit = spread_credit(S, short_K, long_K, T, sigma_dec)
                if credit < min_credit:
                    blocked[ticker]['min_credit'] += 1
                    continue

                open_pos.append(Position(ticker, d, expiry, short_K, long_K, credit, sw))

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
            credit=pos.credit, spread_w=pos.spread_w,
            close_cost=cost, pnl=pnl, reason='OPEN_AT_END',
            held=(last_d - pos.entry_date).days, year=last_d.year,
        ))

    return dict(trades=trades, blocked=blocked, daily_pnl=daily_pnl, max_dd=max_dd)


# ── Metrics helper ─────────────────────────────────────────────────────────────

def extract_metrics(res, ticker=None):
    df = pd.DataFrame(res['trades'])
    if ticker:
        df = df[df['ticker'] == ticker]
    if df.empty:
        return None
    wins  = df[df['pnl'] > 0]
    stops = df[df['reason'] == 'STOP']
    dpnl  = pd.Series(res['daily_pnl'], dtype=float)
    sh    = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    return dict(
        n=len(df),
        wr=len(wins)/len(df)*100,
        stop_pct=len(stops)/len(df)*100,
        avg_credit=df['credit'].mean(),
        pnl=df['pnl'].sum(),
        sharpe=sh,
        max_dd=res['max_dd'],
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    price_df, vix = load_data()

    # SPY baseline (same for all scenarios, $5/$0.25)
    spy_params = {'SPY': {'spread': 5.0, 'min_credit': 0.25}}
    print('\nRunning SPY baseline...')
    res_spy = run_backtest(price_df, vix, ['SPY'], spy_params)
    spy_m   = extract_metrics(res_spy, 'SPY')

    # Per-scenario results
    results_xlp  = []    # XLP-only per scenario
    results_comb = []    # SPY+XLP combined per scenario

    for sc in SCENARIOS:
        print(f"Running XLP scenario {sc['label']}...")
        xlp_p = {'XLP': {'spread': sc['spread'], 'min_credit': sc['min_credit']}}
        xlp_all_p = {**spy_params, **xlp_p}

        res_xlp  = run_backtest(price_df, vix, ['XLP'], xlp_p)
        res_comb = run_backtest(price_df, vix, ['SPY', 'XLP'], xlp_all_p)
        results_xlp.append(extract_metrics(res_xlp, 'XLP'))
        results_comb.append(extract_metrics(res_comb))

    # ── Year-by-year for each XLP scenario ───────────────────────────────────
    print('\n\n' + '=' * 70)
    print('  XLP YEAR-BY-YEAR P&L')
    print('=' * 70)

    xlp_yearly = []
    for i, sc in enumerate(SCENARIOS):
        xlp_p = {'XLP': {'spread': sc['spread'], 'min_credit': sc['min_credit']}}
        res   = run_backtest(price_df, vix, ['XLP'], xlp_p)
        df_t  = pd.DataFrame(res['trades'])
        if not df_t.empty:
            xlp_yearly.append(df_t.groupby('year')['pnl'].sum())
        else:
            xlp_yearly.append(pd.Series(dtype=float))

    all_years = sorted(set().union(*[s.index for s in xlp_yearly]))
    col_w = 14
    print(f'  {"Year":<6}', end='')
    for sc in SCENARIOS:
        print(f'{sc["label"]:>{col_w}}', end='')
    print()
    print('  ' + '-' * (6 + col_w * len(SCENARIOS)))
    for yr in all_years:
        print(f'  {yr:<6}', end='')
        for s in xlp_yearly:
            v = s.get(yr, float('nan'))
            print(f'  ${v:>9,.2f}' if not math.isnan(v) else f'  {"—":>10}', end='')
        print()

    # ── XLP-only comparison table ─────────────────────────────────────────────
    W = 76
    print(f'\n\n{"=" * W}')
    print(f'  XLP-ONLY  —  SCENARIO COMPARISON')
    print(f'{"=" * W}')

    rows = [
        ('Trades', 'n', '{:,}', False),
        ('Win rate', 'wr', '{:.1f}%', False),
        ('Stop-loss rate', 'stop_pct', '{:.1f}%', True),   # flag if > 24.1%
        ('Avg credit', 'avg_credit', '${:.4f}', False),
        ('Total P&L', 'pnl', '${:,.2f}', False),
        ('Sharpe', 'sharpe', '{:.2f}', False),
        ('Max drawdown', 'max_dd', '${:,.2f}', False),
    ]

    col = 20
    print(f'  {"Metric":<{col}}', end='')
    for sc in SCENARIOS:
        print(f'{sc["label"]:>{col}}', end='')
    print()
    print('  ' + '-' * (col + col * len(SCENARIOS)))

    flags_xlp = [[] for _ in SCENARIOS]

    for label, key, fmt, is_stop in rows:
        print(f'  {label:<{col}}', end='')
        for i, m in enumerate(results_xlp):
            if m is None:
                print(f'{"N/A":>{col}}', end='')
                continue
            v = m[key]
            s = fmt.format(v)
            if is_stop and v > SPY_STOP_RATE:
                s += ' ⚠️'
                flags_xlp[i].append(f'stop-loss {v:.1f}% > {SPY_STOP_RATE}%')
            print(f'{s:>{col}}', end='')
        print()

    # Flag: insufficient trades
    for i, m in enumerate(results_xlp):
        if m and m['n'] < SPY_MIN_TRADES:
            flags_xlp[i].append(f'only {m["n"]} trades < {SPY_MIN_TRADES} minimum')

    print(f'\n  Flags:')
    for i, sc in enumerate(SCENARIOS):
        fl = flags_xlp[i]
        if fl:
            print(f'  Scenario {sc["label"]}: ⚠️  {" | ".join(fl)}')
        else:
            print(f'  Scenario {sc["label"]}: ✅  all clear')

    # ── Combined SPY+XLP vs SPY-only ──────────────────────────────────────────
    print(f'\n\n{"=" * W}')
    print(f'  COMBINED SPY+XLP  vs  SPY-ONLY  —  SCENARIO COMPARISON')
    print(f'{"=" * W}')

    comb_rows = [
        ('SPY-only trades', None, '{:,}'),
        ('SPY-only P&L', None, '${:,.2f}'),
        ('SPY-only Sharpe', None, '{:.2f}'),
        ('SPY-only DD', None, '${:,.2f}'),
        ('--- Combined ---', None, ''),
        ('Total trades', 'n', '{:,}'),
        ('Win rate', 'wr', '{:.1f}%'),
        ('Total P&L', 'pnl', '${:,.2f}'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Max drawdown', 'max_dd', '${:,.2f}'),
        ('--- vs SPY-only ---', None, ''),
        ('P&L delta', '_pnl_delta', '${:+,.2f}'),
        ('DD delta', '_dd_delta', '${:+,.2f}'),
        ('Sharpe delta', '_sh_delta', '{:+.2f}'),
    ]

    print(f'  {"Metric":<{col}}', end='')
    for sc in SCENARIOS:
        print(f'{sc["label"]:>{col}}', end='')
    print()
    print('  ' + '-' * (col + col * len(SCENARIOS)))

    flags_comb = [[] for _ in SCENARIOS]
    spy_n   = spy_m['n']   if spy_m else 0
    spy_pnl = spy_m['pnl'] if spy_m else 0.0
    spy_sh  = spy_m['sharpe'] if spy_m else 0.0
    spy_dd  = spy_m['max_dd'] if spy_m else 0.0

    for i, m in enumerate(results_comb):
        if m:
            m['_pnl_delta'] = m['pnl'] - spy_pnl
            m['_dd_delta']  = m['max_dd'] - spy_dd
            m['_sh_delta']  = m['sharpe'] - spy_sh

    for label, key, fmt in comb_rows:
        if label.startswith('---'):
            print(f'  {label}')
            continue
        print(f'  {label:<{col}}', end='')
        for i, m in enumerate(results_comb):
            if key is None:
                # SPY-only fixed values
                if 'trades' in label:
                    s = f'{spy_n:,}'
                elif 'P&L' in label:
                    s = f'${spy_pnl:,.2f}'
                elif 'Sharpe' in label:
                    s = f'{spy_sh:.2f}'
                elif 'DD' in label:
                    s = f'${spy_dd:,.2f}'
                else:
                    s = '—'
            elif m is None:
                s = 'N/A'
            else:
                v = m.get(key, float('nan'))
                s = fmt.format(v) if not (isinstance(v, float) and math.isnan(v)) else 'N/A'
                if key == '_pnl_delta' and isinstance(v, float) and v < 0:
                    flags_comb[i].append(f'combined P&L worse by ${abs(v):,.2f}')
            print(f'{s:>{col}}', end='')
        print()

    print(f'\n  Flags (combined):')
    for i, sc in enumerate(SCENARIOS):
        fl = flags_comb[i]
        if fl:
            print(f'  Scenario {sc["label"]}: ⚠️  {" | ".join(fl)}')
        else:
            print(f'  Scenario {sc["label"]}: ✅  combined beats SPY-only')

    # ── Filter blocks for XLP under each scenario ─────────────────────────────
    print(f'\n\n{"=" * W}')
    print(f'  XLP ENTRY FILTER BLOCKS BY SCENARIO')
    print(f'{"=" * W}')
    for sc in SCENARIOS:
        xlp_p = {'XLP': {'spread': sc['spread'], 'min_credit': sc['min_credit']}}
        res   = run_backtest(price_df, vix, ['XLP'], xlp_p)
        b     = res['blocked']['XLP']
        total = sum(b.values())
        print(f'\n  Scenario {sc["label"]}  (total blocked: {total:,}):')
        for k, v in b.items():
            if v:
                print(f'    {k:<16} {v:>6,}  ({v/total*100:4.1f}%)')

    # ── Final verdict ──────────────────────────────────────────────────────────
    print(f'\n\n{"=" * W}')
    print(f'  FINAL VERDICT')
    print(f'{"=" * W}')
    print(f'\n  SPY-only baseline:  {spy_n} trades  P&L ${spy_pnl:,.2f}'
          f'  Sharpe {spy_sh:.2f}  DD ${spy_dd:,.2f}')
    print()

    any_winner = False
    for i, sc in enumerate(SCENARIOS):
        xlp_m = results_xlp[i]
        comb_m = results_comb[i]
        issues = []
        if xlp_m and xlp_m['n'] < SPY_MIN_TRADES:
            issues.append(f'insufficient sample ({xlp_m["n"]} trades)')
        if xlp_m and xlp_m['stop_pct'] > SPY_STOP_RATE:
            issues.append(f'stop rate {xlp_m["stop_pct"]:.1f}% > {SPY_STOP_RATE}%')
        if comb_m and comb_m['pnl'] < spy_pnl:
            issues.append(f'combined P&L ${comb_m["pnl"]:,.2f} < SPY-only ${spy_pnl:,.2f}')

        status = '❌ REJECT' if issues else '✅ VIABLE'
        if not issues:
            any_winner = True
        n_str    = f'{xlp_m["n"]} trades' if xlp_m else 'N/A'
        sr_str   = f'{xlp_m["stop_pct"]:.1f}% stops' if xlp_m else 'N/A'
        pnl_str  = f'XLP P&L ${xlp_m["pnl"]:,.2f}' if xlp_m else 'N/A'
        comb_str = f'comb ${comb_m["pnl"]:,.2f}' if comb_m else 'N/A'
        print(f'  Scenario {sc["label"]:20}  {status}')
        print(f'    {n_str}  |  {sr_str}  |  {pnl_str}  |  {comb_str}')
        if issues:
            for iss in issues:
                print(f'    ⚠️  {iss}')
        print()

    if not any_winner:
        print('  CONCLUSION: No XLP configuration beats or matches SPY-only.')
        print('  SPY-only is the final answer.')
    else:
        print('  At least one scenario is viable — see details above.')


if __name__ == '__main__':
    main()
