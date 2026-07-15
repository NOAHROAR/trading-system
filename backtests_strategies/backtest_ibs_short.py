#!/usr/bin/env python3
"""
IBS mean-reversion — SHORT HOLD PERIOD variants.
Tests 4 configurations: 2 DTE, 3 DTE, 3 DTE tight stop (-30%), 5 DTE.

Signal:  IBS < 0.20 → bullish (ATM call)   IBS > 0.80 → bearish (ATM put)
Filters: VIX 15-35, proxy IV rank ≤ 50  (identical to original IBS test)
Data:    daily SPY + QQQ + VIX, 2019 start for IV-rank lookback, 2020-2025 results

All prior backtest files are untouched.
"""

import json
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TICKERS     = ['SPY', 'QQQ']
YEARS       = [2020, 2021, 2022, 2023, 2024, 2025]
IBS_BULL    = 0.20        # IBS < this  → bullish
IBS_BEAR    = 0.80        # IBS > this  → bearish
VIX_LOW     = 15
VIX_HIGH    = 35
RISK_FREE   = 0.05
IV_RANK_MAX = 50
COOLDOWN_N  = 3
TODAY       = datetime.today().strftime('%Y-%m-%d')
OUTFILE     = '/Users/noahrourke/trading-system/ibs_short_results.json'

VARIANTS = [
    dict(label='2 DTE',            key='2dte',        dte=2, max_hold=2, stop_pct=-0.45, take_half=0.50, full_exit=1.00),
    dict(label='3 DTE',            key='3dte',        dte=3, max_hold=3, stop_pct=-0.45, take_half=0.50, full_exit=1.00),
    dict(label='3 DTE tight stop', key='3dte_tight',  dte=3, max_hold=3, stop_pct=-0.30, take_half=0.50, full_exit=1.00),
    dict(label='5 DTE',            key='5dte',        dte=5, max_hold=5, stop_pct=-0.45, take_half=0.50, full_exit=1.00),
]


# ── OPTION PRICING ────────────────────────────────────────────────────────────

def bs(S, K, T, r, sig, typ='call'):
    if T < 1e-6 or sig < 1e-6 or S <= 0 or K <= 0:
        iv = max(S - K, 0) if typ == 'call' else max(K - S, 0)
        return max(iv, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == 'call':
        return max(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 0.01)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.01)


def simulate_exit_v(entry_px, fwd, sigma, direction,
                    dte, max_hold, stop_pct, take_half, full_exit):
    """Check daily for stop / take-half trail / full target / expiry."""
    K    = entry_px
    typ  = 'call' if direction == 'bullish' else 'put'
    e0   = bs(entry_px, K, dte / 365, RISK_FREE, sigma, typ)
    half = False

    for day, S in enumerate(fwd[:max_hold], 1):
        T   = max((dte - day) / 365, 0.001)
        cur = bs(float(S), K, T, RISK_FREE, sigma, typ)
        pnl = (cur - e0) / e0

        if pnl <= stop_pct:
            return float(S), day, 'STOP', pnl
        if pnl >= full_exit:
            return float(S), day, 'FULL_TARGET', pnl
        if not half and pnl >= take_half:
            half = True
        if half and pnl < take_half * 0.5:
            return float(S), day, 'TRAIL_EXIT', (take_half + pnl) / 2

    if not fwd:
        return entry_px, 0, 'NO_DATA', 0.0
    day = min(max_hold, len(fwd))
    S   = float(fwd[day - 1])
    cur = bs(S, K, max((dte - day) / 365, 0.001), RISK_FREE, sigma, typ)
    return S, day, 'TIME_EXIT', (cur - e0) / e0


# ── VOLATILITY HELPERS ────────────────────────────────────────────────────────

def rv30_series(close):
    return np.log(close / close.shift(1)).rolling(30).std() * math.sqrt(252)


def get_iv_rank(rv_ser, as_of, lookback=252):
    idx = rv_ser.index.searchsorted(as_of, side='right') - 1
    if idx < lookback:
        return None
    curr = rv_ser.iloc[idx]
    hist = rv_ser.iloc[idx - lookback:idx].dropna()
    if pd.isna(curr) or len(hist) < 20:
        return None
    return float((hist < curr).sum() / len(hist) * 100)


def get_sigma(close, as_of):
    idx = close.index.searchsorted(as_of, side='right')
    sub = close.iloc[max(0, idx - 31):idx]
    if len(sub) < 10:
        return None
    return float(np.log(sub / sub.shift(1)).dropna().std() * math.sqrt(252))


# ── STATISTICS ────────────────────────────────────────────────────────────────

def compute_stats(trades):
    if not trades:
        return {}
    pnls   = [t['pnl_pct'] / 100 for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    eq = pk = 1.0
    mdd = 0.0
    for p in pnls:
        eq *= (1 + p * 0.1)
        if eq > pk:
            pk = eq
        mdd = max(mdd, (pk - eq) / pk)

    arr    = np.array(pnls)
    sharpe = float(arr.mean() / arr.std() * math.sqrt(252)) if arr.std() > 1e-9 else 0.0

    return {
        'total_trades':     len(trades),
        'wins':             len(wins),
        'losses':           len(losses),
        'win_rate':         round(len(wins) / len(trades), 4),
        'avg_winner_pct':   round(np.mean(wins) * 100,   2) if wins   else 0.0,
        'avg_loser_pct':    round(np.mean(losses) * 100, 2) if losses else 0.0,
        'max_drawdown_pct': round(mdd * 100, 2),
        'sharpe':           round(sharpe, 3),
    }


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def download_all():
    print('▶  Downloading data (once, shared across all variants) …')
    daily = {}
    for t in TICKERS:
        print(f'   {t} daily … ', end='', flush=True)
        df = yf.download(t, start='2019-01-01', end=TODAY,
                         interval='1d', auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        daily[t] = df
        print(f'{len(df)} bars')

    print('   VIX daily … ', end='', flush=True)
    vix = yf.download('^VIX', start='2019-01-01', end=TODAY,
                      interval='1d', auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    print(f'{len(vix)} bars')
    return daily, vix


# ── CORE BACKTEST ─────────────────────────────────────────────────────────────

def run_ibs_variant(daily, vix_df, variant):
    """Run one DTE/stop variant across all tickers and years."""
    dte       = variant['dte']
    max_hold  = variant['max_hold']
    stop_pct  = variant['stop_pct']
    take_half = variant['take_half']
    full_exit = variant['full_exit']

    vix_close = vix_df['Close']
    rv_sers   = {t: rv30_series(daily[t]['Close']) for t in TICKERS}

    all_trades   = []
    year_results = {}

    for year in YEARS:
        y0 = pd.Timestamp(f'{year}-01-01')
        y1 = pd.Timestamp(f'{year}-12-31')

        year_trades = []
        total_sig = iv_skip = cd_skip = 0
        raw_sigs  = []

        for ticker in TICKERS:
            df    = daily[ticker]
            df_yr = df[(df.index >= y0) & (df.index <= y1)].copy()
            rng   = df_yr['High'] - df_yr['Low']
            df_yr['ibs'] = (df_yr['Close'] - df_yr['Low']) / rng.replace(0, np.nan)

            for ts, row in df_yr.iterrows():
                ibs_val = row['ibs']
                if pd.isna(ibs_val):
                    continue
                if ibs_val < IBS_BULL:
                    direction = 'bullish'
                elif ibs_val > IBS_BEAR:
                    direction = 'bearish'
                else:
                    continue
                raw_sigs.append({
                    'ts': ts, 'ticker': ticker,
                    'direction': direction,
                    'close': float(row['Close']),
                    'ibs': round(float(ibs_val), 4),
                })

        raw_sigs.sort(key=lambda x: x['ts'])
        day_streak = {}

        for sig in raw_sigs:
            total_sig += 1
            t   = sig['ticker']
            ts  = sig['ts']
            sd  = pd.Timestamp(ts.date())
            c   = sig['close']

            # VIX filter (silent skip, not counted separately)
            vix_row = vix_close[vix_close.index <= sd]
            if vix_row.empty:
                continue
            vix_val = float(vix_row.iloc[-1])
            if pd.isna(vix_val) or not (VIX_LOW <= vix_val <= VIX_HIGH):
                continue

            # IV rank filter
            iv_rank = get_iv_rank(rv_sers[t], sd)
            if iv_rank is None or iv_rank > IV_RANK_MAX:
                iv_skip += 1
                continue

            # Same-day consecutive-loss cooldown
            date_key = ts.date()
            streak   = day_streak.get(date_key, 0)
            if streak >= COOLDOWN_N:
                cd_skip += 1
                continue

            # Sigma
            sigma = get_sigma(daily[t]['Close'], sd)
            if sigma is None or sigma <= 0:
                continue

            # Forward closes for exit simulation
            cd      = daily[t]['Close']
            fwd_idx = cd.index.searchsorted(sd, side='right')
            fwd     = list(cd.iloc[fwd_idx:fwd_idx + max_hold])
            if not fwd:
                continue

            ex_px, days, ex_type, pnl = simulate_exit_v(
                c, fwd, sigma, sig['direction'],
                dte, max_hold, stop_pct, take_half, full_exit,
            )
            result = 'WIN' if pnl > 0 else 'LOSS'
            day_streak[date_key] = (streak + 1) if result == 'LOSS' else 0

            trade = {
                'date':          str(ts.date()),
                'ticker':        t,
                'direction':     sig['direction'],
                'ibs':           sig['ibs'],
                'proxy_iv_rank': round(iv_rank, 1),
                'sigma':         round(sigma, 4),
                'entry_price':   round(c, 2),
                'exit_price':    round(ex_px, 2),
                'days_held':     days,
                'exit_type':     ex_type,
                'pnl_pct':       round(pnl * 100, 2),
                'result':        result,
                'year':          year,
            }
            year_trades.append(trade)
            all_trades.append(trade)

        stats = compute_stats(year_trades)
        year_results[year] = {
            'year':           year,
            'total_signals':  total_sig,
            'iv_skips':       iv_skip,
            'cooldown_skips': cd_skip,
            'signals_taken':  len(year_trades),
            'win_rate':       round(stats.get('win_rate', 0), 4),
            'stats':          stats,
        }

    return year_results, all_trades


# ── REPORTING ─────────────────────────────────────────────────────────────────

def _combined_stats(all_trades):
    n   = len(all_trades)
    if n == 0:
        return 0, 0.0, 0.0
    wr  = len([t for t in all_trades if t['result'] == 'WIN']) / n
    arr = np.array([t['pnl_pct'] / 100 for t in all_trades])
    sh  = float(arr.mean() / arr.std() * math.sqrt(252)) if arr.std() > 1e-9 else 0.0
    return n, wr, sh


def print_variant_table(year_results, all_trades, variant):
    lbl  = variant['label']
    stop = variant['stop_pct']
    print(f"\n{'='*76}")
    print(f"  {lbl.upper()}  |  stop {stop*100:.0f}%  |  IBS<0.20 bull / IBS>0.80 bear")
    print(f"{'='*76}")
    hdr = (f"{'Year':<5}  {'Sigs':>6}  {'IVSkip':>7}  {'Taken':>6}  "
           f"{'Win%':>6}  {'AvgW%':>7}  {'AvgL%':>7}  {'MaxDD%':>7}  {'Sharpe':>7}")
    print(hdr)
    print('─' * 76)

    beats_55 = 0
    for year in YEARS:
        yr  = year_results.get(year, {})
        s   = yr.get('stats', {})
        n   = yr.get('signals_taken', 0)
        wr  = s.get('win_rate', 0)
        if n > 0 and wr >= 0.55:
            beats_55 += 1
        flag = (' ✓' if (n > 0 and wr >= 0.55)
                else  (' ⚠' if (n > 0 and wr < 0.45) else ''))
        print(f"{year:<5}  "
              f"{yr.get('total_signals', 0):>6}  "
              f"{yr.get('iv_skips', 0):>7}  "
              f"{n:>6}  "
              f"{wr:>6.1%}  "
              f"{s.get('avg_winner_pct', 0):>7.1f}  "
              f"{s.get('avg_loser_pct', 0):>7.1f}  "
              f"{s.get('max_drawdown_pct', 0):>7.1f}  "
              f"{s.get('sharpe', 0):>7.2f}{flag}")

    print('─' * 76)
    n_all, wr_all, sh_all = _combined_stats(all_trades)
    print(f"\nCombined  {n_all} trades | {wr_all:.1%} win rate | "
          f"Sharpe {sh_all:.2f} | {beats_55}/6 years ≥55%")


def print_final_comparison(results_by_key):
    """Summary table + honest assessment vs both benchmarks."""
    print(f"\n{'='*80}")
    print('  COMPARISON: SHORT-DTE VARIANTS  vs  IBS 7 DTE  vs  4-CONDITION BASELINE')
    print(f"{'='*80}")

    W = 26
    print(f"\n  {'Variant':<{W}}  {'Trades':>7}  {'Win%':>6}  {'≥55%':>5}  "
          f"{'Sharpe':>7}  {'AvgW%':>7}  {'AvgL%':>7}")
    print("  " + "─" * (W + 54))

    rows = []
    for v in VARIANTS:
        data   = results_by_key[v['key']]
        trades = data['all_trades']
        n, wr, sh = _combined_stats(trades)
        beats  = sum(1 for y in YEARS
                     if data['year_results'][y].get('signals_taken', 0) > 0
                     and data['year_results'][y].get('win_rate', 0) >= 0.55)
        pnls   = [t['pnl_pct'] for t in trades]
        avg_w  = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_l  = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0
        rows.append((v['label'], n, wr, beats, sh, avg_w, avg_l))
        print(f"  {v['label']:<{W}}  {n:>7}  {wr:>6.1%}  {beats:>3}/6  "
              f"{sh:>7.2f}  {avg_w:>7.1f}  {avg_l:>7.1f}")

    print("  " + "─" * (W + 54))
    # Hardcoded benchmarks from prior runs
    print(f"  {'IBS 7 DTE (orig baseline)':<{W}}  {581:>7}  {38.6/100:>6.1%}  {'0/6':>5}  "
          f"{'n/a':>7}  {'n/a':>7}  {'n/a':>7}")
    print(f"  {'4-cond baseline':<{W}}  {318:>7}  {49.7/100:>6.1%}  {'2/6':>5}  "
          f"{3.62:>7.2f}  {'n/a':>7}  {'n/a':>7}")

    # Per-year win rate table
    print(f"\n  Per-year win rates:")
    labels = [v['label'] for v in VARIANTS]
    header = f"  {'Year':<5}"
    for lbl in labels:
        header += f"  {lbl[:14]:>14}"
    print(header)
    print("  " + "─" * (6 + 17 * len(labels)))
    for year in YEARS:
        row = f"  {year:<5}"
        for v in VARIANTS:
            yr  = results_by_key[v['key']]['year_results'][year]
            n   = yr.get('signals_taken', 0)
            wr  = yr.get('win_rate', 0)
            row += f"  {wr:>13.1%}" if n > 0 else f"  {'n/a':>13}"
        print(row)

    # ── Honest assessment ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('  HONEST ASSESSMENT')
    print(f"{'='*80}")

    ibs7_wr  = 0.386
    cond4_wr = 0.497

    best_row = max(rows, key=lambda x: x[2])
    best_lbl, best_n, best_wr, best_beats, best_sh, _, _ = best_row

    improvement_vs_ibs7  = best_wr - ibs7_wr
    improvement_vs_4cond = best_wr - cond4_wr

    print(f"\n  Benchmarks:")
    print(f"    IBS 7 DTE (original) : 38.6% win rate, 581 trades")
    print(f"    4-condition baseline  : 49.7% win rate, 318 trades\n")

    print(f"  Short-DTE results:")
    for lbl, n, wr, beats, sh, avg_w, avg_l in rows:
        delta = wr - ibs7_wr
        sign  = '+' if delta >= 0 else ''
        print(f"    {lbl:<26}  {wr:.1%}  ({sign}{delta:.1%} vs 7 DTE)  Sharpe {sh:.2f}")

    print()

    # Does shortening help?
    any_beats_ibs7  = any(r[2] > ibs7_wr  for r in rows)
    any_beats_4cond = any(r[2] > cond4_wr for r in rows)

    if any_beats_4cond:
        print(f"  ✓ At least one short-DTE variant BEATS the 4-condition baseline ({cond4_wr:.1%}).")
    elif any_beats_ibs7:
        winning = [r for r in rows if r[2] > ibs7_wr]
        print(f"  Shortening the hold period DOES improve IBS win rate.")
        print(f"  {len(winning)} variant(s) beat the 7 DTE IBS baseline:")
        for r in winning:
            print(f"    → {r[0]}: {r[2]:.1%}")
        print(f"  None of the short-DTE variants reach the 4-condition baseline ({cond4_wr:.1%}).")
    else:
        print(f"  Shortening the hold period does NOT improve IBS.")
        print(f"  No short-DTE variant beats the original 7 DTE IBS baseline ({ibs7_wr:.1%}).")
        print(f"  The IBS signal does not have a reliable edge at these timeframes.")
        print(f"  The 4-condition baseline ({cond4_wr:.1%}) remains the stronger system.")

    # Tight stop comparison (3 DTE only)
    r_3dte  = next((r for r in rows if r[0] == '3 DTE'), None)
    r_tight = next((r for r in rows if r[0] == '3 DTE tight stop'), None)
    if r_3dte and r_tight:
        diff = r_tight[2] - r_3dte[2]
        sign = '+' if diff >= 0 else ''
        direction = 'improves' if diff > 0.005 else ('hurts' if diff < -0.005 else 'negligible effect on')
        print(f"\n  3 DTE stop comparison:")
        print(f"    -45% stop → {r_3dte[2]:.1%}   -30% tight stop → {r_tight[2]:.1%}   "
              f"diff {sign}{diff:.1%}")
        print(f"    Tighter stop {direction} win rate for this DTE.")

    print()
    print("  Note: these results are in-sample on the same 2020-2025 period used for")
    print("  all prior tests. Any 'winner' here is not validated out-of-sample.")
    print()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print('=' * 76)
    print('  IBS SHORT-DTE VARIANTS  |  SPY + QQQ  |  2020-2025')
    print(f'  Variants: {", ".join(v["label"] for v in VARIANTS)}')
    print('=' * 76)

    daily, vix_df = download_all()

    results_by_key = {}
    all_output     = {'generated': datetime.now().isoformat()}

    for v in VARIANTS:
        print(f"\n▶  Running {v['label']} (stop {v['stop_pct']*100:.0f}%) …", flush=True)
        yr, trades = run_ibs_variant(daily, vix_df, v)
        print_variant_table(yr, trades, v)

        results_by_key[v['key']] = {
            'label':        v['label'],
            'config':       {k: val for k, val in v.items() if k != 'key'},
            'year_results': yr,
            'all_trades':   trades,
        }
        n, wr, sh = _combined_stats(trades)
        all_output[v['key']] = {
            'label':             v['label'],
            'config':            {k: val for k, val in v.items() if k != 'key'},
            'total_trades':      n,
            'combined_win_rate': round(wr, 4),
            'combined_sharpe':   round(sh, 3),
            'year_results':      {str(y): yr[y] for y in YEARS},
            'sample_trades':     sorted(trades, key=lambda t: t['date'])[-20:],
        }

    print_final_comparison(results_by_key)

    with open(OUTFILE, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f'✓ Saved {OUTFILE}')


if __name__ == '__main__':
    main()
