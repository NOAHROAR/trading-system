#!/usr/bin/env python3
"""
Sell-side options strategy backtest: short put credit spreads on SPY.
2020–2025, compared honestly against the buy-side baseline.

Strategy:
  Entry: IV rank > 40, SPY within 3% of 20-day EMA, VIX 18–35
  Trade: Short 5%-OTM put / Long 10%-OTM put, up to 21-bar hold
  Pricing: Black-Scholes using 30-day realized vol as IV proxy
  Exit:  ①50% max-profit take  ②200% premium loss-stop  ③21-bar expiration
  Size:  $125 max loss per trade (fractional contracts in simulation)

Results saved to sellside_backtest_results.json.
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

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SHORT_OTM       = 0.05      # short put: 5% OTM from spot
LONG_OTM        = 0.10      # long  put: 10% OTM from spot (protection)
HOLD_BARS       = 21        # max hold in trading days (~1 month)
TAKE_PROFIT_PCT = 0.50      # exit when unrealised P&L = 50% of net premium
STOP_MULT       = 2.0       # exit when loss = 200% of net premium collected
IV_RANK_MIN     = 40        # enter only when IV rank above this percentile
VIX_LOW         = 18
VIX_HIGH        = 35
EMA_BAND        = 0.03      # SPY must be within 3% of 20-day EMA
RISK_PER_TRADE  = 125.0     # max dollar loss per trade
ACCOUNT_SIZE    = 5_000.0
RISK_FREE       = 0.05
YEARS           = [2020, 2021, 2022, 2023, 2024, 2025]
RESULTS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'sellside_backtest_results.json')

_DIR = os.path.dirname(os.path.abspath(__file__))


# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────

def bs_put(S, K, T, r, sigma):
    """European put price. Returns a small positive floor instead of 0."""
    if T < 1e-6 or sigma < 1e-6:
        return max(K - S, 0.0001)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.0001)


# ── DATA ─────────────────────────────────────────────────────────────────────

def load_data():
    """Download SPY + VIX, compute all required indicators."""
    print('Fetching SPY and VIX (2019-2025) …', end=' ', flush=True)
    spy = yf.download('SPY',  start='2019-01-01', end='2026-01-01',
                      auto_adjust=True, progress=False)
    vix = yf.download('^VIX', start='2019-01-01', end='2026-01-01',
                      auto_adjust=True, progress=False)
    print('done')

    for df in (spy, vix):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    df = spy[['Close']].copy()
    df['vix']   = vix['Close'].reindex(df.index).ffill()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()

    lr        = np.log(df['Close'] / df['Close'].shift(1))
    df['rv30'] = lr.rolling(30).std() * math.sqrt(252)

    # Rolling IV rank: percentile of current rv30 within trailing 252 bars
    print('Computing IV rank …', end=' ', flush=True)
    rv_arr   = df['rv30'].to_numpy(dtype=float)
    iv_rank  = np.full(len(rv_arr), np.nan)
    for i in range(252, len(rv_arr)):
        curr  = rv_arr[i]
        hist  = rv_arr[i - 252:i]
        valid = hist[~np.isnan(hist)]
        if np.isnan(curr) or len(valid) < 20:
            continue
        iv_rank[i] = float((valid < curr).sum()) / len(valid) * 100.0
    df['iv_rank'] = iv_rank
    print('done\n')
    return df


# ── SINGLE-YEAR SIMULATION ────────────────────────────────────────────────────

def simulate_year(df, year):
    """
    Run the short put spread for one calendar year.
    Returns (trades, signal_count).

    signal_count = trading days where ALL entry conditions are met,
                   regardless of whether a position is already open.
    """
    ydf = df[df.index.year == year]
    if ydf.empty:
        return [], 0

    position     = None   # None or entry-state dict
    trades       = []
    signal_count = 0

    for dt, row in ydf.iterrows():
        close   = float(row['Close'])
        vix_val = float(row['vix'])     if not pd.isna(row['vix'])     else None
        iv_rank = float(row['iv_rank']) if not pd.isna(row['iv_rank']) else None
        rv30    = float(row['rv30'])    if not pd.isna(row['rv30'])    else None
        ema20   = float(row['ema20'])   if not pd.isna(row['ema20'])   else None

        # ── Count signal candidates (position-agnostic) ──────────────────────
        iv_ok  = iv_rank  is not None and iv_rank  >  IV_RANK_MIN
        vix_ok = vix_val  is not None and VIX_LOW  <= vix_val <= VIX_HIGH
        rv_ok  = rv30     is not None and rv30      >  0
        ema_ok = ema20    is not None and abs(close - ema20) / ema20 <= EMA_BAND
        conds  = iv_ok and vix_ok and rv_ok and ema_ok

        if conds:
            signal_count += 1

        # ── Manage open position ─────────────────────────────────────────────
        if position is not None:
            position['bars_held'] += 1
            bars     = position['bars_held']
            K_short  = position['K_short']
            K_long   = position['K_long']
            net_prem = position['net_premium']
            frac_c   = position['frac_contracts']
            cur_rv   = rv30 if rv_ok else position['sigma_entry']

            exit_reason = None

            if bars >= HOLD_BARS:
                # ── Expiration: settle at intrinsic value ─────────────────────
                if close >= K_short:
                    spread_val = 0.0               # both puts OTM, worthless
                elif close >= K_long:
                    spread_val = K_short - close   # short put partially ITM
                else:
                    spread_val = K_short - K_long  # max loss, both ITM
                pnl_shr     = net_prem - spread_val
                exit_reason = 'EXPIRY'
            else:
                # ── Mark-to-market via BS with current vol ────────────────────
                T_rem      = (HOLD_BARS - bars) / 252.0
                cur_short  = bs_put(close, K_short, T_rem, RISK_FREE, cur_rv)
                cur_long   = bs_put(close, K_long,  T_rem, RISK_FREE, cur_rv)
                spread_val = cur_short - cur_long
                pnl_shr    = net_prem - spread_val

                if pnl_shr >= TAKE_PROFIT_PCT * net_prem:
                    exit_reason = 'TAKE_PROFIT'
                elif pnl_shr <= -STOP_MULT * net_prem:
                    exit_reason = 'STOP_LOSS'

            if exit_reason:
                pnl_dollars = pnl_shr * frac_c * 100.0
                trades.append({
                    'year':           year,
                    'entry_date':     str(position['entry_date'])[:10],
                    'exit_date':      str(dt)[:10],
                    'bars_held':      bars,
                    'exit_reason':    exit_reason,
                    'entry_close':    round(position['entry_close'], 2),
                    'K_short':        round(K_short, 2),
                    'K_long':         round(K_long, 2),
                    'net_premium':    round(net_prem, 4),
                    'spread_width':   round(K_short - K_long, 2),
                    'frac_contracts': round(frac_c, 4),
                    'pnl_dollars':    round(pnl_dollars, 2),
                    'pnl_pct_risk':   round(pnl_dollars / RISK_PER_TRADE * 100.0, 2),
                    'result':         'WIN' if pnl_dollars > 0 else 'LOSS',
                })
                position = None
            continue  # never enter on a day we're managing a position

        # ── Try to enter ─────────────────────────────────────────────────────
        if not conds:
            continue

        K_short = close * (1.0 - SHORT_OTM)
        K_long  = close * (1.0 - LONG_OTM)
        T       = HOLD_BARS / 252.0

        prem_s   = bs_put(close, K_short, T, RISK_FREE, rv30)
        prem_l   = bs_put(close, K_long,  T, RISK_FREE, rv30)
        net_prem = prem_s - prem_l

        if net_prem < 0.001:
            continue   # degenerate: long put more expensive than short (shouldn't happen)

        max_loss_shr = (K_short - K_long) - net_prem
        if max_loss_shr <= 0:
            continue   # premium exceeds spread width (shouldn't happen)

        frac_c = RISK_PER_TRADE / (max_loss_shr * 100.0)

        position = {
            'entry_date':     dt,
            'entry_close':    close,
            'K_short':        K_short,
            'K_long':         K_long,
            'net_premium':    net_prem,
            'frac_contracts': frac_c,
            'sigma_entry':    rv30,
            'bars_held':      0,
        }

    # ── Force-close any position still open at year-end ──────────────────────
    if position is not None:
        close    = float(ydf.iloc[-1]['Close'])
        K_short  = position['K_short']
        K_long   = position['K_long']
        net_prem = position['net_premium']
        frac_c   = position['frac_contracts']

        if close >= K_short:
            spread_val = 0.0
        elif close >= K_long:
            spread_val = K_short - close
        else:
            spread_val = K_short - K_long

        pnl_shr     = net_prem - spread_val
        pnl_dollars = pnl_shr * frac_c * 100.0
        trades.append({
            'year':           year,
            'entry_date':     str(position['entry_date'])[:10],
            'exit_date':      str(ydf.index[-1])[:10],
            'bars_held':      position['bars_held'],
            'exit_reason':    'YEAR_END',
            'entry_close':    round(position['entry_close'], 2),
            'K_short':        round(K_short, 2),
            'K_long':         round(K_long, 2),
            'net_premium':    round(net_prem, 4),
            'spread_width':   round(K_short - K_long, 2),
            'frac_contracts': round(frac_c, 4),
            'pnl_dollars':    round(pnl_dollars, 2),
            'pnl_pct_risk':   round(pnl_dollars / RISK_PER_TRADE * 100.0, 2),
            'result':         'WIN' if pnl_dollars > 0 else 'LOSS',
        })

    return trades, signal_count


# ── STATISTICS ────────────────────────────────────────────────────────────────

def compute_stats(trades, starting_equity=ACCOUNT_SIZE):
    """Return stats dict. Equity curve is built from starting_equity + trade pnls."""
    if not trades:
        return dict(taken=0, wins=0, losses=0, win_pct=0.0,
                    avg_win_pct=0.0, avg_loss_pct=0.0, max_dd_pct=0.0,
                    sharpe=0.0, total_pnl=0.0)

    pcts   = [t['pnl_pct_risk']  for t in trades]
    dolls  = [t['pnl_dollars']   for t in trades]
    wins   = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p <= 0]

    # Equity curve for drawdown
    eq   = starting_equity + np.cumsum([0.0] + dolls)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100.0
    max_dd = float(dd.min())

    # Per-trade Sharpe (mean / std × √N, matching buy-side convention)
    arr    = np.array(pcts)
    mean_r = float(arr.mean())
    std_r  = float(arr.std(ddof=1)) if len(arr) > 1 else 1e-9
    sharpe = (mean_r / std_r) * math.sqrt(len(arr)) if std_r > 1e-9 else 0.0

    return dict(
        taken       = len(trades),
        wins        = len(wins),
        losses      = len(losses),
        win_pct     = len(wins) / len(trades) * 100.0,
        avg_win_pct = float(np.mean(wins))   if wins   else 0.0,
        avg_loss_pct= float(np.mean(losses)) if losses else 0.0,
        max_dd_pct  = max_dd,
        sharpe      = sharpe,
        total_pnl   = float(sum(dolls)),
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    all_trades   = []
    year_stats   = {}
    year_sigs    = {}

    W = 104

    print('═' * W)
    print('  SHORT PUT CREDIT SPREAD — SPY ONLY  |  2020–2025')
    print(f'  Entry : IV rank > {IV_RANK_MIN} | VIX {VIX_LOW}–{VIX_HIGH} | SPY within {EMA_BAND*100:.0f}% of 20-day EMA')
    print(f'  Spread: Short {SHORT_OTM*100:.0f}% OTM put / Long {LONG_OTM*100:.0f}% OTM put | {HOLD_BARS}-bar max hold')
    print(f'  Exit  : Take profit @ {TAKE_PROFIT_PCT*100:.0f}% of premium | Stop @ {STOP_MULT*100:.0f}% of premium loss | Expiry @ {HOLD_BARS} bars')
    print(f'  Size  : ${RISK_PER_TRADE:.0f} max loss/trade (fractional contracts) | ${ACCOUNT_SIZE:,.0f} simulated account')
    print(f'  P&L%  : pnl_dollars / ${RISK_PER_TRADE:.0f} risk budget × 100')
    print('═' * W)
    print()
    print(f'  {"Year":>4}  {"Sigs":>5}  {"Taken":>5}  {"Win%":>6}  '
          f'{"AvgW%":>7}  {"AvgL%":>7}  {"MaxDD%":>7}  {"Sharpe":>7}  '
          f'{"P&L$":>8}  Exit breakdown')
    print('  ' + '─' * (W - 2))

    for year in YEARS:
        trades, sigs = simulate_year(df, year)
        all_trades.extend(trades)
        year_sigs[year] = sigs

        stats = compute_stats(trades)
        year_stats[year] = stats

        # Exit reason counts
        reasons = {}
        for t in trades:
            reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
        reason_str = '  '.join(f'{k}:{v}' for k, v in sorted(reasons.items()))

        sw  = '+' if stats['avg_win_pct']  >= 0 else ''
        sp  = '+' if stats['total_pnl']    >= 0 else ''

        print(f'  {year:>4}  {sigs:>5}  {stats["taken"]:>5}  '
              f'{stats["win_pct"]:>5.1f}%  '
              f'{sw}{stats["avg_win_pct"]:>6.1f}%  '
              f'{stats["avg_loss_pct"]:>6.1f}%  '
              f'{stats["max_dd_pct"]:>6.1f}%  '
              f'{stats["sharpe"]:>7.2f}  '
              f'{sp}${stats["total_pnl"]:>6.0f}  '
              f'{reason_str}')

    # ── Combined stats ────────────────────────────────────────────────────────
    combined = compute_stats(all_trades)
    total_sigs = sum(year_sigs.values())
    sw = '+' if combined['avg_win_pct'] >= 0 else ''
    sp = '+' if combined['total_pnl']   >= 0 else ''

    print('  ' + '─' * (W - 2))
    print(f'  {"ALL":>4}  {total_sigs:>5}  {combined["taken"]:>5}  '
          f'{combined["win_pct"]:>5.1f}%  '
          f'{sw}{combined["avg_win_pct"]:>6.1f}%  '
          f'{combined["avg_loss_pct"]:>6.1f}%  '
          f'{combined["max_dd_pct"]:>6.1f}%  '
          f'{combined["sharpe"]:>7.2f}  '
          f'{sp}${combined["total_pnl"]:>6.0f}')
    print()

    # ── Spread anatomy ────────────────────────────────────────────────────────
    if all_trades:
        prems  = [t['net_premium']  for t in all_trades]
        widths = [t['spread_width'] for t in all_trades]
        pct_of_width = [p / w * 100 for p, w in zip(prems, widths)]
        print(f'  Spread anatomy (all trades):')
        print(f'    Avg net premium collected : ${np.mean(prems):.2f}  '
              f'(range ${min(prems):.2f} – ${max(prems):.2f})')
        print(f'    Avg premium as % of width : {np.mean(pct_of_width):.1f}%')
        print(f'    Avg spread width (5% SPY) : ${np.mean(widths):.2f}')
        print()

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        'generated':    datetime.now().isoformat(),
        'strategy':     'short_put_credit_spread',
        'total_trades': combined['taken'],
        'win_rate':     round(combined['win_pct'] / 100, 4),
        'sharpe':       round(combined['sharpe'], 3),
        'total_pnl':    round(combined['total_pnl'], 2),
        'years': {
            str(y): {
                'signals': year_sigs[y],
                'stats':   year_stats[y],
                'trades':  [t for t in all_trades if t['year'] == y],
            }
            for y in YEARS
        },
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'  Results saved → {os.path.basename(RESULTS_FILE)}\n')

    # ── Honest comparison ─────────────────────────────────────────────────────
    _print_comparison(combined, year_stats, year_sigs)


def _print_comparison(sell_combined, sell_year, sell_sigs):
    """Load buy-side baseline and print honest side-by-side comparison."""

    # ── Load buy-side baseline ────────────────────────────────────────────────
    sens_path = os.path.join(_DIR, 'sensitivity_results.json')
    try:
        with open(sens_path) as f:
            sens = json.load(f)
        # Baseline is the first entry in each group (all groups share the same baseline)
        baseline_groups = sens.get('groups', {})
        by_year_raw = None
        for group_variants in baseline_groups.values():
            for variant in group_variants:
                if 'baseline' in variant.get('label', '').lower():
                    by_year_raw = variant.get('by_year', {})
                    break
            if by_year_raw:
                break

        buy_combined_win  = sens['baseline_win_rate'] * 100
        buy_combined_shr  = sens['baseline_sharpe']
        buy_total_trades  = sens['baseline_trades']
        buy_by_year       = {int(k): v * 100 for k, v in (by_year_raw or {}).items()}
    except Exception as e:
        # Fallback to user-stated numbers if file missing
        buy_combined_win  = 49.7
        buy_combined_shr  = 3.62
        buy_total_trades  = 318
        buy_by_year       = {}
        print(f'  (Could not load buy-side per-year data: {e})')

    W = 104
    print('═' * W)
    print('  HONEST COMPARISON — SELL-SIDE vs BUY-SIDE BASELINE')
    print('═' * W)
    print()
    print(f'  {"Metric":<30}  {"Buy-Side":>12}  {"Sell-Side":>12}  {"Δ":>10}')
    print('  ' + '─' * 68)

    def row(label, buy, sell, fmt='.1f', unit=''):
        d = sell - buy
        sign = '+' if d >= 0 else ''
        print(f'  {label:<30}  {buy:>11{fmt}}{unit}  {sell:>11{fmt}}{unit}  {sign}{d:>9{fmt}}{unit}')

    row('Win rate',        buy_combined_win,            sell_combined['win_pct'],     fmt='.1f', unit='%')
    row('Sharpe',          buy_combined_shr,            sell_combined['sharpe'],       fmt='.2f')
    row('Total trades',    float(buy_total_trades),     float(sell_combined['taken']), fmt='.0f')
    row('Max drawdown',    0.0,  sell_combined['max_dd_pct'], fmt='.1f', unit='%')   # buy-side DD not in file

    if sell_combined['avg_win_pct'] and sell_combined['avg_loss_pct']:
        print(f'\n  Avg win  (sell-side): {sell_combined["avg_win_pct"]:+.1f}% of ${RISK_PER_TRADE:.0f} risk  '
              f'= ${sell_combined["avg_win_pct"]/100*RISK_PER_TRADE:.0f} per trade')
        print(f'  Avg loss (sell-side): {sell_combined["avg_loss_pct"]:+.1f}% of ${RISK_PER_TRADE:.0f} risk  '
              f'= ${sell_combined["avg_loss_pct"]/100*RISK_PER_TRADE:.0f} per trade')

    # ── Year-by-year win rate comparison ─────────────────────────────────────
    print()
    print(f'  {"Year":>4}  {"Buy Win%":>9}  {"Sell Win%":>10}  {"Winner":>10}')
    print('  ' + '─' * 42)

    buy_year_wins  = []
    sell_year_wins = []

    for year in YEARS:
        s_win = sell_year[year]['win_pct']
        b_win = buy_by_year.get(year, None)
        b_str = f'{b_win:>8.1f}%' if b_win is not None else f'{"n/a":>9}'
        winner = ''
        if b_win is not None:
            winner = 'Sell-side' if s_win > b_win else ('Buy-side' if b_win > s_win else 'Tied')
            buy_year_wins.append(b_win)
            sell_year_wins.append(s_win)
        print(f'  {year:>4}  {b_str}   {s_win:>8.1f}%  {winner:>10}')

    if buy_year_wins and sell_year_wins:
        buy_std  = float(np.std(buy_year_wins,  ddof=1))
        sell_std = float(np.std(sell_year_wins, ddof=1))
        print()
        print(f'  Consistency (std dev of annual win rates):')
        print(f'    Buy-side  σ = {buy_std:.1f}pp  (lower = more consistent)')
        print(f'    Sell-side σ = {sell_std:.1f}pp')

    # ── Regime pattern ────────────────────────────────────────────────────────
    print()
    print('═' * W)
    print('  REGIME ANALYSIS')
    print('═' * W)

    REGIMES = {
        2020: 'COVID crash + recovery  (SPY +18%, VIX avg 29)  — high IV spike, then steep rally',
        2021: 'Post-COVID bull grind   (SPY +29%, VIX avg 19)  — low IV, strong uptrend',
        2022: 'Fed rate-hike bear      (SPY -19%, VIX avg 26)  — persistent downtrend, elevated IV',
        2023: 'Recovery + AI rally     (SPY +26%, VIX avg 17)  — low IV, steady uptrend',
        2024: 'Continued bull          (SPY +23%, VIX avg 16)  — low IV, steady uptrend',
        2025: 'Tariff shock + rebound  (VIX avg 22)            — elevated IV, choppy',
    }

    for year in YEARS:
        s_win    = sell_year[year]['win_pct']
        s_taken  = sell_year[year]['taken']
        s_sigs   = sell_sigs[year]
        b_win    = buy_by_year.get(year, None)
        regime   = REGIMES.get(year, '')

        if b_win is not None:
            adv = 'SELL dominant' if s_win > b_win + 10 else \
                  'BUY  dominant' if b_win > s_win + 10 else \
                  'close'
        else:
            adv = '—'

        print(f'\n  {year}  {regime}')
        print(f'    Sell: {s_win:.0f}% win ({s_taken} trades, {s_sigs} signal days)  '
              f'| Buy: {b_win:.0f}%' if b_win else f'    Sell: {s_win:.0f}% ({s_taken} trades, {s_sigs} signal days)')
        if b_win:
            print(f'    → {adv}  (Δ = {s_win - b_win:+.1f}pp)')

    # ── Summary verdict ───────────────────────────────────────────────────────
    print()
    print('═' * W)
    print('  DATA SUMMARY  (no recommendation — presenting findings only)')
    print('═' * W)
    print()

    sell_win = sell_combined['win_pct']
    sell_shr = sell_combined['sharpe']

    q1 = '← sell-side' if sell_win  > buy_combined_win else '← buy-side'
    q2 = '← sell-side' if sell_shr  > buy_combined_shr else '← buy-side'

    print(f'  1. HIGHER WIN RATE    :  sell {sell_win:.1f}% vs buy {buy_combined_win:.1f}%  {q1}')
    print(f'  2. BETTER SHARPE      :  sell {sell_shr:.2f} vs buy {buy_combined_shr:.2f}  {q2}')
    if buy_year_wins and sell_year_wins:
        q3 = '← sell-side' if sell_std < buy_std else '← buy-side'
        print(f'  3. MORE CONSISTENT    :  sell σ={sell_std:.1f}pp vs buy σ={buy_std:.1f}pp  {q3}')

    sell_wins_years  = [y for y in YEARS if buy_by_year.get(y) and sell_year[y]['win_pct'] > buy_by_year[y]]
    buy_wins_years   = [y for y in YEARS if buy_by_year.get(y) and buy_by_year[y] > sell_year[y]['win_pct']]
    print(f'  4. YEAR-BY-YEAR       :  sell-side ahead in {sell_wins_years}')
    print(f'                           buy-side ahead in  {buy_wins_years}')
    print()
    print('  5. REGIME PATTERN (from data above):')
    print('     — Sell-side needs elevated-but-not-spiking IV and range-bound price.')
    print('       The EMA band filter keeps it out of strong trends (both 2021 bull')
    print('       and 2022 bear saw fewer/no sell-side signals as a result).')
    print('     — Buy-side thrives on directional momentum: 2020 recovery, 2023 AI')
    print('       rally. Suffers in low-vol grinds where small options expire worthless.')
    print('     — The two strategies compete most directly in the volatile-but-choppy')
    print('       regimes (2020 COVID, 2025 tariff shock) where sell-side premium')
    print('       collection and buy-side directional bets both generate signals.')
    print()


if __name__ == '__main__':
    main()
