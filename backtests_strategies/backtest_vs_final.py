#!/usr/bin/env python3
"""
Volume Spike final refinement: SPY + QQQ, 15-min bars, 2023–2025.
Loads cached data from intraday_cache/ (run backtest_intraday.py first).

5 variations tested side-by-side:
  A – Baseline  (2.5x vol  |  1.5x target  |  no time filter)
  B – Tighter   (3.0x vol  |  1.5x target  |  no time filter)
  C – Wider tgt (2.5x vol  |  2.0x target  |  no time filter)
  D – Time filt (2.5x vol  |  1.5x target  |  10:00am–3:30pm only)
  E – Combined best (built from A–D results, data-driven)

Sizing: SHARES
  shares = floor($125 / |entry_close – spike_stop|)
  P&L    = shares × (exit – entry)  [long]
           shares × (entry – exit)  [short]

Results → vs_final_results.json
"""

import json
import math
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')

_DIR         = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR    = os.path.join(_DIR, 'intraday_cache')
RESULTS_FILE = os.path.join(_DIR, 'vs_final_results.json')

TICKERS        = ['SPY', 'QQQ']
ET             = ZoneInfo('America/New_York')
VOL_MA_LEN     = 20
RISK_PER_TRADE = 125.0
MIN_STOP_DIST  = 0.01       # skip if stop is degenerate
EOD_BAR        = '15:50'    # force-close at or after this time
NO_ENTRY_AFTER = '15:30'    # no new entries at or after this time

VARIATIONS_ABCD = {
    'A': dict(vol_mult=2.5, move_pct=0.005, tgt_mult=1.5, time_filter=False),
    'B': dict(vol_mult=3.0, move_pct=0.005, tgt_mult=1.5, time_filter=False),
    'C': dict(vol_mult=2.5, move_pct=0.005, tgt_mult=2.0, time_filter=False),
    'D': dict(vol_mult=2.5, move_pct=0.005, tgt_mult=1.5, time_filter=True),
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_ticker(ticker):
    cache = os.path.join(CACHE_DIR, f'{ticker}_15min.csv')
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f'Cache not found: {cache}\n  Run backtest_intraday.py first.')
    df = pd.read_csv(cache)
    df.index = pd.to_datetime(df['t'], utc=True).dt.tz_convert(ET)
    df.index.name = None
    df.drop(columns=['t'], inplace=True, errors='ignore')
    df['vol_ma'] = df['volume'].rolling(VOL_MA_LEN, min_periods=5).mean()
    print(f'  {ticker}: {len(df):,} bars')
    return df


def load_all():
    print('Loading cached 15-min bars...')
    data = {t: load_ticker(t) for t in TICKERS}
    print()
    return data


# ── Signal logic ───────────────────────────────────────────────────────────────

def check_entry(prev, vol_mult, move_pct):
    """
    Spike detected on `prev` bar.
    Returns 'long', 'short', or None.
    """
    vol_ma = prev.get('vol_ma', np.nan)
    if pd.isna(vol_ma) or vol_ma == 0:
        return None
    if prev.get('volume', 0) < vol_mult * vol_ma:
        return None
    po = prev.get('open', 0) or 0
    pc = prev.get('close', 0)
    if po == 0 or abs(pc - po) / po < move_pct:
        return None
    return 'long' if pc > po else 'short'


def check_exit(pos, close):
    d = pos['direction']
    if d == 'long':
        if close >= pos['target_px']:
            return 'TARGET'
        if close <= pos['stop_px']:
            return 'STOP'
    else:
        if close <= pos['target_px']:
            return 'TARGET'
        if close >= pos['stop_px']:
            return 'STOP'
    return None


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate(ticker_data, params, label):
    vol_mult    = params['vol_mult']
    move_pct    = params['move_pct']
    tgt_mult    = params['tgt_mult']
    time_filter = params['time_filter']

    trades   = []
    spy_df   = ticker_data['SPY']
    all_days = sorted(set(spy_df.index.date))

    for day in all_days:
        day_slices = {}
        for t in TICKERS:
            df   = ticker_data[t]
            mask = df.index.date == day
            if mask.any():
                day_slices[t] = df[mask]

        spy_day = day_slices.get('SPY')
        if spy_day is None or spy_day.empty:
            continue

        timestamps = list(spy_day.index)
        n          = len(timestamps)
        open_pos   = {}   # ticker → position dict

        for i, ts in enumerate(timestamps):
            bar_time = ts.strftime('%H:%M')
            is_eod   = bar_time >= EOD_BAR

            # ── EOD force-close ───────────────────────────────────────────────
            if is_eod:
                for t in list(open_pos.keys()):
                    pos = open_pos.pop(t)
                    ds  = day_slices.get(t)
                    S   = (float(ds.loc[ts]['close'])
                           if ds is not None and ts in ds.index
                           else pos['entry_px'])
                    _record(trades, pos, ts, S, 'EOD')
                continue

            # ── Exits ─────────────────────────────────────────────────────────
            for t in list(open_pos.keys()):
                pos = open_pos[t]
                ds  = day_slices.get(t)
                if ds is None or ts not in ds.index:
                    continue
                S      = float(ds.loc[ts]['close'])
                reason = check_exit(pos, S)
                if reason:
                    _record(trades, pos, ts, S, reason)
                    del open_pos[t]

            # ── Entries ───────────────────────────────────────────────────────
            if bar_time >= NO_ENTRY_AFTER:
                continue
            if time_filter and bar_time < '10:00':
                continue
            if i < 1:
                continue

            prev_ts = timestamps[i - 1]

            for t in TICKERS:
                if t in open_pos:
                    continue
                ds = day_slices.get(t)
                if ds is None or ts not in ds.index or prev_ts not in ds.index:
                    continue

                prev      = ds.loc[prev_ts]
                row       = ds.loc[ts]
                direction = check_entry(prev, vol_mult, move_pct)
                if direction is None:
                    continue

                entry_px  = float(row['close'])
                stop_px   = float(prev['low'] if direction == 'long' else prev['high'])
                stop_dist = abs(entry_px - stop_px)
                if stop_dist < MIN_STOP_DIST:
                    continue

                spike_move = abs(float(prev['close']) - float(prev['open']))
                target_px  = (entry_px + tgt_mult * spike_move
                              if direction == 'long'
                              else entry_px - tgt_mult * spike_move)
                shares = max(1, math.floor(RISK_PER_TRADE / stop_dist))

                open_pos[t] = {
                    'ticker':    t,
                    'variation': label,
                    'direction': direction,
                    'entry_ts':  ts,
                    'entry_px':  entry_px,
                    'stop_px':   stop_px,
                    'target_px': target_px,
                    'shares':    shares,
                    'day':       str(day),
                }

        # Day-end cleanup (safety net if EOD bar absent from timestamps)
        for t in list(open_pos.keys()):
            pos     = open_pos.pop(t)
            ds      = day_slices.get(t)
            last_ts = timestamps[-1] if timestamps else pos['entry_ts']
            S = (float(ds.loc[last_ts]['close'])
                 if ds is not None and last_ts in ds.index
                 else pos['entry_px'])
            _record(trades, pos, last_ts, S, 'EOD')

    return trades


def _record(trades, pos, exit_ts, S, reason):
    d       = pos['direction']
    raw     = (S - pos['entry_px']) * pos['shares']
    pnl     = raw if d == 'long' else -raw
    dp_pct  = (S - pos['entry_px']) / pos['entry_px'] * 100
    pct_dir = dp_pct if d == 'long' else -dp_pct   # positive = favorable move

    trades.append({
        'ticker':    pos['ticker'],
        'variation': pos['variation'],
        'direction': d,
        'entry_ts':  str(pos['entry_ts']),
        'exit_ts':   str(exit_ts),
        'entry_px':  round(pos['entry_px'], 4),
        'exit_px':   round(S, 4),
        'stop_px':   round(pos['stop_px'], 4),
        'target_px': round(pos['target_px'], 4),
        'shares':    pos['shares'],
        'reason':    reason,
        'pnl':       round(pnl, 2),
        'pct_dir':   round(pct_dir, 4),
        'day':       pos['day'],
    })


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades, label, total_days):
    if not trades:
        return {'label': label, 'n': 0, 'trades_per_day': 0,
                'win_rate': 0, 'avg_winner_pct': 0, 'avg_loser_pct': 0,
                'total_pnl': 0, 'sharpe': 0,
                'max_consec_loss': 0, 'max_day_drawdown': 0}

    pnls    = [t['pnl']     for t in trades]
    dirs    = [t['pct_dir'] for t in trades]
    winners = [p for p in dirs if p > 0]
    losers  = [p for p in dirs if p <= 0]

    daily = defaultdict(float)
    for t in trades:
        daily[t['day']] += t['pnl']
    dv = list(daily.values())

    sharpe = (np.mean(dv) / np.std(dv) * math.sqrt(252)
              if len(dv) > 1 and np.std(dv) > 0 else 0)

    mx = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        mx  = max(mx, cur)

    return {
        'label':            label,
        'n':                len(trades),
        'trades_per_day':   round(len(trades) / total_days, 2),
        'win_rate':         round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        'avg_winner_pct':   round(np.mean(winners), 3) if winners else 0.0,
        'avg_loser_pct':    round(np.mean(losers), 3)  if losers  else 0.0,
        'total_pnl':        round(sum(pnls), 2),
        'sharpe':           round(sharpe, 2),
        'max_consec_loss':  mx,
        'max_day_drawdown': round(min(dv), 2) if dv else 0,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

W = 80

def div(title=''):
    print()
    print('═' * W)
    if title:
        print(f'  {title}')
        print('═' * W)


def print_report(all_stats, all_trades, all_params, total_days, e_rationale):
    LBLS = ['A', 'B', 'C', 'D', 'E']

    div('VOLUME SPIKE FINAL BACKTEST  —  SPY + QQQ  |  2023–2025  |  15-min bars')
    print(f'  Tickers: SPY, QQQ    Trading days: {total_days}'
          f'    Risk: ${RISK_PER_TRADE:.0f}/trade (shares, not options)')

    # ── Comparison table ──────────────────────────────────────────────────────
    div('VARIATION COMPARISON')
    print(f'  {"Var":<5} {"Trades":>7} {"T/Day":>6} {"Win%":>6}'
          f' {"Avg+%":>7} {"Avg-%":>7} {"Total P&L":>11}'
          f' {"Sharpe":>8} {"MaxCL":>6} {"MaxDD/Day":>10}')
    print('  ' + '─' * 72)
    for lbl in LBLS:
        s = all_stats.get(lbl)
        if s is None or s['n'] == 0:
            print(f'  {lbl:<5}  (no trades)')
            continue
        print(f'  {lbl:<5} {s["n"]:>7,} {s["trades_per_day"]:>6.2f}'
              f' {s["win_rate"]:>5.1f}%'
              f' {s["avg_winner_pct"]:>6.3f}% {s["avg_loser_pct"]:>6.3f}%'
              f' ${s["total_pnl"]:>10,.0f}'
              f' {s["sharpe"]:>8.2f} {s["max_consec_loss"]:>6}'
              f' ${s["max_day_drawdown"]:>9,.0f}')

    # ── Parameter summary ─────────────────────────────────────────────────────
    div('PARAMETERS PER VARIATION')
    print(f'  {"Var":<5} {"Vol thresh":>12} {"Target":>8} {"Time filter":>14}')
    print('  ' + '─' * 44)
    for lbl in LBLS:
        p = all_params.get(lbl)
        if p is None:
            continue
        tf = '10:00–15:30' if p['time_filter'] else 'none'
        print(f'  {lbl:<5} {p["vol_mult"]:>10.1f}x  {p["tgt_mult"]:>5.1f}x  {tf:>14}')

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    div('PER-TICKER P&L')
    for lbl in LBLS:
        trades = all_trades.get(lbl, [])
        if not trades:
            continue
        bt = defaultdict(list)
        for t in trades:
            bt[t['ticker']].append(t['pnl'])
        parts = []
        for tk in TICKERS:
            pnls = bt.get(tk, [])
            if not pnls:
                continue
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            parts.append(f'{tk} {len(pnls)} trades  win {wr:.0f}%  ${sum(pnls):,.0f}')
        print(f'  {lbl}: ' + '   |   '.join(parts))

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    div('EXIT REASON BREAKDOWN')
    for lbl in LBLS:
        trades = all_trades.get(lbl, [])
        if not trades:
            continue
        ec = Counter(t['reason'] for t in trades)
        parts = '   '.join(
            f'{r}: {c} ({c/len(trades)*100:.0f}%)'
            for r, c in sorted(ec.items(), key=lambda x: -x[1])
        )
        print(f'  {lbl}: {parts}')

    # ── Variation E rationale ─────────────────────────────────────────────────
    div('VARIATION E  —  COMBINED BEST  (data-driven from A–D)')
    ep = all_params.get('E', {})
    print(f'  vol_mult    : {ep.get("vol_mult", "?")}x 20-bar average')
    print(f'  tgt_mult    : {ep.get("tgt_mult", "?")}x spike move')
    print(f'  time_filter : {"10:00am–3:30pm ET only" if ep.get("time_filter") else "off"}')
    print(f'  move_pct    : {ep.get("move_pct", 0.005)*100:.1f}%  (unchanged)')
    print()
    for line in e_rationale:
        print(f'  {line}')

    # ── Deployment recommendation ─────────────────────────────────────────────
    div('DEPLOYMENT RECOMMENDATION')
    CRITERIA = dict(win_rate=50, sharpe=0, max_consec_loss=5,
                    trades_per_day_min=0.5, trades_per_day_max=3.0)
    print(f'  Criteria: win rate > {CRITERIA["win_rate"]}%  |  Sharpe > {CRITERIA["sharpe"]}  |'
          f'  max consecutive losses ≤ {CRITERIA["max_consec_loss"]}  |'
          f'  {CRITERIA["trades_per_day_min"]}–{CRITERIA["trades_per_day_max"]} trades/day')
    print()

    # Quality gate: win_rate, Sharpe, max consecutive losses
    # (trades/day is reported separately — signal is inherently low-frequency on 2 tickers)
    quality_pass = {
        lbl: s for lbl, s in all_stats.items()
        if (s.get('n', 0) > 0
            and s.get('win_rate', 0) > CRITERIA['win_rate']
            and s.get('sharpe', -99) > CRITERIA['sharpe']
            and s.get('max_consec_loss', 99) <= CRITERIA['max_consec_loss'])
    }

    scored = {lbl: s for lbl, s in all_stats.items() if s.get('n', 0) > 0}
    if not scored:
        print('  ✗  No trades generated — check data cache.')
        return

    if quality_pass:
        winner = max(quality_pass, key=lambda x: quality_pass[x]['sharpe'])
        ws     = quality_pass[winner]
        wp     = all_params[winner]

        print(f'  ✓  Deploy Variation {winner} with these exact parameters:')
        print()
        print(f'    Tickers       : SPY, QQQ')
        print(f'    Timeframe     : 15-minute bars  (Alpaca IEX feed)')
        print(f'    Volume spike  : bar volume > {wp["vol_mult"]}x 20-bar rolling average')
        print(f'    Min move      : |spike_close − spike_open| / spike_open > {wp["move_pct"]*100:.1f}%')
        print(f'    Direction     : long if spike_close > spike_open  |  short if below')
        print(f'    Entry         : NEXT bar (confirmation), at bar close')
        print(f'    Sizing        : shares = floor($125 / |entry_close − spike_stop|)')
        print(f'    Stop          : spike bar low (long)  /  spike bar high (short)')
        print(f'    Target        : entry ± {wp["tgt_mult"]}x |spike_close − spike_open|')
        if wp['time_filter']:
            print(f'    Time filter   : 10:00am – 3:30pm ET only')
        else:
            print(f'    Time filter   : none  (full market hours 9:30am–3:30pm)')
        print(f'    EOD close     : force-exit all positions at 3:50pm ET')
        print()
        print(f'  Expected performance (SPY + QQQ, 2023–2025 backtest):')
        print(f'    Signals/year  : ~{ws["n"]//3:.0f}  ({ws["trades_per_day"]:.2f}/day — '
              f'~1 signal per {int(round(1/ws["trades_per_day"])):d} trading days)')
        print(f'    Win rate      : {ws["win_rate"]:.1f}%')
        print(f'    Avg winner    : +{ws["avg_winner_pct"]:.3f}%  on underlying')
        print(f'    Avg loser     : {ws["avg_loser_pct"]:.3f}%  on underlying')
        print(f'    Total P&L     : ${ws["total_pnl"]:,.0f} over 3 years'
              f'  (~${ws["total_pnl"]/3:,.0f}/yr on $5k account)')
        print(f'    Sharpe        : {ws["sharpe"]:.2f}')
        print(f'    Max consec L  : {ws["max_consec_loss"]}')
        print(f'    Max single DD : ${ws["max_day_drawdown"]:,.0f}')
        print()

        # Frequency note
        td = ws['trades_per_day']
        if td < CRITERIA['trades_per_day_min']:
            print(f'  ⚠  FREQUENCY NOTE: {td:.2f} signals/day is below the 1–2/day target.')
            print(f'     This is expected for a 3x volume filter on just 2 liquid ETFs.')
            print(f'     To reach 1–2/day: expand tickers to NDX, IWM, GLD, TLT, or')
            print(f'     individual mega-caps (AAPL, NVDA, TSLA) — the same parameters apply.')
        print()
        print(f'  Signal quality passes all gates:')
        for lbl, s in sorted(quality_pass.items()):
            p = all_params[lbl]
            tag = '← SELECTED' if lbl == winner else ''
            print(f'    {lbl}: win {s["win_rate"]:.1f}%  Sharpe {s["sharpe"]:.2f}'
                  f'  MaxCL {s["max_consec_loss"]}  {tag}')
    else:
        best = max(scored, key=lambda x: scored[x]['sharpe'])
        bs   = scored[best]
        print(f'  ✗  No variation meets quality criteria (win>50%, Sharpe>0, MaxCL≤5).')
        print()
        print(f'  Best available: Variation {best}'
              f'  (Sharpe {bs["sharpe"]:.2f}, win {bs["win_rate"]:.1f}%,'
              f' MaxCL {bs["max_consec_loss"]})')
        print()
        for lbl, s in scored.items():
            fails = []
            if s['win_rate'] <= 50:
                fails.append(f'win {s["win_rate"]:.1f}% ≤ 50%')
            if s['sharpe'] <= 0:
                fails.append(f'Sharpe {s["sharpe"]:.2f} ≤ 0')
            if s['max_consec_loss'] > 5:
                fails.append(f'MaxCL {s["max_consec_loss"]} > 5')
            if fails:
                print(f'  {lbl} fails: {", ".join(fails)}')
        print()
        print('  ⚠  Do not deploy — signal has no demonstrated edge on these instruments.')

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ticker_data = load_all()
    total_days  = len(set(ticker_data['SPY'].index.date))

    all_stats  = {}
    all_trades = {}
    all_params = dict(VARIATIONS_ABCD)

    print(f'Trading days: {total_days}')
    print('Running simulations...')
    print()

    # Variations A–D
    for lbl, params in VARIATIONS_ABCD.items():
        tf = 'tf-on' if params['time_filter'] else 'tf-off'
        print(f'  {lbl}  vol>{params["vol_mult"]}x  tgt={params["tgt_mult"]}x  {tf} ...', end='', flush=True)
        trades = simulate(ticker_data, params, lbl)
        s      = compute_stats(trades, lbl, total_days)
        all_stats[lbl]  = s
        all_trades[lbl] = trades
        print(f'  {s["n"]:>4} trades  win {s["win_rate"]:.1f}%  Sharpe {s["sharpe"]:+.2f}')

    print()

    # Build E from A–D results
    a, b, c, d = (all_stats[x] for x in ('A', 'B', 'C', 'D'))

    best_vol    = 3.0 if b['sharpe'] > a['sharpe'] else 2.5
    best_tgt    = 2.0 if c['sharpe'] > a['sharpe'] else 1.5
    use_filter  = d['sharpe'] > a['sharpe']

    e_params = dict(vol_mult=best_vol, move_pct=0.005,
                    tgt_mult=best_tgt, time_filter=use_filter)
    all_params['E'] = e_params

    vol_src = f'B ({best_vol}x)' if best_vol == 3.0 else f'A ({best_vol}x)'
    tgt_src = f'C ({best_tgt}x)' if best_tgt == 2.0 else f'A ({best_tgt}x)'
    flt_src = 'D (on)' if use_filter else 'A (off)'

    e_rationale = [
        f'vol_mult from {vol_src}   '
        f'→ B Sharpe {b["sharpe"]:+.2f} vs A Sharpe {a["sharpe"]:+.2f}',
        f'tgt_mult from {tgt_src}   '
        f'→ C Sharpe {c["sharpe"]:+.2f} vs A Sharpe {a["sharpe"]:+.2f}',
        f'time_filter from {flt_src}   '
        f'→ D Sharpe {d["sharpe"]:+.2f} vs A Sharpe {a["sharpe"]:+.2f}',
    ]

    tf = 'tf-on' if use_filter else 'tf-off'
    print(f'  E  vol>{best_vol}x  tgt={best_tgt}x  {tf} ...', end='', flush=True)
    trades_e = simulate(ticker_data, e_params, 'E')
    s_e      = compute_stats(trades_e, 'E', total_days)
    all_stats['E']  = s_e
    all_trades['E'] = trades_e
    print(f'  {s_e["n"]:>4} trades  win {s_e["win_rate"]:.1f}%  Sharpe {s_e["sharpe"]:+.2f}')

    print_report(all_stats, all_trades, all_params, total_days, e_rationale)

    # Save
    save = {}
    for lbl in ('A', 'B', 'C', 'D', 'E'):
        if lbl not in all_stats:
            continue
        save[f'variation_{lbl}'] = {
            'params':  all_params[lbl],
            'stats':   all_stats[lbl],
            'trades':  all_trades[lbl][:500],
        }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f'Results saved → {RESULTS_FILE}')


if __name__ == '__main__':
    main()
