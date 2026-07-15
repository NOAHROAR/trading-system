#!/usr/bin/env python3
"""
weekly_replay.py — VS Variation B signal replay, Mon Jun 23 – Thu Jun 26 2026.

Standalone: no Alpaca, no Discord, no imports from scanner.py.
Pulls 15-min OHLCV from yfinance and reimplements VS signal logic inline.

Usage:
    python3 weekly_replay.py
"""

from datetime import date, time

import pandas as pd
import pytz
import yfinance as yf

# ── STRATEGY PARAMETERS — Variation B ────────────────────────────────────────
TICKERS       = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
VOL_MA_LEN    = 20       # bars in rolling vol average (excludes spike bar)
VOL_MULT      = 3.0      # spike volume must be >3.0× the 20-bar average
MOVE_PCT      = 0.005    # bar price move must be >0.5%  (|close-open|/open)
TGT_MULT      = 1.5      # target = entry ± 1.5× spike_move
MIN_STOP_DIST = 0.01     # skip signal if stop distance < $0.01

ET            = pytz.timezone('US/Eastern')
MARKET_OPEN   = time(9, 30)
MARKET_CLOSE  = time(16, 0)

# Replay window
REPLAY_START  = date(2026, 6, 23)   # Monday
REPLAY_END    = date(2026, 6, 26)   # Thursday (today)

# Pull one extra week of history so the 20-bar vol MA is valid on Day 1
FETCH_FROM    = date(2026, 6, 16)
FETCH_TO      = date(2026, 6, 27)   # yfinance end is exclusive

# Entry cutoff: scanner calls _can_enter()=False at or after 15:30 ET.
# Detection scan fires when spike bar *closes* (spike_bar_start + 15 min).
# → skip spikes whose bar starts at or after 15:15 ET (would be detected at 15:30).
ENTRY_CUTOFF  = time(15, 15)


# ── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_bars(ticker):
    """
    Download 15-min bars from yfinance for one ticker.
    Returns list of dicts with keys: time (ET Timestamp), open, high, low, close, volume.
    Filtered to regular market hours only.
    """
    df = yf.download(
        ticker,
        start=FETCH_FROM.isoformat(),
        end=FETCH_TO.isoformat(),
        interval='15m',
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return []

    # yfinance >=0.2 may return MultiIndex columns even for single-ticker downloads
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    bars = []
    for ts, row in df.iterrows():
        et = (ts.tz_localize('UTC') if ts.tzinfo is None else ts).tz_convert(ET)
        if not (MARKET_OPEN <= et.time() < MARKET_CLOSE):
            continue
        bars.append({
            'time':   et,
            'open':   float(row['Open']),
            'high':   float(row['High']),
            'low':    float(row['Low']),
            'close':  float(row['Close']),
            'volume': float(row['Volume']),
        })
    return bars


# ── VS SIGNAL DETECTION ───────────────────────────────────────────────────────

def detect_spike(bars, i):
    """
    Return spike metadata dict if bars[i] qualifies as a VS spike bar, else None.

    Mirrors scanner.py logic exactly:
      - vol_ratio = bars[i].volume / mean(bars[i-20 : i].volume)
      - vol_ratio > 3.0×  AND  |close-open|/open > 0.5%
    """
    if i < VOL_MA_LEN:
        return None

    bar    = bars[i]
    vol_ma = sum(bars[j]['volume'] for j in range(i - VOL_MA_LEN, i)) / VOL_MA_LEN
    if vol_ma == 0:
        return None

    vol_ratio = bar['volume'] / vol_ma
    if vol_ratio < VOL_MULT:
        return None

    o, c = bar['open'], bar['close']
    if o == 0:
        return None
    move = abs(c - o) / o
    if move < MOVE_PCT:
        return None

    return {
        'direction':  'LONG' if c > o else 'SHORT',
        'vol_ratio':  round(vol_ratio, 2),
        'spike_pct':  round(move * 100, 2),
        'spike_move': abs(c - o),   # used for target calculation
        'bar_high':   bar['high'],  # stop for SHORT
        'bar_low':    bar['low'],   # stop for LONG
    }


# ── OUTCOME CHECKING ──────────────────────────────────────────────────────────

def check_outcome(bars, conf_idx, direction, entry_px, stop_px, target_px, trade_date):
    """
    Walk bars[conf_idx+1:] to find the first target/stop hit or EOD close.

    Uses bar high/low to detect intra-bar hits. If both target and stop are hit
    in the same bar (ambiguous candle), conservatively calls it a LOSS (stop hit
    first — common when a candle wicks both ways).

    Returns (label, exit_price, pnl_pct).
    """
    for j in range(conf_idx + 1, len(bars)):
        b = bars[j]

        # Rolled into next session → EOD close at previous bar's close
        if b['time'].date() != trade_date:
            ep  = bars[j - 1]['close']
            pnl = _pnl(direction, entry_px, ep)
            return ('WIN (EOD)' if pnl >= 0 else 'LOSS (EOD)'), ep, pnl

        if direction == 'LONG':
            hit_tgt  = b['high'] >= target_px
            hit_stop = b['low']  <= stop_px
        else:
            hit_tgt  = b['low']  <= target_px
            hit_stop = b['high'] >= stop_px

        if hit_tgt and not hit_stop:
            return 'WIN',          target_px, _pnl(direction, entry_px, target_px)
        if hit_stop and not hit_tgt:
            return 'LOSS',         stop_px,   _pnl(direction, entry_px, stop_px)
        if hit_tgt and hit_stop:
            return 'LOSS (ambig)', stop_px,   _pnl(direction, entry_px, stop_px)

        # Last bar of this trading day — EOD close at bar's close
        is_last = (j == len(bars) - 1 or bars[j + 1]['time'].date() != trade_date)
        if is_last:
            ep  = b['close']
            pnl = _pnl(direction, entry_px, ep)
            return ('WIN (EOD)' if pnl >= 0 else 'LOSS (EOD)'), ep, pnl

    # Ran out of bars — position still open (script run during market hours)
    return 'OPEN', None, None


def _pnl(direction, entry, exit_px):
    mult = 1 if direction == 'LONG' else -1
    return round(mult * (exit_px - entry) / entry * 100, 2)


# ── REPLAY LOOP ───────────────────────────────────────────────────────────────

def run_replay():
    """
    Scan every bar in the replay window for every ticker.
    Returns (signals_list, cutoff_skip_count).

    Note: no position-limit enforcement (replay shows all signal opportunities).
    """
    signals     = []
    cutoff_skip = 0

    for ticker in TICKERS:
        print(f'  {ticker}…', end=' ', flush=True)
        bars = fetch_bars(ticker)
        print(f'{len(bars)} bars')
        if not bars:
            continue

        # Need at least VOL_MA_LEN bars before bar[i] and one bar after (conf)
        for i in range(VOL_MA_LEN, len(bars) - 1):
            bar = bars[i]
            d   = bar['time'].date()

            # Only flag signals in the replay window (history bars are just for vol MA)
            if not (REPLAY_START <= d <= REPLAY_END):
                continue

            spike = detect_spike(bars, i)
            if not spike:
                continue

            # Entry cutoff: spike at bars[i].time >= ENTRY_CUTOFF → skip
            if bar['time'].time() >= ENTRY_CUTOFF:
                cutoff_skip += 1
                signals.append(_make_row(ticker, bar, spike, skipped=True))
                continue

            conf = bars[i + 1]

            # Confirmation bar must be same day (spike on last bar of day → no conf)
            if conf['time'].date() != d:
                cutoff_skip += 1
                signals.append(_make_row(ticker, bar, spike, skipped=True))
                continue

            entry_px  = conf['close']
            direction = spike['direction']
            stop_px   = bar['low'] if direction == 'LONG' else bar['high']
            stop_dist = abs(entry_px - stop_px)

            if stop_dist < MIN_STOP_DIST:
                continue  # degenerate stop — skip silently

            target_px = (entry_px + TGT_MULT * spike['spike_move'] if direction == 'LONG'
                         else entry_px - TGT_MULT * spike['spike_move'])

            result, exit_px, pnl_pct = check_outcome(
                bars, i + 1, direction, entry_px, stop_px, target_px, d
            )
            signals.append(_make_row(ticker, bar, spike, entry_px, stop_px, target_px,
                                     result, pnl_pct))

    return signals, cutoff_skip


def _make_row(ticker, bar, spike, entry=None, stop=None, target=None,
              result='SKIP (cutoff)', pnl=None, skipped=False):
    t         = bar['time'].strftime('%a %m/%d %H:%M')
    has_entry = (entry is not None)
    return {
        'ticker':    ticker,
        'spike_bar': t,
        'dir':       spike['direction'],
        'vol':       f"{spike['vol_ratio']}×",
        'move':      f"{spike['spike_pct']}%",
        'entry':     f'${entry:.2f}'  if has_entry else '—',
        'stop':      f'${stop:.2f}'   if has_entry else '—',
        'target':    f'${target:.2f}' if has_entry else '—',
        'hit_tgt':   ('YES' if result.startswith('WIN')
                      else '—' if result in ('OPEN', 'SKIP (cutoff)')
                      else 'no') if has_entry else '—',
        'hit_stop':  ('YES' if result.startswith('LOSS')
                      else '—' if result in ('OPEN', 'SKIP (cutoff)')
                      else 'no') if has_entry else '—',
        'result':    result,
        'pnl':       f'{pnl:+.2f}%' if pnl is not None else '—',
    }


# ── TABLE OUTPUT ──────────────────────────────────────────────────────────────

_COLS = [
    ('ticker',    'Ticker',      6),
    ('spike_bar', 'Spike Bar',  14),
    ('dir',       'Dir',         5),
    ('vol',       'Vol×',        5),
    ('move',      'Move%',       6),
    ('entry',     'Entry',       8),
    ('stop',      'Stop',        8),
    ('target',    'Target',      8),
    ('hit_tgt',   'Hit Tgt?',    9),
    ('hit_stop',  'Hit Stop?',  10),
    ('result',    'Result',     13),
    ('pnl',       'P&L%',        6),
]


def print_table(signals):
    print()
    if not signals:
        print('  No signals detected in the replay window.')
        return

    widths = {k: max(mw, len(h), max(len(s[k]) for s in signals))
              for k, h, mw in _COLS}

    div = '  ' + '─┼─'.join('─' * widths[k] for k, *_ in _COLS)
    hdr = '  ' + ' │ '.join(h.ljust(widths[k]) for k, h, _ in _COLS)

    print(f'  VS Variation B — {REPLAY_START}  →  {REPLAY_END}')
    print(div)
    print(hdr)
    print(div)

    prev_day = None
    for s in signals:
        day = s['spike_bar'].split()[0]   # 'Mon', 'Tue', etc.
        if prev_day and day != prev_day:
            print(div)                    # visual separator between days
        prev_day = day
        row = ' │ '.join(s[k].ljust(widths[k]) for k, *_ in _COLS)
        print(f'  {row}')

    print(div)


def print_summary(signals, cutoff_skip):
    taken   = [s for s in signals if s['result'] != 'SKIP (cutoff)']
    wins    = [s for s in taken if s['result'].startswith('WIN')]
    losses  = [s for s in taken if s['result'].startswith('LOSS')]
    open_   = [s for s in taken if s['result'] == 'OPEN']
    decided = len(wins) + len(losses)
    rate    = (len(wins) / decided * 100) if decided else 0.0

    print()
    print('  ── SUMMARY ──────────────────────────────────────────────')
    print(f'  Period          : {REPLAY_START}  →  {REPLAY_END}')
    print(f'  Tickers         : {" · ".join(TICKERS)}')
    print(f'  Signals taken   : {len(taken)}')
    print(f'  Wins            : {len(wins)}')
    print(f'  Losses          : {len(losses)}')
    print(f'  Win rate        : {rate:.0f}%  ({len(wins)}W / {len(losses)}L)')
    print(f'  Still open      : {len(open_)}')
    print(f'  Missed (cutoff) : {cutoff_skip}')
    print()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print()
    print('  ╔══════════════════════════════════════════════════════╗')
    print('  ║    VS Variation B — Weekly Replay                   ║')
    print(f'  ║    {REPLAY_START}  →  {REPLAY_END}                          ║')
    print('  ╚══════════════════════════════════════════════════════╝')
    print()
    print(f'  Params : vol>{VOL_MULT}×  move>{MOVE_PCT*100:.1f}%  '
          f'target={TGT_MULT}×  MA={VOL_MA_LEN} bars')
    print(f'  Cutoff : no new entries from spikes at or after {ENTRY_CUTOFF} ET')
    print()
    print('  Fetching 15-min bars from yfinance…')

    signals, cutoff_skip = run_replay()
    print_table(signals)
    print_summary(signals, cutoff_skip)


if __name__ == '__main__':
    main()
