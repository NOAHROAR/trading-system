#!/usr/bin/env python3
"""
backtest_iron_condor.py
7DTE SPY iron condor vs put-spread-only baseline.

Two condor exit modes tested:
  Mode A — combined stop: close the entire condor when EITHER leg reaches
            200% of its collected credit, OR the combined position hits 50%
            profit target.
  Mode B — independent legs: each leg manages its own 200% stop independently;
            after one leg closes, the surviving leg runs to 50% of remaining
            credit or expiry.

Both condor modes share: same entry filters as the put spread, weekly loss
limit $1000, max 2 concurrent condors, force-close at expiry (daily backtest
approximates the 9:45 ET close using the day's closing price).

Data / pricing methodology mirrors backtest_credit_spread_v2.py exactly:
  sigma = VIX * 0.85, IVR = 252-day VIX percentile.

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
TARGET_DELTA      = 0.20
DELTA_TOL         = 0.05
SPREAD_WIDTH      = 5.0
MIN_CREDIT_LEG    = 0.25   # minimum credit per individual leg
PROFIT_TARGET     = 0.50   # 50% of collected credit
STOP_LOSS         = 2.00   # 200% of leg credit
MIN_IVR           = 30.0
MAX_VIX           = 35.0
SMA_PERIOD        = 20
RISK_FREE         = 0.045
VIX_WINDOW        = 252
MAX_POSITIONS     = 2
WEEKLY_LOSS_LIMIT = 1000.0

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


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_put_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float(max(K - S, 0))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_call_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float(max(S - K, 0))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def find_short_put_strike(S, T, sigma):
    """Strike where put |delta| ≈ TARGET_DELTA. Returns (K, delta) or (None, None)."""
    if sigma <= 0 or T <= 0 or S <= 0:
        return None, None
    d1_target = norm.ppf(1.0 - TARGET_DELTA)   # N^{-1}(0.80) ≈ +0.842
    log_SK    = d1_target * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma**2) * T
    K         = round(S * math.exp(-log_SK))
    if K <= 0:
        return None, None
    d1    = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta = norm.cdf(d1) - 1.0                  # put delta (negative)
    if abs(abs(delta) - TARGET_DELTA) > DELTA_TOL:
        return None, None
    return float(K), float(delta)


def find_short_call_strike(S, T, sigma):
    """Strike where call delta ≈ TARGET_DELTA. Returns (K, delta) or (None, None)."""
    if sigma <= 0 or T <= 0 or S <= 0:
        return None, None
    d1_target = norm.ppf(TARGET_DELTA)           # N^{-1}(0.20) ≈ -0.842
    log_SK    = d1_target * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma**2) * T
    K         = round(S * math.exp(-log_SK))     # K > S since log_SK < 0
    if K <= S:
        return None, None
    d1    = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    delta = norm.cdf(d1)                          # call delta (positive)
    if abs(delta - TARGET_DELTA) > DELTA_TOL:
        return None, None
    return float(K), float(delta)


def put_spread_credit(S, short_K, long_K, T, sigma):
    """Net credit for bear put spread (short high-K put, long low-K put)."""
    return round(bs_put_price(S, short_K, T, sigma) - bs_put_price(S, long_K, T, sigma), 4)


def call_spread_credit(S, short_K, long_K, T, sigma):
    """Net credit for bear call spread (short low-K call, long high-K call)."""
    return round(bs_call_price(S, short_K, T, sigma) - bs_call_price(S, long_K, T, sigma), 4)


# ── Data loading (mirrors backtest_credit_spread_v2.py exactly) ───────────────

def load_data() -> pd.DataFrame:
    print('Loading SPY, ^VIX from yfinance (2019-07-01 → 2026-07-05)...')
    raw = yf.download(
        ['SPY', '^VIX'],
        start='2019-07-01', end='2026-07-05',
        auto_adjust=True, progress=True,
    )

    close = raw['Close'].copy()
    close.columns = [str(c) for c in close.columns]

    spy = close['SPY'].ffill()
    vix = close['^VIX'].ffill()

    sma_spy = spy.rolling(SMA_PERIOD).mean()

    vix_arr = vix.values.astype(float)
    n       = len(vix_arr)
    ivr_arr = np.full(n, np.nan)
    for i in range(VIX_WINDOW, n):
        w           = vix_arr[i - VIX_WINDOW:i]
        ivr_arr[i]  = float((w < vix_arr[i]).sum()) / VIX_WINDOW * 100.0
    ivr = pd.Series(ivr_arr, index=vix.index)

    df = pd.DataFrame({'spy': spy, 'vix': vix, 'sma_spy': sma_spy, 'ivr': ivr})
    df = df[(df.index >= '2020-01-01') & (df.index < '2026-07-05')].dropna()
    print(f'  {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})')
    return df


# ── Position classes ──────────────────────────────────────────────────────────

class PutSpread:
    __slots__ = ('entry_date', 'expiry', 'short_K', 'long_K',
                 'credit', 'profit_tgt', 'stop_cost')

    def __init__(self, entry_date, expiry, short_K, long_K, credit):
        self.entry_date = entry_date
        self.expiry     = expiry
        self.short_K    = short_K
        self.long_K     = long_K
        self.credit     = credit
        self.profit_tgt = round(credit * PROFIT_TARGET, 4)
        self.stop_cost  = round(credit * STOP_LOSS, 4)

    def cost_to_close(self, S, T, sigma):
        return put_spread_credit(S, self.short_K, self.long_K, T, sigma)


class IronCondor:
    __slots__ = (
        'entry_date', 'expiry',
        'short_put',  'long_put',  'put_credit',
        'short_call', 'long_call', 'call_credit',
        'total_credit', 'profit_tgt',
        'put_stop_cost', 'call_stop_cost',
        # Mode B independent-leg state
        'put_closed', 'call_closed',
        'put_close_cost', 'call_close_cost',
        'put_exit', 'call_exit',
    )

    def __init__(self, entry_date, expiry,
                 short_put, long_put, put_credit,
                 short_call, long_call, call_credit):
        self.entry_date    = entry_date
        self.expiry        = expiry
        self.short_put     = short_put
        self.long_put      = long_put
        self.put_credit    = put_credit
        self.short_call    = short_call
        self.long_call     = long_call
        self.call_credit   = call_credit
        self.total_credit  = round(put_credit + call_credit, 4)
        self.profit_tgt    = round(self.total_credit * PROFIT_TARGET, 4)
        self.put_stop_cost = round(put_credit  * STOP_LOSS, 4)
        self.call_stop_cost = round(call_credit * STOP_LOSS, 4)
        # Mode B state
        self.put_closed      = False
        self.call_closed     = False
        self.put_close_cost  = 0.0
        self.call_close_cost = 0.0
        self.put_exit        = None
        self.call_exit       = None

    def put_cost(self, S, T, sigma):
        return put_spread_credit(S, self.short_put, self.long_put, T, sigma)

    def call_cost(self, S, T, sigma):
        return call_spread_credit(S, self.short_call, self.long_call, T, sigma)

    def put_intrinsic(self, S):
        return max(self.short_put - S, 0) - max(self.long_put - S, 0)

    def call_intrinsic(self, S):
        return max(S - self.short_call, 0) - max(S - self.long_call, 0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def next_expiry(d: date) -> date:
    best, best_diff = None, 999
    for delta in range(4, 13):
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


def _current_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


# ── Backtests ─────────────────────────────────────────────────────────────────

def run_put_spread(df: pd.DataFrame) -> dict:
    """Baseline: SPY put credit spread only, with weekly loss limit."""
    trades    = []
    open_pos  = []
    daily_pnl = []
    equity    = 0.0
    peak      = 0.0
    max_dd    = 0.0

    week_start      = None
    weekly_realized = 0.0

    blocked = {k: 0 for k in (
        'macro', 'sma', 'ivr', 'vix', 'min_credit',
        'position_limit', 'underwater', 'delta_miss', 'weekly_loss',
    )}

    rows = df.to_dict('records')
    idx  = [d.date() for d in df.index]

    for i, d in enumerate(idx):
        row   = rows[i]
        sigma = max(float(row['vix']) / 100.0 * 0.85, 0.05)

        # Reset weekly loss counter on new Monday
        mon = _current_monday(d)
        if week_start != mon:
            week_start      = mon
            weekly_realized = 0.0

        # ── Monitor + close ──────────────────────────────────────────────────
        day_pnl = 0.0
        keep    = []
        for pos in open_pos:
            T    = max((pos.expiry - d).days / 365.0, 0.0)
            cost = pos.cost_to_close(float(row['spy']), T, sigma)

            if d >= pos.expiry:
                cost   = max(pos.short_K - float(row['spy']), 0) - max(pos.long_K - float(row['spy']), 0)
                reason = 'EXPIRY'
            elif cost <= pos.profit_tgt:
                reason = 'PROFIT'
            elif cost >= pos.stop_cost:
                reason = 'STOP'
            else:
                keep.append(pos)
                continue

            pnl              = round((pos.credit - cost) * 100, 2)
            day_pnl         += pnl
            weekly_realized += pnl
            trades.append(dict(
                entry=pos.entry_date, exit=d, expiry=pos.expiry,
                short_K=pos.short_K, long_K=pos.long_K,
                credit=pos.credit, close_cost=round(cost, 4),
                pnl=pnl, reason=reason,
                held=(d - pos.entry_date).days, year=d.year,
            ))

        open_pos = keep

        # ── Entry ────────────────────────────────────────────────────────────
        if d.weekday() < 5:
            S       = float(row['spy'])
            sma_val = float(row['sma_spy'])
            vix_val = float(row['vix'])
            ivr_val = float(row['ivr'])

            if len(open_pos) >= MAX_POSITIONS:
                blocked['position_limit'] += 1
            elif is_macro_day(d):
                blocked['macro'] += 1
            elif S <= sma_val:
                blocked['sma'] += 1
            elif ivr_val < MIN_IVR:
                blocked['ivr'] += 1
            elif vix_val >= MAX_VIX:
                blocked['vix'] += 1
            elif weekly_realized <= -WEEKLY_LOSS_LIMIT:
                blocked['weekly_loss'] += 1
            else:
                # Underwater gate
                underwater = any(
                    pos.cost_to_close(S, max((pos.expiry - d).days / 365.0, 0.0), sigma) > pos.credit
                    for pos in open_pos
                )
                if underwater:
                    blocked['underwater'] += 1
                else:
                    expiry = next_expiry(d)
                    if not any(pos.expiry == expiry for pos in open_pos):
                        T       = max((expiry - d).days / 365.0, 1 / 365)
                        short_K, _ = find_short_put_strike(S, T, sigma)
                        if short_K is None:
                            blocked['delta_miss'] += 1
                        else:
                            long_K = short_K - SPREAD_WIDTH
                            credit = put_spread_credit(S, short_K, long_K, T, sigma)
                            if credit < MIN_CREDIT_LEG:
                                blocked['min_credit'] += 1
                            else:
                                open_pos.append(PutSpread(d, expiry, short_K, long_K, credit))

        # ── Daily equity ─────────────────────────────────────────────────────
        S     = float(row['spy'])
        unrealized = sum(
            (pos.credit - pos.cost_to_close(S, max((pos.expiry - d).days / 365.0, 0.0), sigma)) * 100
            for pos in open_pos
        )
        equity  += day_pnl
        total_eq = equity + unrealized
        peak     = max(peak, total_eq)
        max_dd   = max(max_dd, peak - total_eq)
        daily_pnl.append(day_pnl)

    # Force-close remaining
    last_row, last_d = rows[-1], idx[-1]
    sigma = max(float(last_row['vix']) / 100.0 * 0.85, 0.05)
    for pos in open_pos:
        S    = float(last_row['spy'])
        T    = max((pos.expiry - last_d).days / 365.0, 0.0)
        cost = round(pos.cost_to_close(S, T, sigma), 4)
        pnl  = round((pos.credit - cost) * 100, 2)
        trades.append(dict(
            entry=pos.entry_date, exit=last_d, expiry=pos.expiry,
            short_K=pos.short_K, long_K=pos.long_K,
            credit=pos.credit, close_cost=cost,
            pnl=pnl, reason='OPEN_AT_END',
            held=(last_d - pos.entry_date).days, year=last_d.year,
        ))

    return dict(trades=trades, blocked=blocked, daily_pnl=daily_pnl,
                max_dd=max_dd, dates=idx)


def run_condor(df: pd.DataFrame, independent_legs: bool = False) -> dict:
    """
    Iron condor backtest.
    independent_legs=False  → Mode A: whole condor stops when either leg hits 200%.
    independent_legs=True   → Mode B: each leg stops independently at 200%;
                               remaining leg targets 50% of its own credit.
    """
    trades    = []
    open_pos  = []
    daily_pnl = []
    equity    = 0.0
    peak      = 0.0
    max_dd    = 0.0

    week_start      = None
    weekly_realized = 0.0

    blocked = {k: 0 for k in (
        'macro', 'sma', 'ivr', 'vix', 'min_credit',
        'position_limit', 'underwater', 'delta_miss', 'weekly_loss',
    )}

    rows = df.to_dict('records')
    idx  = [d.date() for d in df.index]

    for i, d in enumerate(idx):
        row   = rows[i]
        S     = float(row['spy'])
        sigma = max(float(row['vix']) / 100.0 * 0.85, 0.05)

        mon = _current_monday(d)
        if week_start != mon:
            week_start      = mon
            weekly_realized = 0.0

        # ── Monitor + close condors ──────────────────────────────────────────
        day_pnl = 0.0
        keep    = []

        for pos in open_pos:
            T        = max((pos.expiry - d).days / 365.0, 0.0)
            fully_closed = False

            if not independent_legs:
                # ── Mode A: combined stop ────────────────────────────────────
                pc = pos.put_cost(S, T, sigma)
                cc = pos.call_cost(S, T, sigma)

                if d >= pos.expiry:
                    pc     = pos.put_intrinsic(S)
                    cc     = pos.call_intrinsic(S)
                    reason = 'EXPIRY'
                elif (pc + cc) <= pos.profit_tgt:
                    reason = 'PROFIT'
                elif pc >= pos.put_stop_cost:
                    reason = 'PUT_STOP'
                elif cc >= pos.call_stop_cost:
                    reason = 'CALL_STOP'
                else:
                    keep.append(pos)
                    continue

                pnl          = round((pos.total_credit - pc - cc) * 100, 2)
                day_pnl     += pnl
                weekly_realized += pnl
                trades.append(dict(
                    entry=pos.entry_date, exit=d, expiry=pos.expiry,
                    short_put=pos.short_put, short_call=pos.short_call,
                    put_credit=pos.put_credit, call_credit=pos.call_credit,
                    total_credit=pos.total_credit,
                    put_close=round(pc, 4), call_close=round(cc, 4),
                    pnl=pnl, reason=reason,
                    held=(d - pos.entry_date).days, year=d.year,
                ))

            else:
                # ── Mode B: independent legs ─────────────────────────────────
                pc = pos.put_cost(S, T, sigma)  if not pos.put_closed  else pos.put_close_cost
                cc = pos.call_cost(S, T, sigma) if not pos.call_closed else pos.call_close_cost

                if d >= pos.expiry:
                    if not pos.put_closed:
                        pos.put_close_cost = round(pos.put_intrinsic(S), 4)
                        pos.put_exit       = 'EXPIRY'
                        pos.put_closed     = True
                    if not pos.call_closed:
                        pos.call_close_cost = round(pos.call_intrinsic(S), 4)
                        pos.call_exit       = 'EXPIRY'
                        pos.call_closed     = True
                    fully_closed = True

                else:
                    # Individual leg stop checks
                    if not pos.put_closed and pc >= pos.put_stop_cost:
                        pos.put_close_cost = round(pc, 4)
                        pos.put_exit       = 'STOP'
                        pos.put_closed     = True

                    if not pos.call_closed and cc >= pos.call_stop_cost:
                        pos.call_close_cost = round(cc, 4)
                        pos.call_exit       = 'STOP'
                        pos.call_closed     = True

                    # Profit check on surviving leg(s)
                    rem_credit = ((pos.put_credit  if not pos.put_closed  else 0.0) +
                                  (pos.call_credit if not pos.call_closed else 0.0))
                    rem_cost   = ((pc if not pos.put_closed  else 0.0) +
                                  (cc if not pos.call_closed else 0.0))

                    if rem_credit > 0 and rem_cost <= rem_credit * PROFIT_TARGET:
                        if not pos.put_closed:
                            pos.put_close_cost = round(pc, 4)
                            pos.put_exit       = 'PROFIT'
                            pos.put_closed     = True
                        if not pos.call_closed:
                            pos.call_close_cost = round(cc, 4)
                            pos.call_exit       = 'PROFIT'
                            pos.call_closed     = True

                    fully_closed = pos.put_closed and pos.call_closed

                if fully_closed:
                    pnl    = round((pos.put_credit  - pos.put_close_cost +
                                    pos.call_credit - pos.call_close_cost) * 100, 2)
                    reason = f'{pos.put_exit}+{pos.call_exit}'
                    day_pnl         += pnl
                    weekly_realized += pnl
                    trades.append(dict(
                        entry=pos.entry_date, exit=d, expiry=pos.expiry,
                        short_put=pos.short_put, short_call=pos.short_call,
                        put_credit=pos.put_credit, call_credit=pos.call_credit,
                        total_credit=pos.total_credit,
                        put_close=pos.put_close_cost, call_close=pos.call_close_cost,
                        pnl=pnl, reason=reason,
                        held=(d - pos.entry_date).days, year=d.year,
                    ))
                else:
                    keep.append(pos)
                    continue

        open_pos = keep

        # ── Entry ────────────────────────────────────────────────────────────
        if d.weekday() < 5:
            sma_val = float(row['sma_spy'])
            vix_val = float(row['vix'])
            ivr_val = float(row['ivr'])

            if len(open_pos) >= MAX_POSITIONS:
                blocked['position_limit'] += 1
            elif is_macro_day(d):
                blocked['macro'] += 1
            elif S <= sma_val:
                blocked['sma'] += 1
            elif ivr_val < MIN_IVR:
                blocked['ivr'] += 1
            elif vix_val >= MAX_VIX:
                blocked['vix'] += 1
            elif weekly_realized <= -WEEKLY_LOSS_LIMIT:
                blocked['weekly_loss'] += 1
            else:
                # Underwater gate: any alive put leg currently above entry credit
                underwater = False
                for pos in open_pos:
                    pT = max((pos.expiry - d).days / 365.0, 0.0)
                    if not pos.put_closed and pos.put_cost(S, pT, sigma) > pos.put_credit:
                        underwater = True
                        break
                    if not pos.call_closed and pos.call_cost(S, pT, sigma) > pos.call_credit:
                        underwater = True
                        break

                if underwater:
                    blocked['underwater'] += 1
                else:
                    expiry = next_expiry(d)
                    if not any(pos.expiry == expiry for pos in open_pos):
                        T = max((expiry - d).days / 365.0, 1 / 365)

                        short_put, _  = find_short_put_strike(S, T, sigma)
                        short_call, _ = find_short_call_strike(S, T, sigma)

                        if short_put is None or short_call is None:
                            blocked['delta_miss'] += 1
                        else:
                            long_put  = short_put  - SPREAD_WIDTH
                            long_call = short_call + SPREAD_WIDTH

                            pc = put_spread_credit(S, short_put, long_put, T, sigma)
                            cc = call_spread_credit(S, short_call, long_call, T, sigma)

                            if pc < MIN_CREDIT_LEG or cc < MIN_CREDIT_LEG:
                                blocked['min_credit'] += 1
                            else:
                                open_pos.append(IronCondor(
                                    d, expiry,
                                    short_put, long_put, pc,
                                    short_call, long_call, cc,
                                ))

        # ── Daily equity ─────────────────────────────────────────────────────
        unrealized = 0.0
        for pos in open_pos:
            T = max((pos.expiry - d).days / 365.0, 0.0)
            if not pos.put_closed:
                unrealized += (pos.put_credit  - pos.put_cost(S, T, sigma))  * 100
            if not pos.call_closed:
                unrealized += (pos.call_credit - pos.call_cost(S, T, sigma)) * 100

        equity  += day_pnl
        total_eq = equity + unrealized
        peak     = max(peak, total_eq)
        max_dd   = max(max_dd, peak - total_eq)
        daily_pnl.append(day_pnl)

    # Force-close any remaining condors
    last_row, last_d = rows[-1], idx[-1]
    sigma = max(float(last_row['vix']) / 100.0 * 0.85, 0.05)
    S     = float(last_row['spy'])
    for pos in open_pos:
        T  = max((pos.expiry - last_d).days / 365.0, 0.0)
        pc = round(pos.put_cost(S, T, sigma)  if not pos.put_closed  else pos.put_close_cost, 4)
        cc = round(pos.call_cost(S, T, sigma) if not pos.call_closed else pos.call_close_cost, 4)
        pnl = round((pos.put_credit - pc + pos.call_credit - cc) * 100, 2)
        reason = (f'{pos.put_exit or "OPEN"}+{pos.call_exit or "OPEN"}'
                  if independent_legs else 'OPEN_AT_END')
        trades.append(dict(
            entry=pos.entry_date, exit=last_d, expiry=pos.expiry,
            short_put=pos.short_put, short_call=pos.short_call,
            put_credit=pos.put_credit, call_credit=pos.call_credit,
            total_credit=pos.total_credit,
            put_close=pc, call_close=cc,
            pnl=pnl, reason=reason,
            held=(last_d - pos.entry_date).days, year=last_d.year,
        ))

    return dict(trades=trades, blocked=blocked, daily_pnl=daily_pnl,
                max_dd=max_dd, dates=idx)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _metrics(res: dict) -> dict:
    df_t = pd.DataFrame(res['trades'])
    if df_t.empty:
        return dict(n=0, wr=0.0, pnl=0.0, dd=res['max_dd'], sharpe=0.0, stop_pct=0.0)
    wins     = (df_t['pnl'] > 0).sum()
    wr       = wins / len(df_t) * 100
    total    = df_t['pnl'].sum()
    dpnl     = pd.Series(res['daily_pnl'], dtype=float)
    sharpe   = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    # stop rate: any trade whose reason contains STOP
    stop_pct = df_t['reason'].str.contains('STOP').sum() / len(df_t) * 100
    return dict(n=len(df_t), wr=wr, pnl=total, dd=res['max_dd'],
                sharpe=sharpe, stop_pct=stop_pct)


def print_stats(res: dict, label: str, is_condor: bool = False):
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
    dpnl       = pd.Series(daily_pnl, dtype=float)
    sharpe     = (dpnl.mean() / dpnl.std() * math.sqrt(252)) if dpnl.std() > 0 else 0.0
    stop_mask  = df_t['reason'].str.contains('STOP')
    stop_pct   = stop_mask.sum() / len(df_t) * 100

    if is_condor:
        avg_credit = df_t['total_credit'].mean()
        avg_put_cr = df_t['put_credit'].mean()
        avg_cal_cr = df_t['call_credit'].mean()
    else:
        avg_credit = df_t['credit'].mean()

    yearly = df_t.groupby('year')['pnl'].agg(pnl='sum', trades='count')

    W = 62
    print(f'\n{"="*W}')
    print(f'  {label}')
    print(f'{"="*W}')
    print(f'  Total trades:    {len(df_t):<6}  (wins {len(wins)}  /  losses {len(loss)})')
    print(f'  Win rate:        {win_rate:.1f}%')
    print(f'  Stop-loss rate:  {stop_pct:.1f}%')
    print(f'  Total P&L:       ${total_pnl:>10,.2f}')
    print(f'  Avg win:         ${avg_win:>10.2f}')
    print(f'  Avg loss:        ${avg_loss:>10.2f}')
    if is_condor:
        print(f'  Avg total credit:${avg_credit:>10.4f}  '
              f'(put ${avg_put_cr:.4f}  call ${avg_cal_cr:.4f})')
    else:
        print(f'  Avg credit:      ${avg_credit:>10.4f}')
    print(f'  Sharpe ratio:    {sharpe:>10.2f}')
    print(f'  Max drawdown:    ${max_dd:>10,.2f}')

    print(f'\n  {"Year":<6} {"P&L":>10} {"Trades":>8}')
    print(f'  {"-"*24}')
    for yr, row in yearly.iterrows():
        print(f'  {yr:<6} ${row["pnl"]:>9,.2f} {int(row["trades"]):>8}')

    best_yr  = yearly['pnl'].idxmax()
    worst_yr = yearly['pnl'].idxmin()
    print(f'\n  Best year:  {best_yr}  (${yearly.loc[best_yr,  "pnl"]:,.2f})')
    print(f'  Worst year: {worst_yr}  (${yearly.loc[worst_yr, "pnl"]:,.2f})')

    print(f'\n  Exit reasons:')
    for reason, cnt in df_t['reason'].value_counts().items():
        avg = df_t[df_t['reason'] == reason]['pnl'].mean()
        print(f'    {reason:<26} {cnt:>4} ({cnt/len(df_t)*100:4.1f}%)  avg ${avg:+.2f}')

    total_blocked = sum(blocked.values())
    if total_blocked:
        print(f'\n  Entry filter blocks (total {total_blocked:,}):')
        for k, v in blocked.items():
            if v:
                print(f'    {k:<16} {v:>6,}  ({v/total_blocked*100:4.1f}%)')


def print_period_sharpe(res: dict, label: str, n_periods: int = 4):
    """Split the backtest into n equal calendar periods; report Sharpe per slice."""
    dates     = res['dates']
    daily_pnl = res['daily_pnl']
    trades_df = pd.DataFrame(res['trades']) if res['trades'] else pd.DataFrame()
    n_days    = len(dates)
    chunk     = n_days // n_periods

    print(f'\n  {label} — Sharpe by sub-period:')
    print(f'  {"Period":<25}  {"Days":>4}  {"Trades":>6}  {"Stop%":>6}  '
          f'{"P&L":>10}  {"Sharpe":>7}')
    print(f'  {"-"*68}')

    for p in range(n_periods):
        start   = p * chunk
        end     = (p + 1) * chunk if p < n_periods - 1 else n_days
        d_start = dates[start]
        d_end   = dates[end - 1]

        sl       = pd.Series(daily_pnl[start:end], dtype=float)
        pnl_p    = sl.sum()
        sharpe_p = (sl.mean() / sl.std() * math.sqrt(252)) if sl.std() > 0 else 0.0

        if not trades_df.empty and 'exit' in trades_df.columns:
            mask   = (trades_df['exit'] >= d_start) & (trades_df['exit'] <= d_end)
            sub    = trades_df[mask]
            n_t    = len(sub)
            stop_p = sub['reason'].str.contains('STOP').sum() / n_t * 100 if n_t > 0 else 0.0
        else:
            n_t, stop_p = 0, 0.0

        period_str = f'{d_start} to {d_end}'
        print(f'  {period_str:<25}  {end-start:>4}  {n_t:>6}  '
              f'{stop_p:>5.1f}%  ${pnl_p:>9,.2f}  {sharpe_p:>7.2f}')


def print_comparison(res_ps: dict, res_ca: dict, res_cb: dict):
    """Side-by-side: put spread | condor Mode A | condor Mode B."""
    mps = _metrics(res_ps)
    mca = _metrics(res_ca)
    mcb = _metrics(res_cb)

    # Period-by-period Sharpe for all three
    def period_sharpes(res, n=4):
        daily_pnl = res['daily_pnl']
        n_days = len(daily_pnl)
        chunk  = n_days // n
        out = []
        for p in range(n):
            start = p * chunk
            end   = (p + 1) * chunk if p < n - 1 else n_days
            sl    = pd.Series(daily_pnl[start:end], dtype=float)
            sh    = (sl.mean() / sl.std() * math.sqrt(252)) if sl.std() > 0 else 0.0
            out.append(sh)
        return out

    sh_ps = period_sharpes(res_ps)
    sh_ca = period_sharpes(res_ca)
    sh_cb = period_sharpes(res_cb)

    dates     = res_ps['dates']
    n_days    = len(dates)
    chunk     = n_days // 4

    W = 76
    print(f'\n{"="*W}')
    print(f'  COMPARISON: Put Spread  vs  Condor Mode A  vs  Condor Mode B')
    print(f'{"="*W}')
    print(f'  {"Metric":<24} {"Put Spread":>14} {"Condor A":>14} {"Condor B":>14}')
    print(f'  {"-"*66}')
    print(f'  {"Total trades":<24} {mps["n"]:>14} {mca["n"]:>14} {mcb["n"]:>14}')
    print(f'  {"Win rate":<24} {mps["wr"]:>13.1f}% {mca["wr"]:>13.1f}% {mcb["wr"]:>13.1f}%')
    print(f'  {"Stop rate":<24} {mps["stop_pct"]:>13.1f}% {mca["stop_pct"]:>13.1f}% {mcb["stop_pct"]:>13.1f}%')
    print(f'  {"Total P&L":<24} ${mps["pnl"]:>13,.2f} ${mca["pnl"]:>13,.2f} ${mcb["pnl"]:>13,.2f}')
    print(f'  {"Max drawdown":<24} ${mps["dd"]:>13,.2f} ${mca["dd"]:>13,.2f} ${mcb["dd"]:>13,.2f}')
    print(f'  {"Sharpe (full)":<24} {mps["sharpe"]:>14.2f} {mca["sharpe"]:>14.2f} {mcb["sharpe"]:>14.2f}')
    print(f'\n  {"vs Put Spread:":<24}')
    print(f'  {"P&L delta":<24} {"baseline":>14} ${mca["pnl"]-mps["pnl"]:>+13,.2f} ${mcb["pnl"]-mps["pnl"]:>+13,.2f}')
    print(f'  {"DD delta":<24} {"baseline":>14} ${mca["dd"]-mps["dd"]:>+13,.2f} ${mcb["dd"]-mps["dd"]:>+13,.2f}')
    print(f'  {"Sharpe delta":<24} {"baseline":>14} {mca["sharpe"]-mps["sharpe"]:>+14.2f} {mcb["sharpe"]-mps["sharpe"]:>+14.2f}')

    print(f'\n  {"Sharpe per period:":<24} {"Put Spread":>14} {"Condor A":>14} {"Condor B":>14}')
    print(f'  {"-"*66}')
    for p in range(4):
        start      = p * chunk
        end        = (p + 1) * chunk if p < 3 else n_days
        d_start    = dates[start]
        d_end      = dates[end - 1]
        period_str = f'P{p+1} {d_start.year}-{d_end.year}'
        print(f'  {period_str:<24} {sh_ps[p]:>14.2f} {sh_ca[p]:>14.2f} {sh_cb[p]:>14.2f}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    print('\nRunning put spread baseline (with weekly loss limit)...')
    res_ps = run_put_spread(df)

    print('Running iron condor — Mode A (combined stop)...')
    res_ca = run_condor(df, independent_legs=False)

    print('Running iron condor — Mode B (independent legs)...')
    res_cb = run_condor(df, independent_legs=True)

    print_stats(res_ps, 'SPY PUT SPREAD ONLY (baseline + weekly loss limit)')
    print_period_sharpe(res_ps, 'PUT SPREAD')

    print_stats(res_ca, 'IRON CONDOR — Mode A (combined stop)', is_condor=True)
    print_period_sharpe(res_ca, 'CONDOR MODE A')

    print_stats(res_cb, 'IRON CONDOR — Mode B (independent legs)', is_condor=True)
    print_period_sharpe(res_cb, 'CONDOR MODE B')

    print_comparison(res_ps, res_ca, res_cb)


if __name__ == '__main__':
    main()
