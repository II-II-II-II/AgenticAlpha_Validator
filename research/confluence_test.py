#!/usr/bin/env python3
"""
confluence_test.py — ABLATION test of the Reddit 5-indicator "confluence" strategy.

PRE-REGISTERED DESIGN (written before any result was seen):

  Question:   does stacking 5 indicators beat stacking 1 or 2 — and does ANY of it beat
              simply holding the same stock?
  Universe:   the 166 1-min tickers in ALPACA_DEEP_BACKTEST (2025-09 .. 2026-06).
              NOTE: this watchlist is HINDSIGHT-SELECTED (the 2025-26 hype names appear
              on all 196 days), which would flatter any momentum strategy.
  Control:    that bias is neutralised by scoring every strategy against BUY-AND-HOLD OF
              THE SAME TICKER over the same window. Selection bias hits both sides equally,
              so the only thing measured is whether the indicators ADD anything.
  Execution:  signals from CLOSED 5-min bars, shifted 1 bar, filled at the NEXT bar's OPEN.
              Long/flat only (retail momentum posts are long-biased).
  Costs:      swept 0 / 10 / 25 / 50 bps per side. Micro-cap spreads live at the top end.
  Stats:      per-ticker excess vs hold; median + % of tickers beating hold; ticker-clustered
              t-stat (each ticker one observation, so a couple of lucky names can't carry it).

  Kill rule:  if L5 (all five) does not beat L1/L2 AND the family does not beat hold at
              realistic cost, the "confluence" premise is dead regardless of parameters.

Usage:  python research/confluence_test.py [--dir ...] [--limit N]
"""
import os, sys, glob, argparse, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confluence as C

warnings.filterwarnings("ignore")
DATA = os.environ.get('CANDLE_DIR',
                      os.path.expanduser('~/Desktop/Alpha/TradingAgents/review/ALPACA_DEEP_BACKTEST'))
COSTS_BPS = [0, 10, 25, 50]


def run_one(df, members, params, cost_bps):
    """Long/flat on unanimous agreement of `members`. Returns (strat_ret, hold_ret, ntrades)."""
    sig = C.signals(df, params)
    want = np.ones(len(df), dtype=bool)
    for m in members:
        want &= np.nan_to_num(sig[m]).astype(bool)

    # act on the NEXT bar's open: shift the decision one bar forward
    pos = pd.Series(want).shift(1).fillna(False).to_numpy()
    o = df["Open"].to_numpy()
    # bar return measured open->open (that is what a next-open fill actually earns)
    nxt = np.roll(o, -1)
    r = np.zeros(len(o))
    r[:-1] = nxt[:-1] / o[:-1] - 1.0
    r = np.nan_to_num(r)

    trades = np.abs(np.diff(pos.astype(int), prepend=0))
    cost = trades * (cost_bps / 10000.0)
    strat = np.prod(1.0 + r * pos - cost) - 1.0
    hold = o[-1] / o[0] - 1.0
    return strat, hold, int(trades.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DATA)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_1m.csv")))
    if args.limit:
        files = files[:args.limit]
    print(f"loading {len(files)} tickers from {os.path.basename(args.dir)} ...")

    frames = []
    for f in files:
        try:
            d = C.load_5m(f)
        except Exception:
            d = None
        if d is not None and len(d) > 300:
            frames.append((os.path.basename(f).split("_")[0], d))
    print(f"usable: {len(frames)} tickers | median bars {int(np.median([len(d) for _, d in frames]))}\n")

    rows = []
    for cost in COSTS_BPS:
        for name, members in C.LADDER:
            ex, wins, tr, sr, hr = [], 0, [], [], []
            for tkr, d in frames:
                s, h, n = run_one(d, members, C.DEFAULTS, cost)
                ex.append(s - h); sr.append(s); hr.append(h); tr.append(n)
                wins += (s > h)
            ex = np.array(ex)
            t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 2 and ex.std() > 0 else 0.0
            rows.append(dict(cost=cost, level=name, n=len(ex),
                             med_strat=np.median(sr) * 100, med_hold=np.median(hr) * 100,
                             med_excess=np.median(ex) * 100, mean_excess=ex.mean() * 100,
                             pct_beat_hold=100.0 * wins / len(ex),
                             t_clustered=t, med_trades=np.median(tr)))

    out = pd.DataFrame(rows)
    print("=" * 104)
    print("ABLATION — strategy vs BUY-AND-HOLD OF THE SAME TICKER (per-ticker, ticker-clustered)")
    print("=" * 104)
    for cost in COSTS_BPS:
        sub = out[out.cost == cost]
        print(f"\n--- cost {cost} bps/side ---")
        print(f"{'level':<18}{'medStrat%':>11}{'medHold%':>10}{'medExcess%':>12}"
              f"{'meanExcess%':>13}{'%beatHold':>11}{'t':>8}{'trades':>9}")
        for _, r in sub.iterrows():
            print(f"{r.level:<18}{r.med_strat:>11.1f}{r.med_hold:>10.1f}{r.med_excess:>12.1f}"
                  f"{r.mean_excess:>13.1f}{r.pct_beat_hold:>11.1f}{r.t_clustered:>8.2f}{r.med_trades:>9.0f}")

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confluence_ablation.csv")
    out.to_csv(p, index=False)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
