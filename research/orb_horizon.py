#!/usr/bin/env python3
"""
orb_horizon.py — does the intraday "up after the open" TRIGGER add anything, and over what HOLD?

Reuses orb_portfolio's cached data. Each session, builds three sets and measures forward returns
(BUY open[T+1], SELL close[T+H]) at several horizons H, vs the liquid-breadth control:

  CONTROL : all liquid names (price>=$5, $vol>=$20M)           <- beta / breadth benchmark
  QUALITY : control AND price>30d&180d SMA AND +7d/+30d/+180d   <- pure quality-momentum, NO trigger
  FAV     : QUALITY AND up>=0.5% open->close on day T           <- QUALITY + the ORB intraday trigger

The three comparisons that matter:
  QUALITY vs CONTROL  -> is the quality-momentum screen itself an edge? (this ~ the validated sleeve)
  FAV vs QUALITY      -> does the intraday trigger ADD edge, or just cut the list & add noise?
  FAV vs CONTROL      -> the full screen's edge

Usage:  python research/orb_horizon.py
"""
import sys, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orb_portfolio import load_data

HOR = [1, 5, 10, 20, 40]
MIN_OC, MIN_PRICE, LIQ = 0.5, 5.0, 20e6


def sets_for_day(S, T):
    ctrl, qual, fav = [], [], []
    for s, d in S.items():
        if s == 'SPY':
            continue
        i = d['idx'].get(T)
        if i is None or i < 181 or i + max(HOR) >= len(d['c']):
            continue
        c_i, o_i, v_i = d['c'][i], d['o'][i], d['v'][i]
        if o_i <= 0 or c_i < MIN_PRICE or c_i * v_i < LIQ:
            continue
        fwd = {H: d['c'][i + H] / d['o'][i + 1] - 1 for H in HOR}   # buy next open, sell close[T+H]
        ctrl.append(fwd)
        sma30 = (d['ps'][i + 1] - d['ps'][i + 1 - 30]) / 30
        sma180 = (d['ps'][i + 1] - d['ps'][i + 1 - 180]) / 180
        isqual = (c_i > sma30 and c_i > sma180 and c_i / d['c'][i - 7] - 1 > 0
                  and c_i / d['c'][i - 30] - 1 > 0 and c_i / d['c'][i - 180] - 1 > 0)
        if isqual:
            qual.append(fwd)
            if c_i / o_i - 1 >= MIN_OC / 100:
                fav.append(fwd)
    return ctrl, qual, fav


def main():
    S, cal = load_data(12, 1500)
    acc = {k: {H: [] for H in HOR} for k in ('ctrl', 'qual', 'fav')}
    sizes = {'ctrl': [], 'qual': [], 'fav': []}
    for T in cal:
        ctrl, qual, fav = sets_for_day(S, T)
        if not ctrl:
            continue
        sizes['ctrl'].append(len(ctrl)); sizes['qual'].append(len(qual)); sizes['fav'].append(len(fav))
        for name, grp in (('ctrl', ctrl), ('qual', qual), ('fav', fav)):
            for H in HOR:
                if grp:
                    acc[name][H].append(st.mean(g[H] for g in grp))

    def m(name, H):
        return st.mean(acc[name][H]) * 100 if acc[name][H] else float('nan')

    print("\n" + "=" * 70)
    print(f"  FORWARD RETURN BY HOLD HORIZON — {cal[0]} -> {cal[-1]}")
    print(f"  avg/day sizes: control {st.mean(sizes['ctrl']):.0f} | quality {st.mean(sizes['qual']):.0f} "
          f"| fav {st.mean(sizes['fav']):.0f}")
    print("=" * 70)
    print(f"  {'hold':>6} {'CONTROL':>9} {'QUALITY':>9} {'FAV':>9} | {'QUAL-CTRL':>10} {'FAV-QUAL':>9} {'FAV-CTRL':>9}")
    for H in HOR:
        print(f"  {H:>4}d  {m('ctrl',H):>+8.2f}% {m('qual',H):>+8.2f}% {m('fav',H):>+8.2f}% | "
              f"{m('qual',H)-m('ctrl',H):>+9.2f}% {m('fav',H)-m('qual',H):>+8.2f}% {m('fav',H)-m('ctrl',H):>+8.2f}%")
    print("\n  read: QUAL-CTRL>0 = quality-momentum screen has edge (≈ the validated sleeve).")
    print("        FAV-QUAL>0 = the intraday trigger genuinely ADDS; <=0 = it's just noise+fewer names.")


if __name__ == '__main__':
    main()
