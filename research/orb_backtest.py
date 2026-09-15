#!/usr/bin/env python3
"""
orb_backtest.py — does the ORB-momentum-QUALITY screen find profitable targets CONSISTENTLY?

The screen (built at CLOSE of day T, no look-ahead):
  liquid  : close >= $min_price AND dollar-volume >= $min_dvol   (the control universe)
  up      : liquid AND up >= min_oc% open->close on day T        (raw "up after the open")
  fav     : up AND price>30d&180d SMA AND ret7>0 AND ret30>0 AND ret180>0   (TIGHT quality gate)

Trade model: select at close[T], BUY open[T+1], SELL close[T+1] (a 1-day ORB-style hold). Forward
return is open[T+1]->close[T+1]. (5-day variant also reported.)

The honest test is NOT "do picks go up" (everything goes up in an up market) — it's do picks beat the
AVERAGE LIQUID STOCK that same day (breadth control). That EDGE, and whether it's positive CONSISTENTLY
month over month, is the whole question. Costs noted separately.

Usage:  python research/orb_backtest.py --months 12 --top_liquid 1500
"""
import sys, os, csv, argparse, statistics as st
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
DATA = 'https://data.alpaca.markets'
PAPER = 'https://paper-api.alpaca.markets'
HERE = os.path.dirname(os.path.abspath(__file__))
MAJ = {'NYSE', 'NASDAQ', 'ARCA', 'AMEX', 'BATS'}


def universe():
    a = requests.get(f'{PAPER}/v2/assets', headers=H,
                     params={'status': 'active', 'asset_class': 'us_equity'}, timeout=90).json()
    return sorted({x['symbol'] for x in a if x.get('tradable') and x.get('exchange') in MAJ
                   and x.get('symbol') and '.' not in x['symbol'] and '/' not in x['symbol']
                   and len(x['symbol']) <= 5})


def bars(symbols, start, end, chunk=300):
    out = {}
    for i in range(0, len(symbols), chunk):
        grp = symbols[i:i + chunk]; tok = None
        while True:
            p = {'symbols': ','.join(grp), 'timeframe': '1Day', 'start': start, 'end': end,
                 'feed': 'sip', 'adjustment': 'all', 'limit': 10000}
            if tok:
                p['page_token'] = tok
            j = requests.get(f'{DATA}/v2/stocks/bars', headers=H, params=p, timeout=120).json()
            for s, b in j.get('bars', {}).items():
                out.setdefault(s, []).extend(b)
            tok = j.get('next_page_token')
            if not tok:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=12, help='backtest window length')
    ap.add_argument('--top_liquid', type=int, default=1500, help='universe = top-N by recent $volume')
    ap.add_argument('--min_oc', type=float, default=0.5)
    ap.add_argument('--min_price', type=float, default=5.0)
    ap.add_argument('--min_dvol', type=float, default=20.0, help='$M/day')
    ap.add_argument('--cost', type=float, default=0.15, help='round-trip cost %% (slippage+fees)')
    a = ap.parse_args()
    LIQ = a.min_dvol * 1e6
    today = datetime.now(timezone.utc).date()
    end = today - timedelta(days=1)                                  # free SIP: no recent data
    warmup_start = end - timedelta(days=int((a.months + 10) * 30.5))  # window + 180d warmup buffer

    # ---- 1. liquidity-ranked universe (recent 30 sessions) ----
    print(f"[1] ranking liquidity of full market ...", flush=True)
    U = universe()
    rank = bars(U, (end - timedelta(days=45)).isoformat(), end.isoformat())
    dvol = {s: st.median([x['c'] * x['v'] for x in b]) for s, b in rank.items() if len(b) >= 10}
    liquid_univ = [s for s, _ in sorted(dvol.items(), key=lambda kv: kv[1], reverse=True)[:a.top_liquid]]
    print(f"    {len(U)} eligible -> top {len(liquid_univ)} by median $vol (>= ${dvol[liquid_univ[-1]]/1e6:.0f}M)")

    # ---- 2. pull window + warmup for the universe + SPY ----
    print(f"[2] pulling {a.months}mo + 180d warmup for {len(liquid_univ)} names ...", flush=True)
    px = bars(sorted(set(liquid_univ) | {'SPY'}), warmup_start.isoformat(), end.isoformat())

    # per-stock aligned arrays + prefix sums for fast SMA
    S = {}
    for s, b in px.items():
        seen = {}
        for x in sorted(b, key=lambda x: x['t']):
            seen[x['t'][:10]] = x                 # dedupe: one bar per date (defensive)
        b = [seen[d] for d in sorted(seen)]
        dt = [x['t'][:10] for x in b]
        c = [x['c'] for x in b]; o = [x['o'] for x in b]; v = [x['v'] for x in b]
        ps = [0.0]
        for x in c:
            ps.append(ps[-1] + x)
        S[s] = dict(dt=dt, idx={d: i for i, d in enumerate(dt)}, o=o, c=c, v=v, ps=ps)

    cal = [d for d in S['SPY']['dt'] if d >= (end - timedelta(days=int(a.months * 30.5))).isoformat()]
    spy = S['SPY']

    def fwd(s, i, hold=1):
        d = S[s]; j = i + 1
        if j >= len(d['c']):
            return None
        sell = d['c'][min(i + hold, len(d['c']) - 1)]
        return sell / d['o'][j] - 1

    # ---- 3. simulate ----
    print(f"[3] simulating {len(cal)} sessions ...", flush=True)
    daily = []   # per day: date, n_fav, fav_f, up_f, ctrl_f, spy_f (+ 5d), hit counts
    fav_hits = [0, 0]; fav_rets = []
    for T in cal:
        ctrl, up, fav = [], [], []
        for s, d in S.items():
            if s == 'SPY':
                continue
            i = d['idx'].get(T)
            if i is None or i < 181:
                continue
            c_i, o_i, v_i = d['c'][i], d['o'][i], d['v'][i]
            if c_i < a.min_price or c_i * v_i < LIQ or o_i <= 0:
                continue
            f = fwd(s, i, 1); f5 = fwd(s, i, 5)
            if f is None:
                continue
            ctrl.append((f, f5))
            oc = c_i / o_i - 1
            if oc < a.min_oc / 100:
                continue
            up.append((f, f5))
            sma30 = (d['ps'][i + 1] - d['ps'][i + 1 - 30]) / 30
            sma180 = (d['ps'][i + 1] - d['ps'][i + 1 - 180]) / 180
            r7, r30, r180 = c_i / d['c'][i - 7] - 1, c_i / d['c'][i - 30] - 1, c_i / d['c'][i - 180] - 1
            if c_i > sma30 and c_i > sma180 and r7 > 0 and r30 > 0 and r180 > 0:
                fav.append((f, f5))
                fav_rets.append(f)
                fav_hits[0] += 1 if f > 0 else 0; fav_hits[1] += 1
        if not ctrl:
            continue
        si = spy['idx'].get(T)
        spy_f = (spy['c'][si + 1] / spy['o'][si + 1] - 1) if (si is not None and si + 1 < len(spy['c'])) else 0
        mean = lambda xs, k: (st.mean(x[k] for x in xs) if xs else None)
        daily.append(dict(date=T, n_fav=len(fav), n_up=len(up), n_ctrl=len(ctrl),
                          fav_f=mean(fav, 0), up_f=mean(up, 0), ctrl_f=mean(ctrl, 0),
                          fav_f5=mean(fav, 1), ctrl_f5=mean(ctrl, 1), spy_f=spy_f))

    # ---- 4. aggregate ----
    dd = [r for r in daily if r['fav_f'] is not None]
    def agg(key_p, key_c):
        edges = [r[key_p] - r[key_c] for r in dd]
        return st.mean(edges) * 100, sum(1 for e in edges if e > 0) / len(edges) * 100
    fav_edge, fav_days_pos = agg('fav_f', 'ctrl_f')
    up_edge, up_days_pos = agg('up_f', 'ctrl_f')
    avg_size = st.mean(r['n_fav'] for r in daily)
    fav_abs = st.mean(r['fav_f'] for r in dd) * 100
    ctrl_abs = st.mean(r['ctrl_f'] for r in dd) * 100
    fav_abs5 = st.mean(r['fav_f5'] for r in dd) * 100
    ctrl_abs5 = st.mean(r['ctrl_f5'] for r in dd) * 100

    print("\n" + "=" * 74)
    print(f"  ORB-QUALITY SCREEN BACKTEST — {cal[0]} -> {cal[-1]}  ({len(dd)} sessions)")
    print("=" * 74)
    print(f"  avg FAV list size / day : {avg_size:.1f}   (up-day set avg {st.mean(r['n_up'] for r in daily):.0f}, "
          f"liquid universe avg {st.mean(r['n_ctrl'] for r in daily):.0f})")
    print(f"\n  NEXT-DAY (open->close) mean return:")
    print(f"    FAV picks     {fav_abs:+.3f}%   (after {a.cost}% cost: {fav_abs - a.cost:+.3f}%)")
    print(f"    liquid ctrl   {ctrl_abs:+.3f}%   <- breadth benchmark (avg liquid stock)")
    print(f"    SPY           {st.mean(r['spy_f'] for r in dd)*100:+.3f}%")
    print(f"    >> FAV EDGE vs control: {fav_edge:+.3f}%/day  ({fav_days_pos:.0f}% of days positive)")
    print(f"    (raw up-day edge vs control: {up_edge:+.3f}%/day, {up_days_pos:.0f}% of days — does quality filter add?)")
    print(f"  FAV per-pick hit rate (next-day up): {fav_hits[0]/max(fav_hits[1],1)*100:.1f}%  (n={fav_hits[1]})")
    print(f"\n  5-DAY hold (open[T+1]->close[T+5]):  FAV {fav_abs5:+.2f}%  vs ctrl {ctrl_abs5:+.2f}%  "
          f"edge {fav_abs5-ctrl_abs5:+.2f}%")

    # by month consistency
    bym = defaultdict(list)
    for r in dd:
        bym[r['date'][:7]].append(r['fav_f'] - r['ctrl_f'])
    print(f"\n  BY MONTH — FAV edge vs control (the consistency test):")
    pos = 0
    for m in sorted(bym):
        e = st.mean(bym[m]) * 100; pos += e > 0
        print(f"    {m}  {e:+.3f}%/day  {'+' if e>0 else '-'}{'#'*int(min(abs(e)*20,30))}")
    print(f"    -> {pos}/{len(bym)} months with positive edge")

    f = os.path.join(HERE, f'orb_backtest_daily_{cal[0]}_{cal[-1]}.csv')
    with open(f, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(daily[0].keys())); w.writeheader(); w.writerows(daily)
    print(f"\n  per-day detail -> {os.path.basename(f)}")


if __name__ == '__main__':
    main()
