#!/usr/bin/env python3
"""
orb_portfolio.py — capital-constrained sim of the ORB-quality screen WITH risk management.

Reality check on top of orb_backtest.py: you can't buy 100 names. So cap at N slots (default 10),
size equal-weight, and manage each trade with a STOP and a low PROFIT TARGET (default 2%).

  screen (close[T], no look-ahead): liquid, up>=min_oc% open->close, price>30d&180d SMA, +7d/+30d/+180d
  rank the favorable names, take the top (slots - open positions)
  ENTER at open[T+1]; each day check the daily bar:  High>=target -> take profit ; Low<=stop -> stopped
     (if BOTH breach same day -> assume STOP first, the conservative call) ; else hold to max_hold, exit close
  costs: per-side % on entry and exit.

Daily-bar fills are approximate (no intraday sequence) — treat magnitudes as indicative, the SHAPE
(win rate, expectancy, drawdown, vs SPY) as the signal. Data pull is cached for fast param sweeps.

Usage:  python research/orb_portfolio.py --slots 10 --stop 0.04 --target 0.02 --max_hold 5 --rank pct_oc
"""
import sys, os, csv, argparse, pickle, statistics as st
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
DATA = 'https://data.alpaca.markets'
PAPER = 'https://paper-api.alpaca.markets'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'datalake')
MAJ = {'NYSE', 'NASDAQ', 'ARCA', 'AMEX', 'BATS'}


def _universe():
    a = requests.get(f'{PAPER}/v2/assets', headers=H,
                     params={'status': 'active', 'asset_class': 'us_equity'}, timeout=90).json()
    return sorted({x['symbol'] for x in a if x.get('tradable') and x.get('exchange') in MAJ
                   and x.get('symbol') and '.' not in x['symbol'] and '/' not in x['symbol']
                   and len(x['symbol']) <= 5})


def _bars(symbols, start, end, chunk=300):
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


def load_data(months, top_liquid):
    """Universe rank + window pull + per-stock arrays. Cached to datalake for fast re-runs."""
    end = (datetime.now(timezone.utc).date() - timedelta(days=1))
    ck = os.path.join(CACHE, f'orb_cache_{end}_{months}m_{top_liquid}.pkl')
    if os.path.exists(ck):
        print(f"[cache] {os.path.basename(ck)}")
        return pickle.load(open(ck, 'rb'))
    print(f"[pull] ranking liquidity ...", flush=True)
    U = _universe()
    rank = _bars(U, (end - timedelta(days=45)).isoformat(), end.isoformat())
    dvol = {s: st.median([x['c'] * x['v'] for x in b]) for s, b in rank.items() if len(b) >= 10}
    liq = [s for s, _ in sorted(dvol.items(), key=lambda kv: kv[1], reverse=True)[:top_liquid]]
    warm = (end - timedelta(days=int((months + 10) * 30.5))).isoformat()
    print(f"[pull] {months}mo+warmup for {len(liq)} names ...", flush=True)
    px = _bars(sorted(set(liq) | {'SPY'}), warm, end.isoformat())
    S = {}
    for s, b in px.items():
        seen = {}
        for x in sorted(b, key=lambda x: x['t']):
            seen[x['t'][:10]] = x                 # dedupe: one bar per date (defensive)
        b = [seen[d] for d in sorted(seen)]
        c = [x['c'] for x in b]
        ps = [0.0]
        for x in c:
            ps.append(ps[-1] + x)
        S[s] = dict(dt=[x['t'][:10] for x in b], idx={x['t'][:10]: i for i, x in enumerate(b)},
                    o=[x['o'] for x in b], h=[x['h'] for x in b], l=[x['l'] for x in b], c=c,
                    v=[x['v'] for x in b], ps=ps)
    start = (end - timedelta(days=int(months * 30.5))).isoformat()
    cal = [d for d in S['SPY']['dt'] if d >= start]
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump((S, cal), open(ck, 'wb'))
    return S, cal


def favorable(S, T, min_oc, min_price, LIQ):
    """Ranked list of (sym, pct_oc, dvol) passing the tight screen at close[T]."""
    out = []
    for s, d in S.items():
        if s == 'SPY':
            continue
        i = d['idx'].get(T)
        if i is None or i < 181:
            continue
        c_i, o_i, v_i = d['c'][i], d['o'][i], d['v'][i]
        if o_i <= 0 or c_i < min_price or c_i * v_i < LIQ:
            continue
        oc = c_i / o_i - 1
        if oc < min_oc / 100:
            continue
        sma30 = (d['ps'][i + 1] - d['ps'][i + 1 - 30]) / 30
        sma180 = (d['ps'][i + 1] - d['ps'][i + 1 - 180]) / 180
        if c_i > sma30 and c_i > sma180 and c_i / d['c'][i - 7] - 1 > 0 \
           and c_i / d['c'][i - 30] - 1 > 0 and c_i / d['c'][i - 180] - 1 > 0:
            out.append((s, oc, c_i * v_i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--top_liquid', type=int, default=1500)
    ap.add_argument('--slots', type=int, default=10)
    ap.add_argument('--capital', type=float, default=10000)
    ap.add_argument('--stop', type=float, default=0.04, help='stop loss (0.04 = 4%)')
    ap.add_argument('--target', type=float, default=0.02, help='profit target (0.02 = 2%)')
    ap.add_argument('--max_hold', type=int, default=5, help='days before time-stop exit at close')
    ap.add_argument('--cost', type=float, default=0.1, help='per-side cost %%')
    ap.add_argument('--rank', choices=['pct_oc', 'dvol'], default='pct_oc', help='which names to pick for the slots')
    ap.add_argument('--min_oc', type=float, default=0.5)
    ap.add_argument('--min_price', type=float, default=5.0)
    ap.add_argument('--min_dvol', type=float, default=20.0)
    a = ap.parse_args()
    LIQ = a.min_dvol * 1e6
    cost = a.cost / 100
    S, cal = load_data(a.months, a.top_liquid)

    cash = a.capital
    pos = []            # {sym, entry, shares, held}
    pending = []        # syms scheduled to buy at next open
    trades = []
    eq_curve = []
    for T in cal:
        # 1) fill pending entries at open[T]
        newpos = []
        for s in pending:
            d = S[s]; i = d['idx'].get(T)
            if i is None:
                continue
            entry = d['o'][i]
            if entry <= 0:
                continue
            budget = (cash + sum(p['shares'] * S[p['sym']]['c'][S[p['sym']]['idx'][T]] for p in pos
                                 if T in S[p['sym']]['idx'])) / a.slots
            budget = min(budget, cash)
            sh = int(budget // (entry * (1 + cost)))
            if sh <= 0:
                continue
            cash -= sh * entry * (1 + cost)
            newpos.append(dict(sym=s, entry=entry, shares=sh, held=0))
        pos += newpos
        pending = []
        # 2) manage exits on day T (target / stop / time-stop)
        still = []
        for p in pos:
            d = S[p['sym']]; i = d['idx'].get(T)
            if i is None:
                still.append(p); continue
            hi, lo, c = d['h'][i], d['l'][i], d['c'][i]
            tgt, stp = p['entry'] * (1 + a.target), p['entry'] * (1 - a.stop)
            exit_px = reason = None
            if lo <= stp and hi >= tgt:
                exit_px, reason = stp, 'stop(both)'          # conservative: stop first
            elif hi >= tgt:
                exit_px, reason = tgt, 'target'
            elif lo <= stp:
                exit_px, reason = stp, 'stop'
            elif p['held'] >= a.max_hold:
                exit_px, reason = c, 'time'
            if exit_px is None:
                p['held'] += 1; still.append(p); continue
            proceeds = p['shares'] * exit_px * (1 - cost)
            cash += proceeds
            trades.append(dict(sym=p['sym'], entry_date_i=i, entry=p['entry'], exit=exit_px,
                               ret=(exit_px / p['entry'] - 1) * 100, reason=reason, held=p['held']))
        pos = still
        # 3) screen at close[T] -> schedule entries for tomorrow
        free = a.slots - len(pos)
        if free > 0:
            fav = favorable(S, T, a.min_oc, a.min_price, LIQ)
            fav.sort(key=lambda x: x[1] if a.rank == 'pct_oc' else x[2], reverse=True)
            held = {p['sym'] for p in pos}
            pending = [s for s, _, _ in fav if s not in held][:free]
        # equity mark-to-market
        mv = sum(p['shares'] * S[p['sym']]['c'][S[p['sym']]['idx'][T]] for p in pos if T in S[p['sym']]['idx'])
        eq_curve.append((T, cash + mv))

    # liquidate remainder at last close
    for p in pos:
        d = S[p['sym']]; c = d['c'][d['idx'][cal[-1]]]
        cash += p['shares'] * c * (1 - cost)
        trades.append(dict(sym=p['sym'], entry_date_i=-1, entry=p['entry'], exit=c,
                           ret=(c / p['entry'] - 1) * 100, reason='end', held=p['held']))
    final = eq_curve[-1][1] if eq_curve else a.capital

    # ---- report ----
    rets = [t['ret'] for t in trades]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    eq = [v for _, v in eq_curve]
    peak = eq[0]; mdd = 0
    for v in eq:
        peak = max(peak, v); mdd = min(mdd, v / peak - 1)
    spy = S['SPY']; s0, s1 = spy['c'][spy['idx'][cal[0]]], spy['c'][spy['idx'][cal[-1]]]
    yrs = len(cal) / 252
    print("\n" + "=" * 70)
    print(f"  ORB PORTFOLIO — {cal[0]} -> {cal[-1]} | {a.slots} slots | stop {a.stop*100:.0f}% "
          f"target {a.target*100:.0f}% hold {a.max_hold}d | rank {a.rank}")
    print("=" * 70)
    print(f"  capital  ${a.capital:,.0f} -> ${final:,.0f}   ({(final/a.capital-1)*100:+.1f}% total, "
          f"{((final/a.capital)**(1/yrs)-1)*100:+.1f}%/yr)   maxDD {mdd*100:.1f}%")
    print(f"  SPY buy-hold same window: {(s1/s0-1)*100:+.1f}%")
    print(f"  trades {len(trades)} | win rate {len(wins)/max(len(trades),1)*100:.1f}% | "
          f"avg win {st.mean(wins) if wins else 0:+.2f}% | avg loss {st.mean(losses) if losses else 0:+.2f}% | "
          f"expectancy {st.mean(rets) if rets else 0:+.3f}%/trade")
    exits = defaultdict(int)
    for t in trades:
        exits[t['reason']] += 1
    print(f"  exits: " + ' · '.join(f"{k} {v}" for k, v in sorted(exits.items(), key=lambda x: -x[1])))
    bym = defaultdict(float)
    m0 = a.capital
    for (T, v), (_, pv) in zip(eq_curve, [(None, a.capital)] + eq_curve[:-1]):
        bym[T[:7]] += v - pv
    print("  by-month P&L ($):")
    for m in sorted(bym):
        print(f"    {m}  {bym[m]:+8.0f}")
    f = os.path.join(HERE, f'orb_portfolio_trades_{cal[0]}_{cal[-1]}.csv')
    with open(f, 'w', newline='') as fh:
        if trades:
            w = csv.DictWriter(fh, fieldnames=list(trades[0].keys())); w.writeheader(); w.writerows(trades)
    print(f"  trades -> {os.path.basename(f)}")


if __name__ == '__main__':
    main()
