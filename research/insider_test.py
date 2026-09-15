#!/usr/bin/env python3
"""
insider_test.py — falsifiable event study: do small-cap insider BUYS predict forward returns?

PRE-REGISTERED HYPOTHESIS: tradeable small-caps with open-market insider buys (Form 4 code P) beat a
small-cap control (IWC micro-cap ETF) over 1/3/6 months, after realistic costs, consistently by year.

NO LOOK-AHEAD: signal = the FILING date (public knowledge); ENTER at the next session's OPEN; measure
forward to close[+h]. Prices from Alpaca SIP (adjusted) — NOT from the filing. Control = IWC over the
SAME dates (isolates selection from small-cap beta). SURVIVORSHIP: symbols with no forward price
(delisted / no data) are COUNTED as a "missing bucket" and also stress-tested at -50% to bound the bias.

PASS bar (set before results): positive after-cost edge vs control at 3mo, present at multiple horizons,
positive in a MAJORITY of years, real sample; bonus if edge strengthens with buy size (dose-response).

Usage:  python research/insider_test.py --start 2022q1 --end 2025q4 --min_value 25000
"""
import sys, os, csv, time, argparse, pickle, statistics as st
from collections import defaultdict, Counter
import requests
import insider
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphahconfig as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'datalake')
DATA = 'https://data.alpaca.markets'
Halp = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
HOR = [21, 63, 126]                                 # ~1 / 3 / 6 months
CONTROLS = ['IWC', 'IWM']                           # micro-cap + small-cap ETFs
# CEF / BDC / SPAC / muni-fund noise: their "insider buys" are buyback/DRIP mechanics, not informed bets
FUND_WORDS = ('FUND', 'MUNICIPAL', ' MUNI', 'CLOSED-END', ' MLP', ' ETF', 'ACQUISITION CORP',
              'TERM TRUST', 'INCOME TRUST', 'CAPITAL TRUST', 'INVESTMENT TRUST', 'DIVIDEND', 'YIELD ',
              'PORTFOLIO', 'BLACKROCK', 'NUVEEN', 'PIMCO', 'EATON VANCE', 'GABELLI', 'CLEARBRIDGE',
              'ABRDN', 'INVESCO', 'COHEN & STEERS', 'JOHN HANCOCK', 'TEMPLETON')


def _is_fund(name):
    n = (name or '').upper()
    return any(w in n for w in FUND_WORDS)


def aggregate(events, min_value):
    """One signal per (symbol, filing_date): sum $ bought, flag officer/director, drop fund-like issuers."""
    agg = defaultdict(lambda: dict(value=0.0, officer=False, name=''))
    for e in events:
        if _is_fund(e.get('name')):
            continue
        k = (e['symbol'], e['filing_date'])
        agg[k]['value'] += e['value']; agg[k]['name'] = e.get('name', '')
        rt = (e['role'] + ' ' + e['title']).lower()
        if any(w in rt for w in ('officer', 'ceo', 'cfo', 'chief', 'president', 'director')):
            agg[k]['officer'] = True
    return [dict(symbol=s, filing_date=d, value=v['value'], officer=v['officer'])
            for (s, d), v in agg.items() if v['value'] >= min_value]


def bars(symbols, start, end, chunk=80):
    """Rate-limit-robust Alpaca SIP daily pull (429 backoff + inter-request pacing). {sym:[bars]}."""
    out = {}; syms = sorted(set(symbols) | set(CONTROLS))
    for i in range(0, len(syms), chunk):
        grp = syms[i:i + chunk]; tok = None
        while True:
            p = {'symbols': ','.join(grp), 'timeframe': '1Day', 'start': start, 'end': end,
                 'feed': 'sip', 'adjustment': 'all', 'limit': 10000}
            if tok:
                p['page_token'] = tok
            j = None
            for k in range(7):
                r = requests.get(f'{DATA}/v2/stocks/bars', headers=Halp, params=p, timeout=120)
                if r.status_code == 200:
                    j = r.json(); break
                time.sleep(2 * (k + 1))          # 429 / transient backoff
            if j is None:
                break
            for s, b in j.get('bars', {}).items():
                out.setdefault(s, []).extend(b)
            tok = j.get('next_page_token')
            if not tok:
                break
        time.sleep(0.2)
        if (i // chunk) % 8 == 0:
            print(f"    pulled ~{min(i + chunk, len(syms))}/{len(syms)} symbols", flush=True)
    return out


def build_series(px):
    S = {}
    for s, b in px.items():
        seen = {}
        for x in sorted(b, key=lambda r: r['t']):
            seen[x['t'][:10]] = x
        b = [seen[d] for d in sorted(seen)]
        S[s] = dict(dt=[x['t'][:10] for x in b], idx={x['t'][:10]: i for i, x in enumerate(b)},
                    o=[x['o'] for x in b], c=[x['c'] for x in b], v=[x['v'] for x in b])
    return S


def _single(sym, start, end):
    """Reliable single-symbol pull (controls are critical — never let them ride a droppable chunk)."""
    out = []; tok = None
    for _ in range(300):
        p = {'timeframe': '1Day', 'start': start, 'end': end, 'feed': 'sip', 'adjustment': 'all', 'limit': 10000}
        if tok:
            p['page_token'] = tok
        r = requests.get(f'{DATA}/v2/stocks/{sym}/bars', headers=Halp, params=p, timeout=60)
        if r.status_code != 200:
            time.sleep(2); continue
        j = r.json(); out += j.get('bars', []); tok = j.get('next_page_token')
        if not tok:
            break
    return out


def _ensure_controls(S, start, end):
    changed = False
    for c in CONTROLS:
        if c not in S or not S[c].get('dt'):
            b = _single(c, start, end)
            if b:
                S[c] = build_series({c: b})[c]; changed = True
    return changed


def load_prices(symbols, start, end):
    ck = os.path.join(CACHE, f'insider_px_{start}_{end}.pkl')
    if os.path.exists(ck):
        print(f"[cache] {os.path.basename(ck)}")
        S = pickle.load(open(ck, 'rb'))
        if _ensure_controls(S, start, end):        # patch controls into an older cache
            pickle.dump(S, open(ck, 'wb')); print("    (patched missing controls into cache)")
        return S
    print(f"[pull] {len(symbols)} symbols {start}->{end} (rate-limit-robust) ...", flush=True)
    px = bars(symbols, start, end)
    S = build_series(px)
    _ensure_controls(S, start, end)
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(S, open(ck, 'wb'))
    return S


def ctrl_ret(S, sym, de, dx):
    """Control ETF return over the SAME calendar window open[de]->close[dx]."""
    d = S.get(sym)
    if not d or de not in d['idx'] or dx not in d['idx']:
        return None
    return d['c'][d['idx'][dx]] / d['o'][d['idx'][de]] - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022q1'); ap.add_argument('--end', default='2025q4')
    ap.add_argument('--min_value', type=float, default=25000)
    ap.add_argument('--min_price', type=float, default=2.0); ap.add_argument('--max_price', type=float, default=30.0)
    ap.add_argument('--min_dvol', type=float, default=1e5); ap.add_argument('--max_dvol', type=float, default=3e7)
    ap.add_argument('--cost', type=float, default=0.4, help='round-trip %% (small-cap slippage)')
    a = ap.parse_args()

    qs = insider.quarters(a.start, a.end)
    print(f"[1] loading insider buys {qs[0]}..{qs[-1]} (min ${a.min_value:,.0f}) ...", flush=True)
    ev = aggregate(insider.buys(qs, min_value=a.min_value / 10), a.min_value)  # loose pre-filter, exact after agg
    syms = sorted({e['symbol'] for e in ev})
    print(f"    {len(ev)} (symbol,day) buy signals across {len(syms)} tickers")
    px_start = f"{a.start[:4]}-01-01"
    S = load_prices(syms, px_start, '2026-07-22')

    # ---- event study ----
    res = {h: [] for h in HOR}                 # list of (event_ret, control_ret, edge, year, value, officer)
    missing = {h: 0 for h in HOR}
    kept = 0; tickers_kept = Counter()
    for e in ev:
        d = S.get(e['symbol'])
        if not d:
            for h in HOR:
                missing[h] += 1
            continue
        # entry = first session strictly after the filing date
        entry = next((i for i, dt in enumerate(d['dt']) if dt > e['filing_date']), None)
        if entry is None:
            for h in HOR:
                missing[h] += 1
            continue
        o0 = d['o'][entry]
        if o0 <= 0 or not (a.min_price <= o0 <= a.max_price):
            continue                            # small-cap price band
        dvol = d['c'][entry] * d['v'][entry]
        if not (a.min_dvol <= dvol <= a.max_dvol):
            continue                            # tradeable-but-small liquidity band
        kept += 1; tickers_kept[e['symbol']] += 1
        yr = e['filing_date'][:4]
        for h in HOR:
            if entry + h >= len(d['c']):
                missing[h] += 1; continue
            de, dx = d['dt'][entry], d['dt'][entry + h]
            er = d['c'][entry + h] / o0 - 1
            cr = ctrl_ret(S, 'IWC', de, dx)
            if cr is None:
                missing[h] += 1; continue
            res[h].append((er, cr, (er - a.cost / 100) - cr, yr, e['value'], e['officer'], e['symbol']))

    # ---- report ----
    print("\n" + "=" * 76)
    print(f"  INSIDER-BUY EVENT STUDY — small-caps ${a.min_price:.0f}-{a.max_price:.0f}, "
          f"buys >= ${a.min_value:,.0f} | cost {a.cost}% RT | control IWC")
    print("=" * 76)
    print(f"  kept {kept} tradeable-small-cap events across {len(tickers_kept)} tickers "
          f"(top: {tickers_kept.most_common(3)})")
    print(f"\n  {'hold':>5}{'n':>7}{'event':>9}{'IWC ctrl':>10}{'EDGE(net)':>11}{'hit%':>7}{'edge>0%':>9}{'missing':>9}")
    verdict = {}
    for h in HOR:
        r = res[h]
        if not r:
            continue
        er = st.mean(x[0] for x in r) * 100; cr = st.mean(x[1] for x in r) * 100
        edge = st.mean(x[2] for x in r) * 100
        hit = sum(1 for x in r if x[0] > 0) / len(r) * 100
        ep = sum(1 for x in r if x[2] > 0) / len(r) * 100
        verdict[h] = edge
        print(f"  {h:>4}d{len(r):>7}{er:>+8.2f}%{cr:>+9.2f}%{edge:>+10.2f}%{hit:>6.0f}%{ep:>8.0f}%{missing[h]:>9}")
    # ticker-clustered edge (neutralize concentration): mean within ticker, then across tickers
    if res[63]:
        byt = defaultdict(list)
        for x in res[63]:
            byt[x[6]].append(x[2])
        clustered = st.mean(st.mean(v) for v in byt.values()) * 100
        print(f"\n  TICKER-CLUSTERED 63d edge ({len(byt)} tickers weighted equally): {clustered:+.2f}%"
              f"   (event-level {st.mean(x[2] for x in res[63])*100:+.2f}%) — should agree if not concentration-driven")
    # by-year (3mo) consistency
    print(f"\n  BY YEAR (63d net edge vs IWC — the persistence test):")
    yr = defaultdict(list)
    for x in res[63]:
        yr[x[3]].append(x[2])
    pos = 0
    for y in sorted(yr):
        e = st.mean(yr[y]) * 100; pos += e > 0
        print(f"    {y}  n={len(yr[y]):>5}  edge {e:>+6.2f}%  {'+' if e>0 else '-'}")
    print(f"    -> {pos}/{len(yr)} years positive")
    # dose-response (3mo): does bigger buy => bigger edge?
    print(f"\n  DOSE-RESPONSE (63d net edge by buy size — real signal should strengthen):")
    for lab, lo, hi in [('$25-100k', 25e3, 1e5), ('$100-500k', 1e5, 5e5), ('$500k+', 5e5, 9e15)]:
        seg = [x[2] for x in res[63] if lo <= x[4] < hi]
        if seg:
            print(f"    {lab:<10} n={len(seg):>5}  edge {st.mean(seg)*100:>+6.2f}%")
    off = [x[2] for x in res[63] if x[5]]; non = [x[2] for x in res[63] if not x[5]]
    print(f"    officer/director involved: {st.mean(off)*100:+.2f}% (n={len(off)})  |  "
          f"10%%-owner/other: {st.mean(non)*100:+.2f}% (n={len(non)})" if off and non else "")
    # survivorship stress: assign -50% to the missing (delisted) bucket at 63d
    r63 = res[63]
    if r63:
        stressed = [x[2] for x in r63] + [(-0.50 - a.cost / 100) - 0.0] * missing[63]
        print(f"\n  SURVIVORSHIP STRESS (63d): assign -50%% to all {missing[63]} missing/delisted -> "
              f"edge {st.mean(stressed)*100:+.2f}% (base {verdict.get(63,0):+.2f}%)")
    print(f"\n  PASS? need: 63d edge>0 after cost, multiple horizons, majority years, survives stress.")
    f = os.path.join(HERE, 'insider_events.csv')
    with open(f, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['symbol', 'filing_date', 'value', 'officer'])
        for e in ev:
            w.writerow([e['symbol'], e['filing_date'], round(e['value']), e['officer']])
    print(f"  events -> {os.path.basename(f)}")


if __name__ == '__main__':
    main()
