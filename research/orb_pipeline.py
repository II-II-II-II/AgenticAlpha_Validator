#!/usr/bin/env python3
"""
orb_pipeline.py — ORB reframed: intraday-momentum names that are ALSO technically solid (not junk).

STEP 1  Pull yesterday's completed session. Keep liquid US equities (dollar-volume floor = junk filter)
        that finished UP >= MIN_OC% from the OPEN (open->close). Save the full list.
STEP 2  For each survivor, a multi-timeframe TA scorecard: 7 / 30 / 180-day returns + position vs the
        30d & 180d SMAs. (This encodes the technical-analyst lens quantitatively — fast & reproducible;
        an LLM call per name on a list this big would be hours and add noise, not signal. News ignored.)
STEP 3  Delete names that aren't technically favorable. Output survivors with daily high, close, and %.

FAVORABLE (transparent, adjustable) = price above BOTH the 30d and 180d SMA AND positive 180d return
        (a name in a durable up-trend across every horizon — the opposite of junk).

Usage:  python research/orb_pipeline.py [--min_oc 0.5] [--min_dollar_vol 20] [--date YYYY-MM-DD]
"""
import sys, os, csv, argparse
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


def bars(symbols, start, end, adjustment='raw', chunk=300):
    out = {}
    for i in range(0, len(symbols), chunk):
        grp = symbols[i:i + chunk]; tok = None
        while True:
            p = {'symbols': ','.join(grp), 'timeframe': '1Day', 'start': start, 'end': end,
                 'feed': 'sip', 'adjustment': adjustment, 'limit': 10000}
            if tok:
                p['page_token'] = tok
            j = requests.get(f'{DATA}/v2/stocks/bars', headers=H, params=p, timeout=90).json()
            for s, b in j.get('bars', {}).items():
                out.setdefault(s, []).extend(b)
            tok = j.get('next_page_token')
            if not tok:
                break
    return out


def target_session(today):
    j = requests.get(f'{DATA}/v2/stocks/SPY/bars', headers=H,
                     params={'timeframe': '1Day', 'start': (today - timedelta(days=10)).isoformat(),
                             'feed': 'sip', 'limit': 12}, timeout=20).json()
    prior = [b['t'][:10] for b in j.get('bars', []) if b['t'][:10] < today.isoformat()]
    return prior[-1] if prior else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_oc', type=float, default=0.5, help='min open->close %% (up after the open)')
    ap.add_argument('--min_dollar_vol', type=float, default=20.0, help='min $ volume in millions (junk filter)')
    ap.add_argument('--min_price', type=float, default=5.0, help='min close price (excludes penny/pump crap)')
    ap.add_argument('--date', default=None, help='session date YYYY-MM-DD (default: last completed)')
    a = ap.parse_args()
    today = datetime.now(timezone.utc).date()
    tgt = a.date or target_session(today)
    LIQ = a.min_dollar_vol * 1e6

    # ---------- STEP 1 ----------
    print(f"[step 1] universe ...", flush=True)
    U = universe()
    print(f"         {len(U)} liquid-eligible tickers | target session = {tgt}")
    day = bars(U, tgt, tgt)
    cand = []
    for s, b in day.items():
        b = [x for x in b if x['t'][:10] == tgt]
        if not b:
            continue
        o, hi, c, v = b[0]['o'], b[0]['h'], b[0]['c'], b[0]['v']
        if o <= 0 or c < a.min_price:          # price floor = first junk cut
            continue
        dvol = c * v
        if dvol < LIQ:                          # liquidity floor = second junk cut
            continue
        oc = (c / o - 1) * 100
        if oc >= a.min_oc:
            cand.append(dict(ticker=s, open=round(o, 2), high=round(hi, 2), close=round(c, 2),
                             pct_open_close=round(oc, 2), pct_open_high=round((hi / o - 1) * 100, 2),
                             dollar_vol_m=round(dvol / 1e6, 1)))
    cand.sort(key=lambda r: r['pct_open_close'], reverse=True)
    f1 = os.path.join(HERE, f'orb_up_after_open_{tgt}.csv')
    with open(f1, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand[0].keys())); w.writeheader(); w.writerows(cand)
    open(os.path.join(HERE, f'orb_tickers_{tgt}.txt'), 'w').write('\n'.join(r['ticker'] for r in cand) + '\n')
    print(f"         {len(cand)} up >= {a.min_oc}% after open  (price >= ${a.min_price:.0f}, "
          f"$vol >= ${a.min_dollar_vol:.0f}M)  -> {os.path.basename(f1)}")

    # ---------- STEP 2 ----------
    print(f"[step 2] TA scorecard (7/30/180d) for {len(cand)} names ...", flush=True)
    syms = [r['ticker'] for r in cand]
    # end at tgt, not today: free SIP plan 403s on ranges that include recent (today's) data
    hist = bars(syms, (today - timedelta(days=400)).isoformat(), tgt, adjustment='all')
    def ta(sym):
        b = sorted([x for x in hist.get(sym, []) if x['t'][:10] <= tgt], key=lambda x: x['t'])
        c = [x['c'] for x in b]
        if len(c) < 181:
            return None
        r7, r30, r180 = c[-1]/c[-8]-1, c[-1]/c[-31]-1, c[-1]/c[-181]-1
        sma30, sma180 = sum(c[-30:])/30, sum(c[-180:])/180
        prev_close = c[-2]
        return dict(ret7=r7*100, ret30=r30*100, ret180=r180*100,
                    above30=c[-1] > sma30, above180=c[-1] > sma180, prev_close=prev_close)
    for r in cand:
        r['ta'] = ta(r['ticker'])

    # ---------- STEP 3 ----------
    surv = []
    for r in cand:
        t = r['ta']
        if not t:
            continue
        if t['above30'] and t['above180'] and t['ret180'] > 0:      # FAVORABLE
            daily = (r['close'] / t['prev_close'] - 1) * 100
            surv.append(dict(ticker=r['ticker'], high=r['high'], close=r['close'],
                             pct_open_close=r['pct_open_close'], pct_day=round(daily, 2),
                             ret7=round(t['ret7'], 1), ret30=round(t['ret30'], 1), ret180=round(t['ret180'], 1),
                             dollar_vol_m=r['dollar_vol_m']))
    surv.sort(key=lambda r: r['ret180'], reverse=True)
    scored = [r for r in cand if r['ta']]
    print(f"         scored {len(scored)}/{len(cand)} (rest lacked 180d history) | "
          f"above180d {sum(1 for r in scored if r['ta']['above180'])} | "
          f"+180d ret {sum(1 for r in scored if r['ta']['ret180'] > 0)}")
    f3 = os.path.join(HERE, f'orb_survivors_{tgt}.csv')
    cols = ['ticker', 'high', 'close', 'pct_open_close', 'pct_day', 'ret7', 'ret30', 'ret180', 'dollar_vol_m']
    with open(f3, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(surv)
    print(f"[step 3] {len(surv)} favorable survivors (above 30d & 180d SMA, +180d) -> {os.path.basename(f3)}\n")
    if not surv:
        print("  (none passed — loosen the favorable rule or check data)"); return
    print(f"  {'TICKER':7}{'HIGH':>9}{'CLOSE':>9}{'O->C%':>8}{'DAY%':>8}{'7d%':>7}{'30d%':>8}{'180d%':>8}")
    for r in surv[:40]:
        print(f"  {r['ticker']:7}{r['high']:>9}{r['close']:>9}{r['pct_open_close']:>+8.2f}{r['pct_day']:>+8.2f}"
              f"{r['ret7']:>+7.1f}{r['ret30']:>+8.1f}{r['ret180']:>+8.1f}")
    if len(surv) > 40:
        print(f"  ... +{len(surv)-40} more in {os.path.basename(f3)}")


if __name__ == '__main__':
    main()
