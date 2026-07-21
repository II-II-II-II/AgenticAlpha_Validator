#!/usr/bin/env python3
"""
dip_explore.py — explore "buy X% below the average, sell on a +Y% pop, HOLD if it doesn't recover".

This models the user's ACTUAL strategy: buy dips (price a threshold below a reference), sell into
strength, and if a position doesn't recover, keep holding (no forced loss, no EOD flat). Single
position at a time; cash between.

KEY HONESTY: because we only ever sell winners, win-rate is ~100% BY CONSTRUCTION. The real risk is
hidden in (a) how LONG capital is stuck (hold days), (b) how DEEP a position goes underwater before
recovering — max adverse excursion (MAE), and (c) whether the final open position is stranded and
how deep. Those are the numbers reported, alongside return vs buy-&-hold and exposure (cash drag).

No look-ahead: entry signal uses closes through day i-1; fill at day i's open.
"""
import requests, statistics as st
from datetime import datetime, timezone, timedelta
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}


def bars(sym, start='2019-01-01'):
    out, tok = [], None
    while True:
        p = {'timeframe': '1Day', 'start': start, 'adjustment': 'all', 'limit': 10000, 'feed': 'sip'}
        if tok: p['page_token'] = tok
        r = requests.get(f'https://data.alpaca.markets/v2/stocks/{sym}/bars', headers=H, params=p, timeout=40).json()
        out += r.get('bars', []); tok = r.get('next_page_token')
        if not tok: break
    return {b['t'][:10]: b for b in out}


def bh(px, dates, cap=10000.0):
    e = [cap * px[d]['c'] / px[dates[0]]['o'] for d in dates]
    peak = e[0]; mdd = 0
    for x in e:
        peak = max(peak, x); mdd = max(mdd, (peak-x)/peak)
    return e[-1], (e[-1]/cap-1)*100, mdd*100


def backtest(px, dates, ref_type, N, X, Y, cap=10000.0, slip=0.0004):
    cl = [px[d]['c'] for d in dates]; hi = [px[d]['h'] for d in dates]; lo = [px[d]['l'] for d in dates]
    cash = cap; shares = 0.0; entry = None; eidx = None; mae = 0.0
    equity = []; trades = []; days_in = 0

    def ref(j):
        if ref_type == 'sma':      return sum(cl[max(0, j-N+1):j+1]) / min(j+1, N)
        if ref_type == 'high':     return max(hi[max(0, j-N+1):j+1])
        if ref_type == 'pclose':   return cl[j-1] if j >= 1 else None   # signal = a down day of X%
        return None

    for i in range(2, len(dates)):
        d = dates[i]; o = px[d]['o']
        if shares == 0:                                  # look for ENTRY (signal thru i-1, fill at open i)
            rf = ref(i-1); price = cl[i-1]
            if rf and price <= rf * (1 - X):
                fill = o * (1 + slip); shares = cash / fill; entry = fill; eidx = i; cash = 0.0; mae = 0.0
        if shares > 0:                                   # manage / EXIT on pop
            mae = min(mae, lo[i] / entry - 1)            # track worst underwater
            tgt = entry * (1 + Y)
            if hi[i] >= tgt:
                f = tgt * (1 - slip); cash = shares * f
                trades.append({'ret': f/entry-1, 'hold': i-eidx, 'mae': mae})
                shares = 0.0; entry = None
            else:
                days_in += 1
        equity.append(cash + shares * cl[i])
    # metrics
    finalv = equity[-1]; peak = equity[0]; mdd = 0
    for e in equity:
        peak = max(peak, e); mdd = max(mdd, (peak-e)/peak)
    holds = [t['hold'] for t in trades]; maes = [t['mae'] for t in trades]
    stuck = ''
    if shares > 0:                                       # position still open at end
        cur = cl[-1]/entry - 1
        stuck = f"OPEN {i-eidx}d @ {cur*100:+.0f}%"
    return dict(final=finalv, ret=(finalv/cap-1)*100, mdd=mdd*100, ntr=len(trades),
                avghold=(st.mean(holds) if holds else 0), maxhold=(max(holds) if holds else 0),
                worstmae=(min(maes)*100 if maes else 0), expo=days_in/len(dates)*100, stuck=stuck)


def main():
    print("loading TQQQ / QQQ ...")
    tq = bars('TQQQ'); qq = bars('QQQ')
    dates = sorted(set(tq) & set(qq))
    print(f"span {dates[0]} -> {dates[-1]} ({len(dates)}d)\n")
    ft, rt, mt = bh(tq, dates); fq, rq, mq = bh(qq, dates)
    print(f"BENCHMARK  buy&hold TQQQ  ${ft:,.0f} ({rt:+.0f}%)  maxDD {mt:.0f}%")
    print(f"BENCHMARK  buy&hold QQQ   ${fq:,.0f} ({rq:+.0f}%)  maxDD {mq:.0f}%")
    print("\n" + "="*104)
    print(f"{'entry rule':<26}{'pop':>5}{'final$':>10}{'ret%':>8}{'maxDD%':>8}{'trds':>5}{'avgHold':>8}{'maxHold':>8}{'worstMAE':>9}{'expo%':>7}  stuck")
    print("-"*104)
    scen = [('sma', 20), ('sma', 50), ('high', 20), ('pclose', 1)]
    labels = {('sma',20):'3/5/8% below SMA20', ('sma',50):'3/5/8% below SMA50',
              ('high',20):'5/8/12% below 20d-high', ('pclose',1):'3/5/8% down-day'}
    for ref_type, N in scen:
        Xs = [0.03,0.05,0.08] if ref_type != 'high' else [0.05,0.08,0.12]
        for X in Xs:
            for Y in [0.05, 0.10]:
                r = backtest(tq, dates, ref_type, N, X, Y)
                name = f"{X*100:.0f}% below {ref_type}{N if ref_type!='pclose' else ''}".replace('pclose','prior-close')
                print(f"{name:<26}{Y*100:>4.0f}%{r['final']:>10,.0f}{r['ret']:>+8.0f}{r['mdd']:>8.0f}{r['ntr']:>5}"
                      f"{r['avghold']:>8.0f}{r['maxhold']:>8}{r['worstmae']:>+9.0f}{r['expo']:>7.0f}  {r['stuck']}")
    print("\n(TQQQ, $10k, 0.04% slip. worstMAE = deepest an open position went underwater before recovering.")
    print(" 100% win-rate is BY DESIGN — the risk lives in maxHold, worstMAE, and 'stuck'.)")


if __name__ == '__main__':
    main()
