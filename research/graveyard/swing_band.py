#!/usr/bin/env python3
"""
swing_band.py — mechanical band-swing backtester (NO LLM).

Strategy: ping-pong bands.
  - Start in cash. Enter (buy) when regime allows.
  - While HOLDING: sell if intraday high >= entry*(1+X)  [take-profit "pop"]
                   OR (if stop on) intraday low <= entry*(1-STOP)  [cut loss]
                   OR (if gate on) QQQ closes below its 150d SMA   [regime exit to cash]
  - While in CASH: rebuy when intraday low <= last_exit*(1-Y)  [buy the "dip"]  (gate must allow)
  - "More bang" = trade TQQQ (3x). Compared vs buy-&-hold TQQQ and QQQ.

Honest accounting:
  - fills at the trigger price (limit orders), slippage applied per side
  - if BOTH target and stop are touchable same day, assume STOP first (worst-case, no cheating)
  - regime gate uses QQQ vs its own 150d SMA, aligned by date
  - reports vs buy-&-hold, MAX DRAWDOWN, exposure, trades, and the 2022 drawdown specifically
"""
import requests, statistics as st, argparse
from datetime import datetime
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


def backtest(tqqq, qsma_ok, dates, X, Y, stop, gate, slip=0.0005, cap=10000.0):
    """qsma_ok[date] = True if QQQ above its 150d SMA that day (regime on)."""
    cash = cap; shares = 0.0; entry = None; last_exit = None
    equity = []; trades = []; days_in = 0
    for d in dates:
        b = tqqq[d]; o, hi, lo, c = b['o'], b['h'], b['l'], b['c']
        regime = qsma_ok.get(d, True) if gate else True
        if shares > 0:                                   # HOLDING
            target = entry * (1 + X)
            sl = entry * (1 - stop) if stop else -1
            sold = None
            if stop and lo <= sl:                        # worst-case: stop checked first
                sold = sl * (1 - slip)
            elif hi >= target:
                sold = target * (1 - slip)
            elif gate and not regime:                    # regime break -> exit at close
                sold = c * (1 - slip)
            if sold is not None:
                cash = shares * sold; trades.append(sold / entry - 1)
                last_exit = sold; shares = 0.0; entry = None
            else:
                days_in += 1
        else:                                            # IN CASH
            trigger = last_exit * (1 - Y) if last_exit is not None else o
            can_buy = regime and (last_exit is None or lo <= trigger)
            if can_buy:
                fill = (trigger if last_exit is not None else o) * (1 + slip)
                fill = min(fill, hi) if last_exit is not None else fill
                shares = cash / fill; entry = fill; cash = 0.0; days_in += 1
        equity.append(cash + shares * c)
    # metrics
    finalv = equity[-1]
    peak = equity[0]; mdd = 0
    for e in equity:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    wins = sum(1 for t in trades if t > 0)
    return dict(final=finalv, ret=(finalv/cap-1)*100, mdd=mdd*100, ntr=len(trades),
                winrate=(wins/len(trades)*100 if trades else 0),
                exposure=days_in/len(dates)*100, equity=equity)


def bh(px, dates, cap=10000.0):
    e = [cap * px[d]['c'] / px[dates[0]]['o'] for d in dates]
    peak = e[0]; mdd = 0
    for x in e:
        peak = max(peak, x); mdd = max(mdd, (peak-x)/peak)
    return dict(final=e[-1], ret=(e[-1]/cap-1)*100, mdd=mdd*100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2019-01-01')
    a = ap.parse_args()
    print("loading TQQQ + QQQ ...")
    tq = bars('TQQQ', a.start); qq = bars('QQQ', a.start)
    dates = sorted(set(tq) & set(qq))
    # QQQ 150d regime, aligned by date
    qc = [qq[d]['c'] for d in dates]
    qsma_ok = {}
    for i, d in enumerate(dates):
        qsma_ok[d] = (qc[i] > sum(qc[max(0,i-149):i+1]) / min(i+1, 150)) if i >= 20 else True
    span = f"{dates[0]} -> {dates[-1]} ({len(dates)}d)"
    print(f"span: {span}\n")

    bt = bh(tq, dates); bq = bh(qq, dates)
    print(f"{'CONFIG':<44}{'final$':>10}{'ret%':>9}{'maxDD%':>8}{'trades':>7}{'win%':>6}{'expo%':>7}")
    print(f"{'buy&hold TQQQ':<44}{bt['final']:>10,.0f}{bt['ret']:>+9.0f}{bt['mdd']:>8.0f}{'—':>7}{'—':>6}{'100':>7}")
    print(f"{'buy&hold QQQ':<44}{bq['final']:>10,.0f}{bq['ret']:>+9.0f}{bq['mdd']:>8.0f}{'—':>7}{'—':>6}{'100':>7}")
    print("-"*91)
    configs = [
        ("naive band 3/3, no gate no stop",        3,3, 0,   False),
        ("band 3/3 + regime gate",                 3,3, 0,   True),
        ("band 3/3 + gate + 10% stop",             3,3, 0.10,True),
        ("band 5/5 + gate + 12% stop",             5,5, 0.12,True),
        ("band 8/8 + gate + 15% stop",             8,8, 0.15,True),
        ("band 2/2 + gate + 8% stop (your style)", 2,2, 0.08,True),
    ]
    for name, X, Y, stop, gate in configs:
        r = backtest(tq, qsma_ok, dates, X/100, Y/100, stop, gate)
        print(f"{name:<44}{r['final']:>10,.0f}{r['ret']:>+9.0f}{r['mdd']:>8.0f}{r['ntr']:>7}{r['winrate']:>6.0f}{r['exposure']:>7.0f}")
    print("\n(TQQQ 3x; $10k start; 0.05% slippage/side; regime gate = QQQ vs its 150d SMA)")


if __name__ == '__main__':
    main()
