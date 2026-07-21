#!/usr/bin/env python3
"""
dip_scanner.py — morning SHORT-WINDOW reversal screener for tech high-volume names (Alpaca data).

Finds names that have REVERTED DOWN (near the low of their recent range) inside a CHOPPY/mean-reverting
regime — i.e. "primed for a reversal" per raw TA. NOT a backtest and NOT a forecast: it's a screener that
describes the CURRENT regime + anchors a return *estimate* to each stock's OWN recent dip-bounce behavior.
Recent chop can flip to a trend without warning, so it's regime-gated (QQQ vs 150d) and every trade needs a stop.

Per name over the lookback window:
  price   latest Alpaca close
  amp%    avg daily range (high-low)/close        -> room to trade
  eff     |net move|/total path (0=chop, 1=trend) -> low = ranging
  ac1     lag-1 return autocorr (neg=reverting)
  rngpos  where price sits in its N-day range (0=at low=DIP now, 1=at high)
  up_top% upside if it mean-reverts to its N-day high
  bounce  avg fwd 3-day return AFTER dip days (<-2%) + win/n   (its own recent behavior)
Rank = z(amp)+z(-eff)+z(-ac1). SETUP flag = reverted down (rngpos<0.30) + ranging (eff<0.4) + bounce>0.

Usage:  python dip_scanner.py                 # 60-day window (default)
        python dip_scanner.py --lookback 30
        python dip_scanner.py AAPL NVDA ...
"""
import argparse
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, requests
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
DATA = 'https://data.alpaca.markets'

UNIV = ("AAPL MSFT NVDA AMD AVGO META GOOGL AMZN TSLA NFLX INTC MU QCOM TXN AMAT ORCL CSCO ADBE CRM "
        "NOW PANW SNOW PLTR SMCI ARM MRVL ASML TSM ON NXPI COIN HOOD UBER SOFI RGTI IONQ RKLB DELL ANET "
        "QQQ TQQQ SOXL SMH SOXX XLK").split()


def get_bars(syms, start):
    out = {}
    tok = None
    while True:
        p = {'symbols': ','.join(syms), 'timeframe': '1Day', 'start': start,
             'limit': 10000, 'feed': 'iex', 'adjustment': 'all'}
        if tok:
            p['page_token'] = tok
        j = requests.get(f"{DATA}/v2/stocks/bars", headers=H, params=p, timeout=25).json()
        for s, bars in j.get('bars', {}).items():
            out.setdefault(s, []).extend(bars)
        tok = j.get('next_page_token')
        if not tok:
            break
    return {s: pd.DataFrame(b) for s, b in out.items() if b}


def regime(data_qqq):
    """BULL if QQQ above its 150-day SMA."""
    c = data_qqq['c']
    if len(c) < 150:
        return 'UNKNOWN', np.nan, float(c.iloc[-1])
    sma = float(c.tail(150).mean()); last = float(c.iloc[-1])
    return ('BULL' if last > sma else 'BEAR'), sma, last


def metrics(df, N):
    df = df.tail(N)
    if len(df) < max(8, N // 3):
        return None
    c, h, l = df['c'].values, df['h'].values, df['l'].values
    r = np.diff(c) / c[:-1]
    amp = float(np.mean((h - l) / c) * 100)
    net = c[-1] / c[0] - 1
    path = float(np.sum(np.abs(r)))
    eff = abs(net) / path if path > 0 else np.nan
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 else np.nan
    rhi, rlo = h.max(), l.min()
    price = float(c[-1])
    rngpos = (price - rlo) / (rhi - rlo) if rhi > rlo else 0.5
    up_top = (rhi - price) / price * 100
    # forward 3-day return after a down day (>2%) — the stock's own recent bounce behavior
    bnc = [c[j + 3] / c[j] - 1 for j in range(1, len(c) - 3) if r[j - 1] < -0.02]
    b_avg = float(np.mean(bnc) * 100) if bnc else np.nan
    b_win, b_n = (int(np.sum(np.array(bnc) > 0)), len(bnc)) if bnc else (0, 0)
    trend10 = float(c[-1] / c[-11] - 1) if len(c) >= 11 else 0.0   # recent slope: downtrend = knife
    return dict(name=None, price=price, amp=amp, eff=eff, ac1=ac1, rngpos=rngpos,
                up_top=up_top, b_avg=b_avg, b_win=b_win, b_n=b_n, trend10=trend10)


def fit_score(m):
    """Deterministic 0-100 pattern-fit: near a dip + MEAN-REVERTING (not trending) + recent bounce.
    Penalizes downtrends hard (RGTI-style knives) via ac1>0 and a negative 10-day slope."""
    s = 50.0
    s += max(-15, min(15, (0.30 - m['rngpos']) * 60))          # near the low = good
    s += max(-20, min(20, (-m['ac1']) * 60))                   # ac1<0 revert = good; ac1>0 trend = bad
    t = m.get('trend10', 0.0)
    s += max(-28, t * 220) if t < 0 else min(10, t * 100)      # falling knife penalty (downtrend)
    if m['b_n']:
        s += max(-10, min(15, (m['b_win'] / m['b_n'] - 0.5) * 40 + (m['b_avg'] or 0) * 2))
    return round(max(0.0, min(100.0, s)))


def z(s):
    s = pd.Series(s, dtype=float); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def scan(names, lookback):
    """Callable API: returns (regime_info_dict, ranked_df, raw_data). Used by dip_report.py."""
    start = (datetime.now(timezone.utc) - timedelta(days=260)).date().isoformat()
    data = get_bars(list(set(names + ['QQQ'])), start)
    reg, sma, qlast = regime(data.get('QQQ', pd.DataFrame({'c': [np.nan]})))
    rows = []
    for n in names:
        if n in data:
            m = metrics(data[n], lookback)
            if m:
                m['name'] = n; rows.append(m)
    df = pd.DataFrame(rows)
    if not df.empty:
        df['fit'] = df.apply(fit_score, axis=1)
        # SETUP = at a dip + MEAN-REVERTING (ac1<0) + not a downtrend + recent bounce + fit>=55
        df['setup'] = ((df.rngpos < 0.35) & (df.ac1 < 0) & (df.trend10 > -0.10) &
                       (df.b_avg > 0) & (df.fit >= 55))
        df = df.sort_values(['setup', 'fit'], ascending=False)
    return dict(regime=reg, sma=sma, qqq=qlast), df, data


def run(names, N, data):
    rows = []
    for n in names:
        if n not in data:
            continue
        m = metrics(data[n], N)
        if m:
            m['name'] = n; rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  no data for {N}d"); return
    df['score'] = z(df.amp) + z(-df.eff) + z(-df.ac1)
    df['setup'] = (df.rngpos < 0.30) & (df.eff < 0.40) & (df.b_avg > 0)
    df = df.sort_values(['setup', 'score'], ascending=False)
    print(f"\n=== {N}-day reversal screen — sorted (SETUP first, then chop-quality) ===")
    print(f"{'name':<6}{'price':>9}{'amp%':>6}{'eff':>6}{'ac1':>7}{'rngpos':>7}{'up_top%':>8}{'bounce3d':>13}{'score':>7} SETUP")
    for _, r in df.head(22).iterrows():
        b = f"{r.b_avg:+.1f}%({r.b_win}/{r.b_n})" if r.b_n else "  -"
        print(f"{r['name']:<6}{r.price:>9.2f}{r.amp:>6.1f}{r.eff:>6.2f}{r.ac1:>+7.2f}{r.rngpos:>7.2f}"
              f"{r.up_top:>+8.1f}{b:>13}{r.score:>7.2f}  {'<<<' if r.setup else ''}")
    ns = int(df.setup.sum())
    print(f"  {ns} SETUP(s): reverted-down (rngpos<0.30) + ranging (eff<0.40) + positive recent bounce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tickers', nargs='*')
    ap.add_argument('--lookback', type=int, default=60)
    a = ap.parse_args()
    names = [t.upper() for t in a.tickers] or UNIV
    start = (datetime.now(timezone.utc) - timedelta(days=260)).date().isoformat()   # long enough for 150d regime
    print(f"pulling {len(names)} tech names from Alpaca...")
    data = get_bars(list(set(names + ['QQQ'])), start)
    reg, sma, qlast = regime(data['QQQ'])
    print(f"REGIME: {reg}  (QQQ {qlast:.2f} vs 150d {sma:.2f})   "
          f"{'-> chop-in-uptrend, dip-buys OK' if reg=='BULL' else '-> TREND BROKEN, dip-buys = knife-catch, SKIP'}")
    run(names, a.lookback, data)
    print("\n(screener, not a forecast: estimates = each stock's OWN recent behavior. Regime-gate + stops mandatory.)")


if __name__ == '__main__':
    main()
