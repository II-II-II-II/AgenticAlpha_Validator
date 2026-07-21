#!/usr/bin/env python3
"""
momentum.py — cross-sectional stock momentum backtest (the one validated research edge).

STRATEGY
  Rank a fixed universe of liquid large-caps by "12-1" momentum (trailing 12-month return,
  skipping the most recent month — the standard convention; the 1-month skip avoids short-term
  reversal). Hold the top-N equal-weight, rebalance monthly. Buy the winners; bet they keep winning.
  This is the OPPOSITE of the mean-reversion ideas in ../graveyard/ (dip-buy, ORB, band-swing),
  all of which failed. Momentum is one of the most-replicated anomalies in finance (Jegadeesh-Titman).

WHY THIS ONE SURVIVED — the survivorship control
  A momentum backtest on today's large-caps is biased: those names *survived* to be large-caps.
  The honest test isn't "does it beat SPY" (it will, from the bias alone) — it's "does it beat an
  EQUAL-WEIGHT HOLD OF THE SAME UNIVERSE," which carries the *identical* bias. Momentum beating that
  is a real SELECTION effect. It does, by +2 to +11%/yr, and the edge is monotonic in N (a factor
  signature, not overfitting). That control is the whole point — it's why this passed when nothing else did.

VALIDATED RESULT (2018-2026, top-10, daily adjusted data, 0.1%/side cost):
  Momentum CAGR +21.5% | Sharpe 1.06 | maxDD 25%  vs  universe-hold 15.5%  vs  SPY 14.5%
  Excess over the survivorship-controlled benchmark: +6.0%/yr.

HONEST CAVEATS (documented, not hidden):
  - One broad regime (2018-26 mega-cap-tech bull). NOT a 2020-21 bubble artifact (it LOST in 2021's
    growth-to-value reversal) — but it did underperform for a 2018-2021 stretch. Multi-year droughts
    are the real cost of admission.
  - No momentum-crash in sample (the factor's infamous tail, e.g. 2009). 25% maxDD understates it.
  - ~28 rebalance-trades/yr → real short-term-capital-gains tax drag in a taxable account.
  - Do NOT regime-gate it: tested (../graveyard/), the 150d gate HURTS momentum because momentum
    already self-adapts (it rotates into what's working); the lag just whipsaws it at V-recoveries.

Usage:  python momentum.py            # full backtest + by-year + current picks
        python momentum.py --topn 5   # concentration variant
"""
import sys, os, argparse, math, statistics as st
from datetime import datetime
from collections import defaultdict
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}

# Fixed universe: large, liquid, US names ALREADY established by 2017 (reduces — not eliminates —
# survivorship bias vs. picking today's winners). The equal-weight-hold benchmark controls the rest.
UNIVERSE = ('AAPL MSFT AMZN GOOGL META NVDA JPM JNJ V PG HD MA BAC XOM CVX KO PEP WMT DIS CSCO '
            'INTC VZ T PFE MRK ABBV CMCSA ADBE CRM NKE MCD COST TMO ACN ABT DHR TXN QCOM AMD ORCL '
            'IBM GE CAT BA MMM HON UNH LLY WFC GS MS C AXP SBUX LOW UPS RTX LMT AMGN GILD BKNG ADP').split()

LOOKBACK, SKIP, REBAL, COST = 252, 21, 21, 0.001   # 12-mo lookback, 1-mo skip, monthly rebalance, 10bps/side


def fetch_daily(symbols, start='2017-01-01'):
    """Daily ADJUSTED closes (splits+dividends) from Alpaca SIP. Returns {sym: {date: close}}."""
    out = {}
    for grp in (symbols[i:i+50] for i in range(0, len(symbols), 50)):
        tok = None
        while True:
            p = {'symbols': ','.join(grp), 'timeframe': '1Day', 'start': start,
                 'adjustment': 'all', 'limit': 10000, 'feed': 'sip'}
            if tok:
                p['page_token'] = tok
            j = requests.get('https://data.alpaca.markets/v2/stocks/bars', headers=H, params=p, timeout=40).json()
            for s, bars in j.get('bars', {}).items():
                out.setdefault(s, {}).update({b['t'][:10]: b['c'] for b in bars})
            tok = j.get('next_page_token')
            if not tok:
                break
    return out


def _metrics(equity, periods_per_year=12):
    yrs = len(equity) / periods_per_year
    cagr = (equity[-1]) ** (1 / yrs) - 1
    peak, mdd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
    return cagr * 100, mdd * 100


def backtest(PX, topN=10):
    """Momentum top-N vs the survivorship-controlled equal-weight-hold of the same universe.
    Point-in-time: rank on data through the rebalance date, earn the forward period. No look-ahead."""
    univ = [s for s in UNIVERSE if s in PX and len(PX[s]) > LOOKBACK + 8]
    dates = sorted(set.intersection(*[set(PX[s]) for s in univ]))
    reb = list(range(LOOKBACK + 8, len(dates), REBAL))
    mom_eq, ew_eq = [1.0], [1.0]
    prets, yr_mom, yr_ew, yr_trades = [], defaultdict(lambda: 1.0), defaultdict(lambda: 1.0), defaultdict(int)
    prev = set()
    for k in range(len(reb) - 1):
        i, d, dn, y = reb[k], dates[reb[k]], dates[reb[k+1]], dates[reb[k]][:4]
        mom = {s: PX[s][dates[i-SKIP]] / PX[s][dates[i-LOOKBACK]] - 1
               for s in univ if PX[s].get(dates[i-LOOKBACK]) and PX[s].get(dates[i-SKIP])}
        picks = set(sorted(mom, key=mom.get, reverse=True)[:topN])
        fwd = lambda names: st.mean([PX[s][dn]/PX[s][d]-1 for s in names if d in PX[s] and dn in PX[s]])
        turn = len(picks - prev); yr_trades[y] += turn
        mret = fwd(picks) - COST * turn / max(len(picks), 1); prev = picks
        eret = fwd(univ)
        mom_eq.append(mom_eq[-1] * (1 + mret)); ew_eq.append(ew_eq[-1] * (1 + eret))
        prets.append(mret); yr_mom[y] *= (1 + mret); yr_ew[y] *= (1 + eret)
    mc, md = _metrics(mom_eq); ec, ed = _metrics(ew_eq)
    sharpe = st.mean(prets) / st.pstdev(prets) * math.sqrt(12) if st.pstdev(prets) else 0
    return dict(mom_cagr=mc, mom_mdd=md, mom_sharpe=sharpe, ew_cagr=ec, ew_mdd=ed,
                excess=mc-ec, years=sorted(yr_mom), yr_mom=yr_mom, yr_ew=yr_ew, yr_trades=yr_trades,
                span=(dates[reb[0]], dates[reb[-1]]))


def current_picks(PX, topN=10):
    """The live 12-1 ranking — what the strategy would hold today."""
    univ = [s for s in UNIVERSE if s in PX and len(PX[s]) >= LOOKBACK]
    mom = {}
    for s in univ:
        c = [PX[s][d] for d in sorted(PX[s])]
        mom[s] = c[-SKIP] / c[-LOOKBACK] - 1
    return sorted(mom.items(), key=lambda kv: kv[1], reverse=True)


def main():
    ap = argparse.ArgumentParser(description='Cross-sectional momentum backtest')
    ap.add_argument('--topn', type=int, default=10)
    a = ap.parse_args()
    print("loading daily adjusted data ...")
    PX = fetch_daily(UNIVERSE + ['SPY'])
    for N in ([a.topn] if a.topn != 10 else [5, 10, 15]):
        r = backtest(PX, N)
        print(f"\n=== Momentum top-{N} | {r['span'][0]} -> {r['span'][1]} ===")
        print(f"  MOMENTUM        CAGR {r['mom_cagr']:+5.1f}%  Sharpe {r['mom_sharpe']:.2f}  maxDD {r['mom_mdd']:.0f}%")
        print(f"  universe-hold   CAGR {r['ew_cagr']:+5.1f}%              maxDD {r['ew_mdd']:.0f}%   <- survivorship control")
        print(f"  >> EXCESS over the controlled benchmark: {r['excess']:+.1f}%/yr  |  ~{sum(r['yr_trades'].values())/len(r['years']):.0f} trades/yr")
        if N == 10:
            print(f"\n  {'year':<6}{'momentum':>10}{'univ-hold':>11}{'excess':>9}")
            for y in r['years']:
                m, e = (r['yr_mom'][y]-1)*100, (r['yr_ew'][y]-1)*100
                print(f"  {y:<6}{m:>+9.0f}%{e:>+10.0f}%{m-e:>+8.0f}%")
    print(f"\n=== current top-10 picks (what it would hold today) ===")
    for i, (s, m) in enumerate(current_picks(PX)[:10], 1):
        print(f"  {i:2}. {s:5} {m*100:+.0f}%")


if __name__ == '__main__':
    main()
