#!/usr/bin/env python3
"""
qc4_tax.py — the honest after-TAX QC4 number. You can't call +24% validated until Uncle Sam is paid.

QC4 flips regimes within months, so most round-trips are held <1yr => SHORT-TERM gains, taxed as
ORDINARY INCOME (the worst rate). This sim tracks every lot (FIFO), classifies each realized gain
short vs long term by actual holding days, nets gains/losses within each category per tax year,
and DEDUCTS the tax bill from the account each year-end so the drag COMPOUNDS (not a one-off haircut).

Model: same QC4 as the dashboard — build_weights(sma_bull=150, s2_mode='rsi'), MOO execution
(weights at close[t] -> traded at open[t+1]); the un-risked slice parks in SHY. Alpaca SIP daily,
dividend-adjusted. Reports PRE-tax vs AFTER-tax across a few bracket scenarios.

Usage:  python research/qc4_tax.py --start 2018-01-01
"""
import sys, os, argparse
from datetime import datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphahconfig as cfg
from letf_strategy import build_weights
import requests

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
ASSETS = ['TQQQ', 'SQQQ', 'TLT', 'SHY']
COST = 5 / 1e4


def pull(syms, start):
    out = {}
    for s in syms:
        tok = None; rows = []
        while True:
            p = {'timeframe': '1Day', 'start': start, 'feed': 'sip', 'adjustment': 'all', 'limit': 10000}
            if tok:
                p['page_token'] = tok
            j = requests.get(f'https://data.alpaca.markets/v2/stocks/{s}/bars', headers=H, params=p, timeout=60).json()
            rows += j.get('bars', [])
            tok = j.get('next_page_token')
            if not tok:
                break
        out[s] = rows
    return out


def simulate(PX, OP, W, dates, pay_tax, st_rate, lt_rate, capital=10000):
    """Lot-level FIFO sim. Returns (equity_series, realized_by_year, tax_paid_total)."""
    cash = capital
    lots = {a: [] for a in ASSETS}        # each lot: [shares, cost_price, buy_date]
    realized = {}                          # year -> [st_gain, lt_gain]
    tax_paid = 0.0
    eq = []
    tax_accrued_year = None
    for k in range(1, len(dates)):
        d, prevd = dates[k], dates[k - 1]
        w = W.loc[prevd]                   # weights decided at close[t-1]
        op = OP.loc[d]                     # executed at open[t]
        # portfolio value at today's open
        pv = cash + sum(sum(l[0] for l in lots[a]) * op[a] for a in ASSETS)
        for a in ASSETS:
            if not np.isfinite(op[a]) or op[a] <= 0:
                continue
            cur = sum(l[0] for l in lots[a])
            tgt = (w.get(a, 0.0) * pv) / op[a]
            delta = tgt - cur
            if delta > 1e-6:               # BUY
                cash -= delta * op[a] * (1 + COST)
                lots[a].append([delta, op[a], d])
            elif delta < -1e-6:            # SELL (FIFO)
                need = -delta
                cash += need * op[a] * (1 - COST)
                yr = d[:4]; realized.setdefault(yr, [0.0, 0.0])
                while need > 1e-9 and lots[a]:
                    sh, cp, bd = lots[a][0]
                    take = min(sh, need)
                    gain = take * (op[a] - cp)
                    hold = (datetime.fromisoformat(d) - datetime.fromisoformat(bd)).days
                    realized[yr][1 if hold >= 365 else 0] += gain
                    lots[a][0][0] -= take; need -= take
                    if lots[a][0][0] <= 1e-9:
                        lots[a].pop(0)
        # year-end: settle tax on this year's realized gains, deduct from cash
        if pay_tax and (k == len(dates) - 1 or dates[k + 1][:4] != d[:4]):
            yr = d[:4]; st, lt = realized.get(yr, [0.0, 0.0])
            bill = max(st, 0) * st_rate + max(lt, 0) * lt_rate     # losses offset within-year only (simplified)
            cash -= bill; tax_paid += bill
        eq.append((d, cash + sum(sum(l[0] for l in lots[a]) * PX.loc[d, a] for a in ASSETS)))
    return pd.Series(dict(eq)), realized, tax_paid


def cagr(series, capital=10000):
    yrs = len(series) / 252
    return (series.iloc[-1] / capital) ** (1 / yrs) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--capital', type=float, default=10000)
    a = ap.parse_args()
    warm = '2017-01-01'
    raw = pull(['QQQ'] + ASSETS, warm)
    PX = pd.DataFrame({s: {b['t'][:10]: b['c'] for b in raw[s]} for s in raw}).dropna()
    OP = pd.DataFrame({s: {b['t'][:10]: b['o'] for b in raw[s]} for s in raw}).reindex(PX.index)
    W = build_weights(PX, sma_bull=150, sma_slow=152, rsi_lo=35, rsi_bear=40, s2_mode='rsi')
    W['SHY'] = (1 - W[['TQQQ', 'SQQQ', 'TLT']].sum(axis=1)).clip(lower=0)
    dates = [d for d in PX.index if d >= a.start]

    gross, realized, _ = simulate(PX, OP, W, dates, False, 0, 0, a.capital)
    tot_st = sum(v[0] for v in realized.values()); tot_lt = sum(v[1] for v in realized.values())
    st_share = tot_st / (tot_st + tot_lt) * 100 if (tot_st + tot_lt) else 0

    print("\n" + "=" * 72)
    print(f"  QC4 AFTER-TAX — {dates[0]} -> {dates[-1]}  (${a.capital:,.0f} start, taxes paid annually)")
    print("=" * 72)
    print(f"  PRE-TAX:  ${gross.iloc[-1]:,.0f}   CAGR {cagr(gross, a.capital)*100:+.1f}%")
    print(f"  realized gains over life:  short-term ${tot_st:,.0f}  |  long-term ${tot_lt:,.0f}")
    print(f"  >> {st_share:.0f}% of gains are SHORT-TERM (taxed as ordinary income) — the tax problem in one number\n")
    print(f"  {'scenario':<34}{'ST/LT rate':>12}{'after-tax $':>14}{'CAGR':>9}{'drag':>8}")
    scen = [('24% bracket, fed only', 0.24, 0.15),
            ('32% bracket, fed only', 0.32, 0.15),
            ('37% top bracket, fed only', 0.37, 0.20),
            ('37% fed + 13.3% CA state', 0.37 + 0.133, 0.20 + 0.133),
            ('3.8% NIIT + 37% fed (no state)', 0.37 + 0.038, 0.20 + 0.038)]
    g = cagr(gross, a.capital) * 100
    for name, st, lt in scen:
        net, _, tax = simulate(PX, OP, W, dates, True, min(st, 0.6), min(lt, 0.5), a.capital)
        print(f"  {name:<34}{f'{st*100:.0f}/{lt*100:.0f}%':>12}{net.iloc[-1]:>14,.0f}{cagr(net,a.capital)*100:>+8.1f}%{g-cagr(net,a.capital)*100:>+7.1f}%")
    print("\n  note: taxes deducted each year-end (compounding drag). Losses offset only within year/category")
    print("  (no carryforward) — a mild simplification. Leveraged-ETF wash-sale nuances ignored.")


if __name__ == '__main__':
    main()
