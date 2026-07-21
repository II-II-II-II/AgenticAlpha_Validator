#!/usr/bin/env python3
"""
ETF Dual-Momentum Rotation Backtest
===================================
The user's testable theory: each rebalance, look at the best-performing ETFs over the
last X days, hold the top N, and SETTLE IN CASH when leaders are weak (absolute-momentum gate).

  - Relative momentum: rank the universe by trailing `lookback`-day return, hold top `top_n`.
  - Absolute momentum: a selected ETF is only held if its trailing return beats cash (SHY);
    otherwise that slot goes to cash. So in broad downturns the book rotates to cash/SHY.
  - Equal weight across held slots. Rebalance every `freq` trading days (1=daily,5=weekly,21=monthly).
  - Turnover cost charged at `cost_bps` per unit of weight traded (ETF spreads are tiny; default 5bps).

No look-ahead: the signal at close[t] uses only prices <= t, and earns the return t -> t+1.
Short-history ETFs (BUZZ/VEGN/RVER) are simply ineligible until they have a full lookback window.

Split is CHRONOLOGICAL (momentum needs consecutive days): train 60% / test 20% / vault 20%.
Everything is judged against buy-and-hold SPY on return, CAGR, Sharpe, and max drawdown.

Usage:
  python etf_rotation.py                       # grid on train, validate best on test, reveal vault
  python etf_rotation.py --single --lookback 90 --top_n 3 --freq 5
"""
import argparse
import numpy as np
import pandas as pd

CASH = 'SHY'            # cash proxy: 1-3yr treasuries, ~near-zero vol
TRADING_DAYS = 252


def load_prices(path='datalake/etf_daily.parquet', start=None, end=None):
    df = pd.read_parquet(path)
    px = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    return px


def run_strategy(px, lookback, top_n, freq, cost_bps, abs_mom=True, rank_pool=None):
    """Returns a daily strategy-return Series across the full date index."""
    rets = px.pct_change()
    mom = px / px.shift(lookback) - 1.0          # trailing lookback return per ticker
    dates = px.index
    pool = list(px.columns) if rank_pool is None else [c for c in rank_pool if c in px.columns]

    weights = pd.Series(0.0, index=px.columns)
    w_hist = pd.DataFrame(0.0, index=dates, columns=px.columns)
    cost_series = pd.Series(0.0, index=dates)

    for i, d in enumerate(dates):
        if i % freq == 0 and i >= lookback:
            m = mom.loc[d, pool].dropna()
            cash_mom = mom.loc[d, CASH] if CASH in mom.columns and not np.isnan(mom.loc[d, CASH]) else 0.0
            ranked = m.sort_values(ascending=False)
            picks = []
            for tk in ranked.index:
                if len(picks) >= top_n:
                    break
                if abs_mom and ranked[tk] <= max(cash_mom, 0.0):
                    continue                      # weak leader -> this slot becomes cash
                picks.append(tk)
            new_w = pd.Series(0.0, index=px.columns)
            if picks:
                for tk in picks:
                    new_w[tk] = 1.0 / top_n       # empty slots stay in cash (sum<1 -> rest uninvested=cash)
            cost_series[d] = np.abs(new_w - weights).sum() * (cost_bps / 1e4)
            weights = new_w
        w_hist.loc[d] = weights

    # Earn t->t+1: yesterday's weights times today's returns, minus the day's rebalance cost
    port = (w_hist.shift(1) * rets).sum(axis=1) - cost_series
    return port.fillna(0.0)


def metrics(daily, freq_days=TRADING_DAYS):
    daily = daily.dropna()
    if len(daily) < 2:
        return dict(total=0, cagr=0, sharpe=0, maxdd=0, n=len(daily))
    eq = (1 + daily).cumprod()
    total = eq.iloc[-1] - 1
    yrs = len(daily) / freq_days
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    sharpe = (daily.mean() / daily.std() * np.sqrt(freq_days)) if daily.std() > 0 else 0
    maxdd = (eq / eq.cummax() - 1).min()
    return dict(total=total, cagr=cagr, sharpe=sharpe, maxdd=maxdd, n=len(daily))


def bh_spy(px, idx):
    r = px['SPY'].pct_change().reindex(idx).fillna(0.0)
    return metrics(r)


def fmt(m):
    return (f"tot {m['total']*100:+7.1f}%  CAGR {m['cagr']*100:+6.1f}%  "
            f"Sharpe {m['sharpe']:+5.2f}  maxDD {m['maxdd']*100:6.1f}%  n={m['n']}")


def split(dates):
    n = len(dates)
    return dates[:int(n*.6)], dates[int(n*.6):int(n*.8)], dates[int(n*.8):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datalake/etf_daily.parquet')
    ap.add_argument('--start', default=None, help='restrict window start, e.g. 2023-06-20')
    ap.add_argument('--end', default=None)
    ap.add_argument('--cost_bps', type=float, default=5.0)
    ap.add_argument('--single', action='store_true')
    ap.add_argument('--lookback', type=int, default=90)
    ap.add_argument('--top_n', type=int, default=3)
    ap.add_argument('--freq', type=int, default=5)
    ap.add_argument('--no_abs', action='store_true', help='disable absolute-momentum cash gate')
    a = ap.parse_args()

    px = load_prices(a.data, start=a.start, end=a.end)
    train, test, vault = split(px.index)
    print(f"[*] {len(px.columns)} ETFs | {px.index[0].date()} -> {px.index[-1].date()} | "
          f"train {train[0].date()}..{train[-1].date()} | test {test[0].date()}..{test[-1].date()} | "
          f"vault {vault[0].date()}..{vault[-1].date()}")
    print(f"[*] cost {a.cost_bps}bps/trade-unit | cash proxy {CASH}\n")

    if a.single:
        port = run_strategy(px, a.lookback, a.top_n, a.freq, a.cost_bps, abs_mom=not a.no_abs)
        for label, idx in [('TRAIN', train), ('TEST', test), ('VAULT', vault), ('FULL', px.index)]:
            s = metrics(port.reindex(idx).fillna(0.0))
            b = bh_spy(px, idx)
            print(f"{label:6} STRAT {fmt(s)}")
            print(f"{'':6} SPY   {fmt(b)}\n")
        return

    # ---- Grid search on TRAIN only, then confirm the winner OOS ----
    grid = [(lb, n, fq) for lb in (20, 60, 90, 120, 200) for n in (1, 2, 3, 5) for fq in (1, 5, 21)]
    print(f"[*] Grid: {len(grid)} combos, scored on TRAIN by Sharpe (vs SPY train Sharpe "
          f"{bh_spy(px, train)['sharpe']:.2f})\n")
    rows = []
    for lb, n, fq in grid:
        port = run_strategy(px, lb, n, fq, a.cost_bps, abs_mom=not a.no_abs)
        rows.append((lb, n, fq, metrics(port.reindex(train).fillna(0.0)), port))
    rows.sort(key=lambda r: r[3]['sharpe'], reverse=True)

    print("Top 8 TRAIN configs:")
    print(f"{'lookbk':>6}{'topN':>5}{'freq':>5}   train metrics")
    for lb, n, fq, m, _ in rows[:8]:
        print(f"{lb:>6}{n:>5}{fq:>5}   {fmt(m)}")

    lb, n, fq, _, port = rows[0]
    print(f"\n[*] Best-on-train: lookback={lb} top_n={n} freq={fq}. Out-of-sample check:\n")
    for label, idx in [('TRAIN', train), ('TEST', test), ('VAULT', vault), ('FULL', px.index)]:
        s = metrics(port.reindex(idx).fillna(0.0)); b = bh_spy(px, idx)
        verdict = 'BEATS SPY' if s['sharpe'] > b['sharpe'] else 'loses to SPY'
        print(f"{label:6} STRAT {fmt(s)}")
        print(f"{'':6} SPY   {fmt(b)}   <- {verdict} on Sharpe\n")


if __name__ == '__main__':
    main()
