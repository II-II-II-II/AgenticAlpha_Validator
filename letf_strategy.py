#!/usr/bin/env python3
"""
Leveraged-ETF tactical ensemble — faithful CORE replica of the user's bot.
Four sub-models, 25% each, weights SUMMED into one net portfolio. Regime-gated by
SMA (20/200/202) + RSI, exactly as described. Execution is MOO: the signal uses data
through today's CLOSE and the position earns the next OPEN -> following-OPEN return
(no intrabar look-ahead, mirrors a Market-On-Open order).

Honest benchmark for a 3x strategy is BUY-AND-HOLD TQQQ, not SPY (comparing leverage to
unlevered SPY is rigged). SPY shown only for reference.

Usage: python letf_strategy.py --start 2024-06-21 --end 2026-06-21
"""
import argparse
import numpy as np, pandas as pd

REF = 'QQQ'        # regime reference (the underlying we read trend/RSI from)
COST = 5 / 1e4
TRADING_DAYS = 252


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


def fmt(m):
    return (f"tot {m['total']*100:+7.1f}%  CAGR {m['cagr']*100:+6.1f}%  "
            f"Sharpe {m['sharpe']:+5.2f}  maxDD {m['maxdd']*100:6.1f}%  n={m['n']}")


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_weights(px, sma_bull=200, sma_slow=202, rsi_lo=30, rsi_bear=50,
                  s2_mode='rsi', dip_pct=0.02, dip_hold=5, crash_floor=0.05):
    """Four transparent sub-models, each 25%, returning weights on TQQQ/SQQQ/TLT/(cash).
    Summed into a net target. Mirrors 'merge overlapping picks' design.

    s2_mode controls the dip-buy sleeve (S2):
      'rsi' (BASELINE) — dip-buy TQQQ when RSI < rsi_lo, regardless of trend.
      'dip' (VALIDATED) — buy TQQQ after a sharp -dip_pct down DAY, ONLY above trend, NOT a crash,
                          and hold the sleeve dip_hold days (the bounce window)."""
    q = px[REF]
    s200 = q.rolling(sma_bull).mean()
    s_slow = q.rolling(sma_slow).mean()
    r = rsi(q, 14)
    cols = px.columns
    W = pd.DataFrame(0.0, index=px.index, columns=cols)

    bull = q > s200
    bull_slow = q > s_slow
    # T10 — trend-long: in bull, hold TQQQ; else cash
    W['TQQQ'] += np.where(bull, 0.25, 0.0)
    # T11 — slower trend, bear->bonds: in bull(slow) TQQQ, else TLT
    W['TQQQ'] += np.where(bull_slow, 0.25, 0.0)
    W['TLT'] += np.where(~bull_slow, 0.25, 0.0)
    # S2 — dip-buy sleeve (mode-dependent)
    if s2_mode == 'dip':
        qd = q.pct_change()
        trig = bull & (qd <= -dip_pct) & (qd > -crash_floor)   # sharp modest dip, above trend, not a crash
        active = trig.rolling(dip_hold).max().fillna(0) > 0     # hold the bounce window
        W['TQQQ'] += np.where(active & bull, 0.25, 0.0)         # require still above trend while held
    else:
        W['TQQQ'] += np.where(r < rsi_lo, 0.25, 0.0)            # BASELINE: RSI oversold, any trend
    # S3 — defensive short: in bear AND momentum still falling (RSI<rsi_bear) -> SQQQ, else cash
    W['SQQQ'] += np.where((~bull) & (r < rsi_bear), 0.25, 0.0)

    W[q.rolling(max(sma_bull, sma_slow)).mean().isna()] = 0.0  # warmup
    return W


def run(px, **kw):
    """MOO: weights decided at close[t] earn open[t+1]->open[t+2]."""
    W = build_weights(px, **kw)
    op = px_open  # set globally below
    oret = op.shift(-1) / op - 1.0           # oret[t] = open[t+1]/open[t]-1
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    # signal W[t-1] (decided at close t-1) earns open[t]->open[t+1] = oret[t]; no look-ahead
    port = (W.shift(1) * oret).sum(axis=1) - turn.shift(1).fillna(0) * COST
    return port.fillna(0.0)


def main():
    global px_open
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-06-21')
    ap.add_argument('--end', default='2026-06-21')
    a = ap.parse_args()

    raw = pd.read_parquet('datalake/etf_daily.parquet')
    px = raw.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    px_open = raw.pivot(index='Date', columns='Ticker', values='Open').sort_index()
    # keep full history for warmup, slice for reporting
    full_port = run(px)
    mask = (px.index >= pd.Timestamp(a.start)) & (px.index <= pd.Timestamp(a.end))
    idx = px.index[mask]
    cret = px.pct_change()

    print(f"[*] LETF ensemble | {idx[0].date()} -> {idx[-1].date()} | MOO exec | 5bps cost\n")
    s = metrics(full_port.reindex(idx).fillna(0.0))
    bt = metrics(cret['TQQQ'].reindex(idx).fillna(0.0))   # honest benchmark
    bs = metrics(cret['SPY'].reindex(idx).fillna(0.0))    # reference
    print(f"{'STRATEGY':16}{fmt(s)}")
    print(f"{'buy-hold TQQQ':16}{fmt(bt)}   <- the HONEST benchmark (3x)")
    print(f"{'buy-hold SPY':16}{fmt(bs)}   <- reference only\n")
    v1 = 'BEATS' if s['sharpe'] > bt['sharpe'] else 'loses to'
    print(f"=> Strategy {v1} buy-hold TQQQ on Sharpe ({s['sharpe']:+.2f} vs {bt['sharpe']:+.2f}); "
          f"maxDD {s['maxdd']*100:.0f}% vs {bt['maxdd']*100:.0f}%\n")

    # Worst drawdown window (the April-2025 tariff selloff falls in here)
    eq = (1 + full_port.reindex(idx).fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    print(f"[*] Strategy worst day in window: {full_port.reindex(idx).min()*100:.1f}% | "
          f"deepest drawdown: {dd.min()*100:.1f}% on {dd.idxmin().date()}")

    # Parameter sensitivity: is 200 vs 202 vs 195 SMA fragile?
    print("\n[*] SMA sensitivity (overfit check — robust = numbers barely move):")
    for sb in (195, 200, 202, 210):
        p = run(px, sma_bull=sb, sma_slow=sb + 2)
        m = metrics(p.reindex(idx).fillna(0.0))
        print(f"    SMA={sb:>3}  tot {m['total']*100:+6.1f}%  Sharpe {m['sharpe']:+.2f}  maxDD {m['maxdd']*100:6.1f}%")


if __name__ == '__main__':
    main()
