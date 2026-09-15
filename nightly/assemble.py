#!/usr/bin/env python3
"""
assemble.py — build the complete context document for the Sunday-night local-model report.

WHY THIS EXISTS: the report model runs disk-offloaded at well under 1 tok/s, so it gets
exactly ONE shot — one context in, one report out, zero tool calls. Everything it could
possibly need has to be gathered here, by fast ordinary Python, before it is invoked.

THE BINDING CONSTRAINT IS CONTEXT SIZE, NOT OUTPUT SIZE. Per nightly/bench.py, at the
pessimistic bound an 8k context costs ~29 min of prefill and still leaves room for a
3,000-token report; a 32k context costs ~2 hours and starves the generation. So this
module enforces a hard token budget and trims the lowest-value section (news) to fit.

Usage:
  python nightly/assemble.py                 # print the context document
  python nightly/assemble.py -o ctx.txt      # write it
  python nightly/assemble.py --budget 8000   # change the token ceiling
"""
import os, sys, json, argparse, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research'))

CHARS_PER_TOKEN = 3.8          # conservative for English prose + tables
DEFAULT_BUDGET = 8000


def est_tokens(s):
    return int(len(s) / CHARS_PER_TOKEN)


def _load_portfolio():
    with open(os.path.join(HERE, 'portfolio.json')) as f:
        return json.load(f)


# ---------------------------------------------------------------- price data
def _history(tickers, period='3mo'):
    from yahooquery import Ticker
    import pandas as pd
    h = Ticker(' '.join(tickers)).history(period=period, interval='1d')
    out = {}
    for t in tickers:
        try:
            s = h.loc[t]['adjclose']
            out[t] = [(str(i)[:10], float(v)) for i, v in zip(s.index, s.values)]
        except Exception:
            continue
    return out


def _pct(series, n):
    """% change over the last n sessions."""
    if not series or len(series) <= n:
        return None
    return (series[-1][1] / series[-1 - n][1] - 1) * 100


# ---------------------------------------------------------------- blocks
def portfolio_block(pf):
    tks = [h['ticker'] for h in pf['holdings']]
    hist = _history(tks)
    rows, total = [], 0.0
    for h in pf['holdings']:
        s = hist.get(h['ticker'])
        if not s:
            rows.append((h, None, None, None, None)); continue
        px = s[-1][1]; val = px * h['shares']; total += val
        rows.append((h, px, val, _pct(s, 5), _pct(s, 21)))

    L = [f"## PORTFOLIO — {pf['account']}",
         f"Established {pf['established']} · next scheduled review {pf['next_scheduled_review']}",
         "",
         f"{'ticker':<7}{'role':<24}{'price':>9}{'value':>11}{'weight':>8}{'target':>8}{'drift':>8}{'1wk':>8}{'1mo':>8}"]
    for h, px, val, w1, m1 in rows:
        if px is None:
            L.append(f"{h['ticker']:<7}{h['role']:<24}{'(price unavailable)':>52}"); continue
        wt = val / total * 100
        drift = wt - h['target_pct']
        L.append(f"{h['ticker']:<7}{h['role']:<24}{px:>9.2f}{val:>11,.0f}{wt:>7.1f}%"
                 f"{h['target_pct']:>7}%{drift:>+7.1f}{w1:>7.1f}%{m1:>7.1f}%")
    L.append(f"{'TOTAL':<7}{'':<24}{'':>9}{total:>11,.0f}")

    # band check — the ONLY thing that can trigger an action
    band = pf.get('band_pct_relative', 25) / 100.0
    breaches = [f"{h['ticker']} at {val/total*100:.1f}% vs target {h['target_pct']}%"
                for h, px, val, _, _ in rows if px
                and abs(val / total * 100 - h['target_pct']) > h['target_pct'] * band]
    L += ["", f"BAND CHECK (±{pf.get('band_pct_relative',25)}% relative): "
              + ("NO BREACHES — no action required" if not breaches else "; ".join(breaches))]
    return "\n".join(L)


def regime_block():
    L = ["## MARKET REGIME"]
    try:
        hist = _history(['QQQ', 'SPY'], period='2y')
        q = [v for _, v in hist['QQQ']]
        sma150 = sum(q[-150:]) / 150
        sma150_prev = sum(q[-155:-5]) / 150
        L.append(f"QQQ {q[-1]:.2f} vs 150d SMA {sma150:.2f} = {(q[-1]/sma150-1)*100:+.2f}% "
                 f"({'ABOVE — risk-on' if q[-1] > sma150 else 'BELOW — gate triggered'}), "
                 f"150d {'rising' if sma150 > sma150_prev else 'FALLING'}")
        L.append(f"Distance to gate: {(sma150/q[-1]-1)*100:+.1f}%")
    except Exception as e:
        L.append(f"(regime unavailable: {e})")
    try:
        import macro
        c = macro.compact()
        if c:
            L.append("")
            for k, v in c.items():
                L.append(f"  {k}: {v}")
    except Exception as e:
        L.append(f"(macro block unavailable: {e})")
    return "\n".join(L)


SECTORS = {'XLK': 'technology', 'XLF': 'financials', 'XLV': 'healthcare', 'XLE': 'energy',
           'XLI': 'industrials', 'XLY': 'cons disc', 'XLP': 'cons staples', 'XLU': 'utilities',
           'XLB': 'materials', 'XLRE': 'real estate', 'XLC': 'communications'}


def market_block():
    hist = _history(list(SECTORS) + ['SPY', 'VT', 'GLD', 'TLT'])
    rows = []
    for t, name in SECTORS.items():
        s = hist.get(t)
        if s:
            rows.append((name, t, _pct(s, 5), _pct(s, 21)))
    rows.sort(key=lambda r: -(r[2] or -99))
    L = ["## SECTOR PERFORMANCE (what led and lagged)", f"{'sector':<16}{'1wk':>8}{'1mo':>9}"]
    for name, t, w, m in rows:
        L.append(f"{name:<16}{w:>7.1f}%{m:>8.1f}%")
    L.append("")
    for t, lbl in (('SPY', 'S&P 500'), ('VT', 'world equity'), ('GLD', 'gold'), ('TLT', 'long treasuries')):
        s = hist.get(t)
        if s:
            L.append(f"{lbl:<16}{_pct(s,5):>7.1f}%{_pct(s,21):>8.1f}%")
    return "\n".join(L)


def news_block(limit=25):
    try:
        import oracle_forecast as of
        items = of.get_news()
    except Exception as e:
        return f"## NEWS\n(unavailable: {e})"
    L = ["## NEWS — headlines from the period (source: Alpaca)"]
    for n in items[:limit]:
        ts = (n.get('created') or '')[:16].replace('T', ' ')
        tk = ','.join(n.get('tickers', [])[:4])
        L.append(f"- [{ts}] {n.get('headline','')[:150]}" + (f"  ({tk})" if tk else ""))
    return "\n".join(L)


def guardrails_block(pf):
    return ("## STANDING CONSTRAINTS (context for the analyst — not to be re-litigated)\n"
            f"- This is a deliberately passive, set-and-forget portfolio. Next scheduled "
            f"review is {pf['next_scheduled_review']}.\n"
            f"- Rebalancing happens ONLY on a band breach (±{pf.get('band_pct_relative',25)}% "
            f"relative). Dividends reinvest automatically.\n"
            f"- CRASH PROTOCOL: {pf['crash_protocol']}\n"
            "- The owner has a documented history of selling at lows and buying at highs. "
            "On 2026-07-29 he wanted to liquidate at the exact bottom; four days later the "
            "index was 6.7% higher. This report exists to explain, never to prompt action.")


# ---------------------------------------------------------------- assemble
def assemble(budget=DEFAULT_BUDGET):
    pf = _load_portfolio()
    today = dt.date.today()
    header = (f"# MARKET & PORTFOLIO CONTEXT — week ending {today.isoformat()}\n"
              f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by assemble.py.\n"
              f"All figures below are pre-computed facts. Do not estimate or infer prices.\n")

    fixed = [header, portfolio_block(pf), regime_block(), market_block(), guardrails_block(pf)]
    used = sum(est_tokens(b) for b in fixed) + 100
    remaining = max(0, budget - used)

    # news is the elastic section — trim it to whatever budget is left
    n = 25
    while n > 3:
        nb = news_block(n)
        if est_tokens(nb) <= remaining:
            break
        n -= 3
    else:
        nb = news_block(3)

    doc = "\n\n".join(fixed[:4] + [nb, fixed[4]])
    return doc, est_tokens(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', help='write to file')
    ap.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    a = ap.parse_args()
    doc, toks = assemble(a.budget)
    if a.out:
        open(a.out, 'w').write(doc)
        print(f"wrote {a.out} — ~{toks:,} tokens (budget {a.budget:,})")
        if toks > a.budget:
            print("  !! OVER BUDGET — prefill will cost more window than planned")
    else:
        print(doc)
        print(f"\n--- ~{toks:,} tokens (budget {a.budget:,}) ---", file=sys.stderr)


if __name__ == '__main__':
    main()
