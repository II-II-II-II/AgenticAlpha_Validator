#!/usr/bin/env python3
"""
macro.py — the objective macro backdrop: a "zoom out" readout of what actually moves the market.

Two data families, deliberately kept SEPARATE because they behave completely differently:

  MARKET-BASED (real-time, forward-looking) — via Yahoo, the same feeds the rest of the repo uses.
    Treasury yields + the yield curve, the dollar (DXY), market-implied inflation, oil, credit (HYG).
    This is the market's LIVE vote on the fundamentals; it front-runs the hard prints and is the
    more useful of the two for a trader. Works everywhere.

  HARD DATA (lagging, low-frequency) — via FRED (St. Louis Fed, free, the gold standard).
    Headline/core CPI, core PCE (the Fed's preferred gauge), unemployment, nonfarm payrolls,
    weekly jobless claims, the fed funds rate, consumer sentiment. These print monthly/quarterly
    with a delay and are largely PRICED IN by the time they release — they define the REGIME/
    backdrop; they are NOT a timing signal.

HONEST FRAMING (read this before adding "just one more indicator"):
  More metrics = better situational awareness, NOT a better timing edge. This repo has already
  killed news-sentiment and a 15-minute direction probe on exactly this premise. Macro here is for
  zooming out and understanding the backdrop after a rough stretch — not for predicting tomorrow.

ENV NOTE: FRED is firewalled from the sandboxed agent environment (it times out), but works fine
from a normal machine. market_macro() works everywhere; hard_macro() degrades gracefully to an
empty list if FRED is unreachable, so callers (e.g. the dashboard) never break.

Usage:  python research/macro.py
"""
import csv, io, sys, os, json
from datetime import datetime
import requests

UA = {'User-Agent': 'Mozilla/5.0'}
YQ = 'https://query1.finance.yahoo.com/v8/finance/chart/{}'
FREDCSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv'


# ---------------- MARKET-BASED (Yahoo — real-time, forward-looking) ----------------
def _yahoo_closes(ticker, rng='1y'):
    """Daily closes for a ticker (most recent last). [] on failure."""
    try:
        r = requests.get(YQ.format(ticker), params={'range': rng, 'interval': '1d'},
                         headers=UA, timeout=12)
        res = r.json()['chart']['result'][0]
        return [x for x in res['indicators']['quote'][0]['close'] if x is not None]
    except Exception:
        return []


def _norm_yield(v):
    """Yahoo yield indices (^TNX/^FVX/^IRX) are sometimes quoted x10 (42.5 == 4.25%)."""
    return v / 10 if v and v > 20 else v


def _ago(series, n):
    return series[-n] if len(series) >= n else (series[0] if series else None)


def market_macro():
    """List of market-based macro rows: {label, value, trend, tone, note}.
    tone is from a risk-asset holder's view: 'good' = tailwind, 'bad' = headwind, 'neutral' = mixed."""
    rows = []
    t10, t5, irx = _yahoo_closes('^TNX'), _yahoo_closes('^FVX'), _yahoo_closes('^IRX')
    if t10 and irx:
        y10, yshort = _norm_yield(t10[-1]), _norm_yield(irx[-1])
        y10_yr = _norm_yield(_ago(t10, 252))
        rising = y10 > y10_yr
        rows.append(dict(
            label='10yr Treasury yield', value=f'{y10:.2f}%',
            trend=f'{y10_yr:.2f}% yr ago → {"rising" if rising else "falling"}',
            tone='bad' if rising else 'good',
            note='the single macro variable that most directly pressures long-duration / leveraged tech'))
        curve = y10 - yshort
        inv = curve < 0
        rows.append(dict(
            label='Yield curve (10yr − 13wk)', value=f'{curve:+.2f}',
            trend='INVERTED' if inv else 'positive / normal',
            tone='bad' if inv else 'good',
            note='inversion is the classic recession lead-indicator (12–18mo lag)'))
    dxy = _yahoo_closes('DX-Y.NYB')
    if dxy:
        chg = (dxy[-1] / dxy[0] - 1) * 100
        rows.append(dict(
            label='US Dollar (DXY)', value=f'{dxy[-1]:.1f}',
            trend=f'{chg:+.1f}% / yr',
            tone='bad' if chg > 3 else ('good' if chg < -3 else 'neutral'),
            note='strong $ = tighter global conditions + safe-haven demand; a drag on multinationals'))
    tip, ief = _yahoo_closes('TIP'), _yahoo_closes('IEF')
    if tip and ief and len(tip) >= 63 and len(ief) >= 63:
        r_now, r_3mo = tip[-1] / ief[-1], tip[-63] / ief[-63]
        chg = (r_now / r_3mo - 1) * 100
        rising = chg > 0.3
        rows.append(dict(
            label='Implied inflation (TIP/IEF)', value=f'{chg:+.1f}% / 3mo',
            trend='expectations rising' if rising else 'contained / easing',
            tone='bad' if rising else 'good',
            note='rough market-priced inflation proxy; the real-time read the CPI print lags'))
    oil = _yahoo_closes('CL=F')
    if oil:
        chg = (oil[-1] / _ago(oil, 21) - 1) * 100
        rows.append(dict(
            label='Crude oil (WTI)', value=f'${oil[-1]:.0f}',
            trend=f'{chg:+.0f}% / 1mo',
            tone='bad' if chg > 8 else ('good' if chg < -8 else 'neutral'),
            note='a sharp spike feeds headline inflation and squeezes the consumer'))
    cr = _credit_spread()
    if cr:
        rows.append(cr)
    return rows


def _credit_spread():
    """HY credit-spread gauge from ETF proxies: HYG (junk) vs IEF (Treasuries), 5y percentile.
    Lower ratio = wider spread = more default stress. Reports where today sits in the 5y range.
    Credit LEADS equities — junk cracking while stocks hold is the real crisis tell (cf. 2007)."""
    hyg, ief = _yahoo_closes('HYG', '5y'), _yahoo_closes('IEF', '5y')
    if not hyg or not ief:
        return None
    n = min(len(hyg), len(ief))
    ratio = [hyg[-n + i] / ief[-n + i] for i in range(n)]
    now = ratio[-1]
    tight_pct = sum(1 for x in ratio if x <= now) / len(ratio) * 100   # 100 = tightest/calmest in 5y
    r1 = ratio[-21] if len(ratio) >= 21 else ratio[0]
    chg = (now / r1 - 1) * 100
    label = 'tight / calm' if tight_pct > 70 else ('WIDE / stressed' if tight_pct < 30 else 'mid-range')
    return dict(
        label='HY credit spread (proxy)', value=f'{tight_pct:.0f}th pct',
        trend=f'{label} · {"widening" if chg < -1 else "stable/tightening"} 1mo',
        tone='good' if tight_pct > 70 else ('bad' if tight_pct < 30 else 'neutral'),
        note='junk vs Treasuries; 100th pct = tightest in 5y (calm but complacent), 0th = widest (real stress)')


# ---------------- HARD DATA (FRED — lagging, defines the regime) ----------------
FRED_UA = {'User-Agent': 'python-requests/2.31'}   # FRED's CDN tarpits a bare 'Mozilla/5.0' (bot sig)
                                                   # -> ReadTimeout; a plain non-browser UA sails through.


def _fred(series):
    """[(date, value)] for a FRED series, oldest→newest. [] on failure/unreachable."""
    try:
        r = requests.get(FREDCSV, params={'id': series}, headers=FRED_UA, timeout=12)
        rows = list(csv.reader(io.StringIO(r.text)))[1:]
        return [(d, float(v)) for d, v in rows if v not in ('.', '')]
    except Exception:
        return []


def _yoy(data):
    """Year-over-year % change of a monthly FRED series (latest vs 12 obs back)."""
    if len(data) < 13:
        return None
    return (data[-1][1] / data[-13][1] - 1) * 100


def hard_macro():
    """List of FRED hard-data rows: {label, value, trend, tone, asof}. Empty if FRED unreachable."""
    rows = []
    cpi = _fred('CPIAUCSL')          # headline CPI
    if cpi:
        y = _yoy(cpi)
        rows.append(dict(label='Inflation — CPI', value=f'{y:+.1f}% YoY', asof=cpi[-1][0],
                         trend='above 2% target' if y > 2.3 else 'near target',
                         tone='bad' if y > 3 else ('neutral' if y > 2.3 else 'good')))
    core = _fred('CPILFESL')         # core CPI (ex food/energy)
    if core:
        y = _yoy(core)
        rows.append(dict(label='Core CPI', value=f'{y:+.1f}% YoY', asof=core[-1][0],
                         trend='sticky' if y > 3 else 'cooling',
                         tone='bad' if y > 3.3 else ('neutral' if y > 2.5 else 'good')))
    pce = _fred('PCEPILFE')          # core PCE — the Fed's preferred gauge
    if pce:
        y = _yoy(pce)
        rows.append(dict(label='Core PCE (Fed gauge)', value=f'{y:+.1f}% YoY', asof=pce[-1][0],
                         trend="the Fed's actual target metric",
                         tone='bad' if y > 3 else ('neutral' if y > 2.3 else 'good')))
    un = _fred('UNRATE')
    if un:
        cur, prev = un[-1][1], (un[-13][1] if len(un) >= 13 else un[0][1])
        rows.append(dict(label='Unemployment', value=f'{cur:.1f}%', asof=un[-1][0],
                         trend=f'{prev:.1f}% yr ago → {"rising" if cur > prev + 0.2 else "steady/falling"}',
                         tone='bad' if cur > prev + 0.5 else 'good'))
    pay = _fred('PAYEMS')            # nonfarm payrolls (level, thousands)
    if pay and len(pay) >= 2:
        chg = pay[-1][1] - pay[-2][1]
        rows.append(dict(label='Nonfarm payrolls', value=f'{chg:+,.0f}k jobs', asof=pay[-1][0],
                         trend='monthly change',
                         tone='good' if chg > 100 else ('neutral' if chg > 0 else 'bad')))
    ic = _fred('ICSA')              # weekly initial jobless claims
    if ic:
        cur = ic[-1][1]
        mo_ago = ic[-5][1] if len(ic) >= 5 else ic[0][1]
        rows.append(dict(label='Jobless claims (weekly)', value=f'{cur/1000:.0f}k', asof=ic[-1][0],
                         trend=f'{"rising" if cur > mo_ago * 1.05 else "steady/falling"} vs a month ago',
                         tone='bad' if cur > mo_ago * 1.1 else 'good'))
    hy = _fred('BAMLH0A0HYM2')       # ICE BofA US High-Yield OAS (the authoritative HY spread, %)
    if hy:
        bps = hy[-1][1] * 100
        rows.append(dict(label='HY credit spread (OAS)', value=f'{bps:.0f} bps', asof=hy[-1][0],
                         trend='tight' if bps < 400 else ('elevated' if bps < 600 else 'stressed'),
                         tone='good' if bps < 450 else ('neutral' if bps < 600 else 'bad')))
    ig = _fred('BAMLC0A0CM')         # ICE BofA US Corporate (investment-grade) OAS (%)
    if ig:
        bps = ig[-1][1] * 100
        rows.append(dict(label='IG credit spread (OAS)', value=f'{bps:.0f} bps', asof=ig[-1][0],
                         trend='tight' if bps < 120 else ('elevated' if bps < 170 else 'stressed'),
                         tone='good' if bps < 130 else ('neutral' if bps < 170 else 'bad')))
    ff = _fred('FEDFUNDS')
    if ff:
        rows.append(dict(label='Fed funds rate', value=f'{ff[-1][1]:.2f}%', asof=ff[-1][0],
                         trend='policy rate', tone='neutral'))
    sent = _fred('UMCSENT')
    if sent:
        cur = sent[-1][1]
        prev = sent[-13][1] if len(sent) >= 13 else sent[0][1]
        rows.append(dict(label='Consumer sentiment (UMich)', value=f'{cur:.0f}', asof=sent[-1][0],
                         trend=f'{prev:.0f} yr ago → {"up" if cur > prev else "down"}',
                         tone='good' if cur > prev else 'bad'))
    return rows


def snapshot():
    """Combined dict: {market, hard, fred_ok, credit_regime}. Persist with save_snapshot()."""
    hard = hard_macro()
    return dict(market=market_macro(), hard=hard, fred_ok=bool(hard), credit_regime=credit_regime())


# ---------------- credit regime tag (validated on 2007-2026: coincident classifier, not a timer) ----
def credit_regime():
    """The 'what KIND of selloff' tag, from the historical analysis: credit spreads and equity
    declines are strongly CO-INCIDENT (weekly corr ~-0.74) but credit does NOT lead. Its real value
    is classifying the regime:
      CALM              — HY spreads tight, no stress.
      WATCH             — spreads softening but no selloff yet (the early tell if it continues).
      CREDIT-QUIET      — stocks falling but credit shrugging => rates/valuation selloff, historically
                          shallower & recoverable (2022 -25%, 2025 -19%, 2018 template).
      CREDIT-CONFIRMED  — stocks falling AND spreads blowing out => systemic/deep (2008 -55%, 2020 -34%).
    Transparent heuristic thresholds; this is CONTEXT, not a prediction."""
    hyg, ief, spy = _yahoo_closes('HYG', '5y'), _yahoo_closes('IEF', '5y'), _yahoo_closes('SPY', '3mo')
    if not (hyg and ief and spy):
        return dict(label='unknown', tone='neutral', detail='data unavailable')
    n = min(len(hyg), len(ief))
    ratio = [hyg[-n + i] / ief[-n + i] for i in range(n)]
    now = ratio[-1]
    tight_pct = sum(1 for x in ratio if x <= now) / len(ratio) * 100      # 100 = tightest/calmest in 5y
    widening = (now / (ratio[-21] if len(ratio) >= 21 else ratio[0]) - 1) < -0.01
    spx_1mo = (spy[-1] / spy[-21] - 1) * 100 if len(spy) >= 21 else 0.0
    selloff = spx_1mo <= -3
    if tight_pct < 25 or (selloff and widening and tight_pct < 55):
        lab, tone, det = ('CREDIT-CONFIRMED STRESS', 'bad',
            'spreads wide/widening WITH stocks down — systemic template (2008/2020). The one to fear.')
    elif selloff:
        lab, tone, det = ('CREDIT-QUIET SELLOFF', 'neutral',
            'stocks down but credit calm — rates/valuation template (2022/2025), historically recoverable.')
    elif widening and tight_pct < 60:
        lab, tone, det = ('WATCH — spreads widening', 'neutral',
            'no selloff yet but credit softening; the early tell if it keeps going.')
    else:
        lab, tone, det = ('CALM — credit tight', 'good',
            'HY spreads tight, no stress; if stocks fall from here the base rate favors a recoverable dip.')
    return dict(label=lab, tone=tone, detail=det, hy_percentile=round(tight_pct),
                spx_1mo=round(spx_1mo, 1), widening=bool(widening))


# ---------------- persistence: one store, shared by the user and the agents ----------------
STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datalake')


def save_snapshot(snap=None):
    """Persist to datalake/ so BOTH the dashboard and the Oracle agents read the SAME data, offline.
    macro_latest.json = current full snapshot; macro_history.jsonl = one compact row per day (trendable)."""
    snap = snap or snapshot()
    os.makedirs(STORE, exist_ok=True)
    snap = dict(snap, date=datetime.now().strftime('%Y-%m-%d'),
                ts=datetime.now().isoformat(timespec='seconds'))
    json.dump(snap, open(os.path.join(STORE, 'macro_latest.json'), 'w'), indent=2)
    reg = snap.get('credit_regime') or {}
    row = dict(date=snap['date'], credit_regime=reg.get('label'), hy_percentile=reg.get('hy_percentile'),
               **{r['label']: r['value'] for r in snap.get('market', [])},
               **{r['label']: r['value'] for r in snap.get('hard', [])})
    hp = os.path.join(STORE, 'macro_history.jsonl')
    lines = [l for l in open(hp).read().splitlines() if l.strip()] if os.path.exists(hp) else []
    if lines:
        try:
            if json.loads(lines[-1]).get('date') == snap['date']:
                lines.pop()                    # replace today's row -> one row per day
        except Exception:
            pass
    lines.append(json.dumps(row))
    open(hp, 'w').write('\n'.join(lines) + '\n')
    return snap


def latest():
    """Read the stored snapshot (offline, no network) — the agents' fast path. {} if never saved."""
    try:
        return json.load(open(os.path.join(STORE, 'macro_latest.json')))
    except Exception:
        return {}


def compact(snap=None):
    """Small flat dict for injecting into agent prompts (kept short on purpose)."""
    snap = snap or latest()
    if not snap:
        return {}
    out = {'credit_regime': (snap.get('credit_regime') or {}).get('label')}
    for r in snap.get('market', []) + snap.get('hard', []):
        out[r['label']] = f"{r['value']} · {r.get('trend', '')}".strip(' ·')
    return out


# ---------------- CLI ----------------
def _print(rows, hard=False):
    for r in rows:
        tone = {'good': '🟢', 'bad': '🔴', 'neutral': '⚪'}.get(r['tone'], ' ')
        asof = f"  [{r['asof']}]" if hard and r.get('asof') else ''
        print(f"  {tone} {r['label']:<30} {r['value']:>16}   {r['trend']}{asof}")
        if not hard and r.get('note'):
            print(f"       └ {r['note']}")


def main():
    save = '--save' in sys.argv
    snap = snapshot()
    print("\n" + "=" * 78)
    print("  MACRO BACKDROP — zooming out")
    print("=" * 78)
    reg = snap['credit_regime']
    dot = {'good': '🟢', 'bad': '🔴', 'neutral': '⚪'}.get(reg['tone'], ' ')
    print(f"\n  CREDIT REGIME: {dot} {reg['label']}")
    print(f"     {reg['detail']}")
    print(f"     HY spread {reg.get('hy_percentile')}th pct (100=tightest) · SPY 1mo {reg.get('spx_1mo')}%")
    print("\nMARKET-BASED (real-time, forward-looking — the market's live vote):\n")
    _print(snap['market'])
    print("\nHARD DATA (FRED — lagging, defines the regime; blocked in agent sandbox):\n")
    if snap['hard']:
        _print(snap['hard'], hard=True)
    else:
        print("  FRED unreachable from here (fine on your own machine — just run this there).")
    print("\n" + "-" * 78)
    print("  Reminder: this is CONTEXT, not a timing signal. Macro is already priced in;")
    print("  it tells you the regime, not tomorrow's move.")
    print("-" * 78)
    if save:
        save_snapshot(snap)
        print(f"\n  saved -> datalake/macro_latest.json + macro_history.jsonl")
    print()


if __name__ == '__main__':
    main()
