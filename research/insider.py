#!/usr/bin/env python3
"""
insider.py — SEC EDGAR insider-transaction fetcher (Form 3/4/5). Open-market BUYS (code P) only.

Uses the SEC's FREE structured quarterly "Insider Transactions Data Sets" (flat TSVs — no XML).
Emits one record per non-derivative code-P purchase, joined to the issuer ticker + the FILING date
(the date the PUBLIC learns — used for no-look-ahead entry) + the insider's role. Quarterly zips
are cached to datalake/insider_cache/.

SEC fair-access policy requires a descriptive User-Agent with contact; reachable via plain requests
(unlike FRED, EDGAR does not tarpit a normal UA — but a browser-spoof UA is still bad manners here).

Usage (self-test):  python research/insider.py 2025q4
"""
import os, csv, zipfile, io, sys, requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'datalake', 'insider_cache')
UA = {'User-Agent': 'qc-research insider-study contact@qc-research.local'}
BASE = 'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets'
_MON = {m: i for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}


def _date(s):
    """'31-DEC-2025' -> '2025-12-31'. '' -> None."""
    s = (s or '').strip()
    if not s:
        return None
    try:
        d, m, y = s.split('-')
        return f"{int(y):04d}-{_MON[m.upper()]:02d}-{int(d):02d}"
    except Exception:
        return None


def quarters(start='2022q1', end='2026q1'):
    sy, sq = int(start[:4]), int(start[5]); ey, eq = int(end[:4]), int(end[5])
    out, y, q = [], sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"{y}q{q}"); q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def _zip(q):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{q}_form345.zip")
    if not os.path.exists(p) or os.path.getsize(p) < 1000:
        r = requests.get(f"{BASE}/{q}_form345.zip", headers=UA, timeout=120)
        r.raise_for_status()
        open(p, 'wb').write(r.content)
    return p


def buys(qs, min_value=0.0):
    """Code-P open-market BUYS across quarters. Returns list of dict(symbol, filing_date, trans_date,
    shares, price, value, role, title, accession). filing_date = public-knowledge date (no look-ahead)."""
    events = []
    for q in qs:
        z = zipfile.ZipFile(_zip(q))

        def rd(name):
            with z.open(name) as fh:
                yield from csv.DictReader(io.TextIOWrapper(fh, encoding='utf-8', errors='replace'), delimiter='\t')

        sub = {r['ACCESSION_NUMBER']: r for r in rd('SUBMISSION.tsv')}
        role = {}
        for r in rd('REPORTINGOWNER.tsv'):
            role.setdefault(r['ACCESSION_NUMBER'], r)     # first reporting owner
        for r in rd('NONDERIV_TRANS.tsv'):
            if r.get('TRANS_CODE') != 'P' or r.get('TRANS_ACQUIRED_DISP_CD') != 'A':
                continue                                   # open-market purchase, shares acquired
            s = sub.get(r['ACCESSION_NUMBER'])
            if not s:
                continue
            sym = (s.get('ISSUERTRADINGSYMBOL') or '').strip().upper()
            if not sym or sym in ('NONE', 'N/A', ''):
                continue
            try:
                sh, pr = float(r['TRANS_SHARES']), float(r['TRANS_PRICEPERSHARE'])
            except Exception:
                continue
            if sh <= 0 or pr <= 0:
                continue
            val = sh * pr
            if val < min_value:
                continue
            ro = role.get(r['ACCESSION_NUMBER'], {})
            fd = _date(s.get('FILING_DATE'))
            if not fd:
                continue
            events.append(dict(symbol=sym, name=(s.get('ISSUERNAME') or '').strip(),
                               filing_date=fd, trans_date=_date(r.get('TRANS_DATE')),
                               shares=sh, price=pr, value=val,
                               role=ro.get('RPTOWNER_RELATIONSHIP', ''), title=ro.get('RPTOWNER_TITLE', ''),
                               accession=r['ACCESSION_NUMBER']))
    return events


if __name__ == '__main__':
    qs = sys.argv[1:] or ['2025q4']
    ev = buys(qs)
    print(f"{qs}: {len(ev)} code-P open-market buys")
    tot = sum(e['value'] for e in ev)
    print(f"  total $ bought: ${tot/1e9:.2f}B | median ${sorted(e['value'] for e in ev)[len(ev)//2]:,.0f}")
    import collections
    c = collections.Counter(e['symbol'] for e in ev)
    print(f"  unique tickers: {len(c)} | most-bought: {c.most_common(5)}")
    for e in ev[:4]:
        print(f"   {e['symbol']:8} filed {e['filing_date']} | ${e['value']:>12,.0f} | {e['title'][:30]}")
