#!/usr/bin/env python3
"""
score.py — objective quality scoring for a generated market report.

The premise: "which report is better" has no ground truth, but the failure mode that
actually matters DOES. A single-shot model with no fact-checking loop fails by stating
things confidently that aren't in its source data. Since assemble.py produced that source
data, every claim is checkable.

Three tiers, hardest first:

  1. FABRICATION (fatal)   — every number in the report must trace to the context document,
                             or be arithmetic derivable from it. Unsourced figures are
                             hallucinations and they make the whole report untrustworthy.
  2. UNSUPPORTED CAUSATION — "X moved because Y" is only legitimate if Y is a headline that
                             actually appears in the context. This is THE failure mode for
                             a news-correlation report.
  3. COMPLIANCE (pass/fail)— zero action language. A beautifully written report that tells
                             you to trim a position has failed, regardless of prose quality.

Deliberately NOT included: an LLM judge. One model grading another on a task with no
ground truth just launders opinion into a number. These checks require no judgment.

Usage:
  python nightly/score.py --report r.md --context ctx.txt
  python nightly/score.py --report r.md --context ctx.txt --json
"""
import os, re, sys, json, argparse

# words that mean the report crossed from explaining into advising
ACTION_WORDS = [
    r'\byou should\b', r'\bi recommend\b', r'\brecommend(?:ed|ation)?\b', r'\bconsider (?:buying|selling|trimming|adding|rotating)\b',
    r'\bshould (?:buy|sell|trim|add|reduce|increase|rotate|exit|enter)\b',
    r'\btime to (?:buy|sell|trim|exit)\b', r'\bworth (?:buying|selling|trimming)\b',
    r'\bsuggest(?:s|ed)? (?:buying|selling|trimming|reducing|adding)\b',
    r'\btake profits?\b', r'\bcut (?:losses|exposure)\b', r'\brebalance now\b',
]
# causal connectives that assert a news->price link
CAUSAL = [r'\bbecause of\b', r'\bdriven by\b', r'\bdue to\b', r'\battributed to\b',
          r'\bon news of\b', r'\bfollowing (?:the )?(?:news|report|announcement)\b',
          r'\bin response to\b', r'\bas a result of\b', r'\bafter (?:the )?(?:news|report|announcement)\b',
          r'\bsparked by\b', r'\btriggered by\b', r'\bcaused by\b']
# hedging that makes a causal claim acceptable (we asked for hypotheses, not certainties)
HEDGES = [r'\bmay\b', r'\bmight\b', r'\blikely\b', r'\bpossibl(?:y|e)\b', r'\bappears?\b',
          r'\bsuggests?\b', r'\bconsistent with\b', r'\bcoincid(?:ed|es|ent)\b',
          r'\bplausibl(?:y|e)\b', r'\bcorrelat(?:ed|ion|es)\b', r'\bhypothes(?:is|ise|ize)\b',
          r'\bcannot be confirmed\b', r'\bno causal\b', r'\bunclear\b', r'\bpotentially\b']

# Trailing unit suffixes must NOT break extraction: the context writes "150d SMA" and
# "263 bps" while a report writes "150-day" and "263bps". Without this, identical figures
# fail to match and every one looks like a hallucination.
NUM = re.compile(r'(?<![\w.])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)'
                 r'(?:\s*(?:d|day|days|bps|bp|k|m|b|pct|%|yr|mo|wk))?(?![\d])', re.I)

# Numbers inside a conditional/hypothetical clause are the analyst proposing a threshold
# ("if spreads widen beyond 350bps"), not asserting a fact. The prompt explicitly ASKS for
# these in the "What Would Change The Picture" section, so counting them as fabrication
# punishes the model for following instructions.
HYPOTHETICAL = re.compile(
    r'\b(?:if|were|would|should|e\.?g\.?|for example|say|suppose|hypothetical|beyond|'
    r'exceed(?:s|ed)?|above|below|break(?:s)?|threshold|watch for|toward)\b|[<>≥≤]', re.I)


def _numbers(text):
    """All numeric tokens, normalised (commas stripped, sign dropped, unit suffix tolerated)."""
    out = set()
    for m in NUM.finditer(text):
        raw = m.group(1).replace(',', '')
        try:
            out.add(round(abs(float(raw)), 2))
        except ValueError:
            continue
    return out


def check_fabrication(report, context, tol=0.02):
    """Numbers ASSERTED in the report that don't trace to the context.

    Skips: small integers (list numbering, years, counts) and figures appearing in
    hypothetical clauses, which are requested by the prompt rather than fabricated."""
    ctx = _numbers(context)
    hypo = set()
    for s in _sentences(report):
        if HYPOTHETICAL.search(s):
            hypo |= _numbers(s)
    rep = _numbers(report)

    unsourced = []
    for v in sorted(rep):
        if v <= 100 and float(v).is_integer():
            continue                                   # 1..100 integers: too noisy to police
        if v in hypo:
            continue                                   # proposed threshold, not a claim
        if any(abs(v - c) <= max(tol, abs(c) * tol) for c in ctx):
            continue
        unsourced.append(v)
    return unsourced, len(rep)


def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def check_causation(report, context):
    """Causal claims, split into hedged vs asserted, and whether a headline plausibly backs them."""
    # headline vocabulary from the context's NEWS block
    news = context.split('## NEWS', 1)[-1] if '## NEWS' in context else context
    news_words = set(re.findall(r'[a-z]{5,}', news.lower()))

    hedged, asserted = [], []
    for s in _sentences(report):
        if not any(re.search(p, s, re.I) for p in CAUSAL):
            continue
        is_hedged = any(re.search(h, s, re.I) for h in HEDGES)
        # does the sentence share meaningful vocabulary with any headline?
        toks = set(re.findall(r'[a-z]{5,}', s.lower()))
        overlap = len(toks & news_words)
        rec = {'sentence': s[:180], 'news_overlap': overlap}
        (hedged if is_hedged else asserted).append(rec)
    return hedged, asserted


def check_compliance(report):
    hits = []
    for p in ACTION_WORDS:
        for m in re.finditer(p, report, re.I):
            s = report[max(0, m.start() - 60):m.end() + 60].replace('\n', ' ')
            hits.append({'pattern': p, 'context': s.strip()})
    return hits


def check_sections(report, required=('performance', 'position', 'news', 'regime')):
    low = report.lower()
    return {r: (r in low) for r in required}


def score(report, context):
    unsourced, total_nums = check_fabrication(report, context)
    hedged, asserted = check_causation(report, context)
    actions = check_compliance(report)
    sections = check_sections(report)
    toks = int(len(report) / 3.8)

    # trust score: fabrication and action language are the heavy penalties
    pts = 100
    pts -= min(50, len(unsourced) * 10)          # unsourced numbers are fatal-ish
    pts -= min(30, len(actions) * 15)            # action language breaks the safety rule
    pts -= min(15, len(asserted) * 5)            # unhedged causation
    pts -= 0 if all(sections.values()) else 5
    pts = max(0, pts)

    return dict(trust_score=pts, tokens=toks, numbers_in_report=total_nums,
                unsourced_numbers=unsourced, action_language=actions,
                causal_hedged=len(hedged), causal_asserted=asserted,
                sections=sections)


def render(s, name=''):
    L = [f"{'='*58}", f"REPORT SCORE{(' — ' + name) if name else ''}", '=' * 58]
    ok = lambda b: '✅' if b else '❌'
    L.append(f"  numbers in report        {s['numbers_in_report']}")
    L.append(f"  unsourced numbers        {len(s['unsourced_numbers'])} {ok(not s['unsourced_numbers'])}"
             + (f"  {s['unsourced_numbers'][:6]}" if s['unsourced_numbers'] else ''))
    L.append(f"  action language          {len(s['action_language'])} {ok(not s['action_language'])}")
    for a in s['action_language'][:3]:
        L.append(f"      ! ...{a['context'][:90]}...")
    L.append(f"  causal claims hedged     {s['causal_hedged']}")
    L.append(f"  causal claims ASSERTED   {len(s['causal_asserted'])} {ok(not s['causal_asserted'])}")
    for c in s['causal_asserted'][:3]:
        L.append(f"      ! {c['sentence'][:88]}  (news overlap {c['news_overlap']})")
    miss = [k for k, v in s['sections'].items() if not v]
    L.append(f"  required sections        {sum(s['sections'].values())}/{len(s['sections'])} "
             f"{ok(not miss)}" + (f"  missing: {miss}" if miss else ''))
    L.append(f"  length                   {s['tokens']:,} tokens")
    L.append('-' * 58)
    L.append(f"  TRUST SCORE              {s['trust_score']}%")
    L.append('=' * 58)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--context', required=True)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    s = score(open(a.report).read(), open(a.context).read())
    print(json.dumps(s, indent=2) if a.json else render(s, a.name or os.path.basename(a.report)))
    sys.exit(0 if s['trust_score'] >= 70 else 1)


if __name__ == '__main__':
    main()
