#!/usr/bin/env python3
"""
ORACLE FORECAST — daily bull/bear day predictor (NO TRADING, track-record only).
================================================================================
The honest experiment: each morning the agent ingests everything we have — price, gap,
TA (SMA 7/50/150, RSI, momentum), realized volatility, and live news sentiment — and the
LLM commits to a BULL / BEAR / NEUTRAL call for the day WITH a decision matrix and rationale.
We log it, then VERIFY against the actual QQQ open->close at EOD and score the running accuracy.

No capital is risked. The point is to find out whether the agent can beat the 54.5% base rate
of an up day BEFORE we ever let it touch money. If it can't, we never trade it.

Usage:
  python oracle_forecast.py --forecast          # morning: make + log today's call
  python oracle_forecast.py --verify            # EOD: score all open forecasts, print accuracy
  python oracle_forecast.py --forecast --provider gemini   # use cloud model instead of local qwen
  python oracle_forecast.py --report            # just print the scorecard

Definition: a day is BULL if QQQ close > open (intraday), else BEAR. NEUTRAL calls are logged
but excluded from accuracy (they count as "stood aside").
"""
import os, json, argparse, re, smtplib, time
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import alphahconfig as cfg

LOG = 'oracle_forecasts.jsonl'
RECIPIENT = getattr(cfg, 'ALERT_EMAIL', '') or getattr(cfg, 'GMAIL_USER', '')   # inbox for verdicts (override via cfg.ALERT_EMAIL)
MIN_FEEDBACK_DAYS = 40   # in-context feedback stays DORMANT below this (would fit noise) — see oracle_calibration.py
ET = ZoneInfo('America/New_York')        # market clock — robust regardless of the machine's timezone


def now_et():
    return datetime.now(ET)


REF = 'QQQ'
# index + heavyweights + bonds + geo/macro proxies (defense=LMT, oil=USO, gold=GLD = risk-on/off + war/tariff tone)
NEWS_TICKERS = ['QQQ', 'SPY', 'DIA', 'NVDA', 'MSFT', 'AAPL', 'TLT', 'LMT', 'USO', 'GLD']
DATA = 'https://data.alpaca.markets'
HDR = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY, 'accept': 'application/json'}


# ---------------- market data ----------------
def _bars(symbol, lookback=400):
    start = (datetime.utcnow() - timedelta(days=lookback)).strftime('%Y-%m-%d')
    p = {'symbols': symbol, 'timeframe': '1Day', 'start': start, 'limit': 10000, 'feed': 'sip', 'adjustment': 'all'}
    r = requests.get(f'{DATA}/v2/stocks/bars', headers=HDR, params=p, timeout=30)
    if r.status_code == 403:
        p['feed'] = 'iex'; r = requests.get(f'{DATA}/v2/stocks/bars', headers=HDR, params=p, timeout=30)
    bars = r.json().get('bars', {}).get(symbol, [])
    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['t']).dt.tz_convert('US/Eastern').dt.tz_localize(None).dt.normalize()
    return df.set_index('Date')[['o', 'h', 'l', 'c', 'v']].rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Vol'})


def _snapshot(symbol):
    for feed in ('sip', 'iex'):
        r = requests.get(f'{DATA}/v2/stocks/snapshots', headers=HDR, params={'symbols': symbol, 'feed': feed}, timeout=20)
        if r.status_code == 200:
            return r.json().get('snapshots', r.json()).get(symbol, {})
    return {}


def is_trading_day():
    """True only on a real NYSE session today (weekends + market holidays excluded)."""
    today = now_et().strftime('%Y-%m-%d')
    try:
        r = requests.get('https://paper-api.alpaca.markets/v2/calendar',
                         headers=HDR, params={'start': today, 'end': today}, timeout=15)
        days = r.json()
        return bool(days) and any(d.get('date') == today for d in days)
    except Exception as e:
        print(f"[!] calendar check failed ({e}); assuming weekday is tradeable.")
        return now_et().weekday() < 5


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_signals():
    df = _bars(REF)
    if df.empty or len(df) < 160:
        raise SystemExit("[!] Not enough QQQ history.")
    today = pd.Timestamp(now_et().date())
    hist = df[df.index < today]                      # closed bars only
    c = hist['Close']
    pc = float(c.iloc[-1])
    snap = _snapshot(REF)
    op = (snap.get('dailyBar') or {}).get('o') or (snap.get('latestTrade') or {}).get('p') or pc
    op = float(op)
    sma7, sma50, sma150 = c.rolling(7).mean().iloc[-1], c.rolling(50).mean().iloc[-1], c.rolling(150).mean().iloc[-1]
    rv = c.pct_change().tail(20).std() * np.sqrt(252) * 100
    stack = sum([pc > sma7, pc > sma50, pc > sma150])
    return {
        'qqq_prev_close': round(pc, 2), 'open': round(op, 2), 'gap_pct': round((op / pc - 1) * 100, 2),
        'sma7': round(float(sma7), 2), 'sma50': round(float(sma50), 2), 'sma150': round(float(sma150), 2),
        'above_7d': bool(pc > sma7), 'above_50d': bool(pc > sma50), 'above_150d': bool(pc > sma150),
        'dist_50d_pct': round((pc / sma50 - 1) * 100, 2), 'dist_150d_pct': round((pc / sma150 - 1) * 100, 2),
        'rsi14': round(float(rsi(c).iloc[-1]), 1),
        'mom_1d': round((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2), 'mom_5d': round((c.iloc[-1] / c.iloc[-6] - 1) * 100, 2),
        'realized_vol_20d': round(float(rv), 1),
        'trend_stack': 'bull' if stack == 3 else ('bear' if stack == 0 else 'mixed'),
    }


# ---------------- news ----------------
def get_news():
    out = []
    p = {'symbols': ','.join(NEWS_TICKERS), 'limit': 25, 'include_content': False, 'sort': 'desc'}
    try:
        r = requests.get(f'{DATA}/v1beta1/news', headers=HDR, params=p, timeout=20)
        for a in r.json().get('news', []):
            out.append({'headline': a.get('headline', ''), 'tickers': a.get('symbols', []), 'created': a.get('created_at', '')})
    except Exception as e:
        print(f"[!] Alpaca news error: {e}")
    # general/macro headlines — geopolitics (war, tariffs, Fed, elections) often isn't tagged to one ticker
    try:
        r = requests.get(f'{DATA}/v1beta1/news', headers=HDR,
                         params={'limit': 15, 'include_content': False, 'sort': 'desc'}, timeout=20)
        for a in r.json().get('news', []):
            out.append({'headline': a.get('headline', ''), 'tickers': a.get('symbols', []),
                        'created': a.get('created_at', ''), 'src': 'macro'})
    except Exception as e:
        print(f"[!] general news error: {e}")
    # optional: tradingnews.press (real-time API) if a key is configured
    tn_key = getattr(cfg, 'TRADINGNEWS_API_KEY', '')
    if tn_key:
        try:
            r = requests.get('https://api.tradingnews.press/v1/news',
                             headers={'Authorization': f'Bearer {tn_key}'},
                             params={'limit': 25, 'asset_class': 'equity'}, timeout=15)
            for a in r.json().get('data', r.json().get('news', [])):
                out.append({'headline': a.get('headline', a.get('title', '')),
                            'tickers': a.get('tickers', a.get('symbols', [])), 'created': a.get('timestamp', ''), 'src': 'tradingnews'})
        except Exception as e:
            print(f"[!] tradingnews.press error (skipping): {e}")
    # de-dup by headline (the ticker + macro pulls overlap)
    seen, uniq = set(), []
    for n in out:
        h = n.get('headline', '')
        if h and h not in seen:
            seen.add(h); uniq.append(n)
    return uniq[:30]


# ---------------- LLM ----------------
def _ollama_ready(base, tries=6, wait=10):
    """Wait for local Ollama to be up (it can be slow to start after a reboot)."""
    for _ in range(tries):
        try:
            if requests.get(f'{base}/api/tags', timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(wait)
    return False


def call_llm(prompt, provider, model):
    """Retry wrapper — a transient blip (Ollama not up post-reboot, network hiccup) shouldn't cost a run."""
    if provider == 'ollama':
        _ollama_ready(getattr(cfg, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
    last = None
    for attempt in range(3):
        try:
            return _call_llm_once(prompt, provider, model)
        except Exception as e:
            last = e
            print(f"[!] LLM attempt {attempt+1}/3 failed: {e}")
            time.sleep(15)
    raise last


def _call_llm_once(prompt, provider, model):
    if provider == 'claude':                          # frontier reasoning via Anthropic API
        key = getattr(cfg, 'ANTHROPIC_API_KEY', '')
        if not key:
            raise SystemExit("[!] provider=claude needs cfg.ANTHROPIC_API_KEY (paid Anthropic API key).")
        r = requests.post('https://api.anthropic.com/v1/messages',
                          headers={'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
                          json={'model': model, 'max_tokens': 1024, 'messages': [{'role': 'user', 'content': prompt}]}, timeout=90)
        j = r.json()
        if 'content' not in j:
            raise SystemExit(f"[!] Anthropic API error: {j}")
        return j['content'][0]['text']
    if provider == 'gemini':
        key = getattr(cfg, 'GEMINI_API_KEY', '')
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
        r = requests.post(url, json={'contents': [{'parts': [{'text': prompt}]}]}, timeout=60)
        return r.json()['candidates'][0]['content']['parts'][0]['text']
    # default: ollama (local)
    url = f"{getattr(cfg, 'OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
    r = requests.post(url, json={'model': model, 'prompt': prompt, 'stream': False, 'format': 'json'}, timeout=300)
    return r.json().get('response', '')


PROMPT = """You are a disciplined macro day-trader forecasting the NASDAQ-100 (QQQ) for TODAY's regular session.
COMMIT to a call: will today be a BULL day (QQQ closes ABOVE its open) or a BEAR day (closes BELOW its open)?
Default to BULL or BEAR — pick whichever the evidence favors, even if only slightly. NEUTRAL is a LAST RESORT
for a genuine 50/50 and should be RARE; do not hide in it.

Two hard requirements, because a forecaster that says the same thing every day is worthless:
1. CONVICTION MUST VARY with the actual strength of the evidence — a strong day might be 75, a marginal day 55.
   NEVER output a default/middle number. If you find yourself writing ~50 every day, you are not doing the job.
2. Be genuinely willing to call BEAR. "Always BULL in a bull market" scores well by luck and proves nothing —
   only a call that sometimes goes against the drift has any information in it.

QUANTITATIVE SIGNALS (as of today's open):
{signals}

RECENT NEWS HEADLINES (most recent first):
{news}

Weigh trend, momentum, the opening gap, RSI, volatility, news tone, AND macro/geopolitical risk
(war/escalation, tariffs & trade, Fed/rates, elections) — these set the risk-on vs risk-off tone.

RULES FOR A USEFUL CALL (do not be vague):
- READ THE HEADLINES. If any headline is materially bullish or bearish for equities — e.g. geopolitical
  DE-escalation, a Fed surprise, futures jumping/falling, a major earnings shock — you MUST reflect it in
  the news/geopolitics matrix and name it. Do NOT default catalysts to "0" when a clear one is present.
- Your rationale must CITE the specific headline(s) or signal driving your view. "Mixed signals, hard to
  call" is a banned non-answer — say WHICH signals conflict and which one you weight more.
- Always give a directional `lean` (BULL or BEAR — your best tilt even at low conviction). `call` may be
  NEUTRAL (you'd stand aside), but `lean` must commit.

For "matrix", rate EACH factor with exactly one symbol — "+" (bullish), "-" (bearish), or "0" (neutral) —
except "volatility" which is one of "calm" | "elevated" | "high". For "key_driver", name the SPECIFIC
catalyst (quote the headline or signal), never a generic phrase like "mixed signals".

Respond with STRICT JSON only, matching this exact shape. The values below are a FORMAT EXAMPLE ONLY
(an unrelated hypothetical) — do not copy its direction; derive your own from today's data:
{{"call":"BEAR","lean":"BEAR","conviction":62,
  "key_driver":"Hotter-than-expected CPI print pushing rate-cut odds lower",
  "matrix":{{"trend":"-","momentum":"-","gap":"-","rsi":"-","news":"-","geopolitics":"0","volatility":"high"}},
  "rationale":"An example only: hot inflation + price below the 50-day with falling momentum point risk-off into the session.",
  "invalidation":"A reclaim of the 50-day average on strong breadth would void this bearish read."}}"""


def build_feedback():
    """DORMANT until MIN_FEEDBACK_DAYS verified calls exist. Returns an in-context track-record
    block so the agent can self-correct its calibration — or '' if not enough data (so it can't fit noise)."""
    scored = [r for r in load() if r.get('hit') is not None]
    if len(scored) < MIN_FEEDBACK_DAYS:
        return ''
    recent = scored[-30:]
    acc = sum(r['hit'] for r in recent) / len(recent)
    hi = [r for r in recent if r['decision'].get('conviction', 0) >= 70]
    hi_acc = (sum(r['hit'] for r in hi) / len(hi) * 100) if hi else None
    lines = [f"YOUR RECENT TRACK RECORD (last {len(recent)} verified calls): {acc*100:.0f}% directional accuracy."]
    if hi_acc is not None:
        lines.append(f"Your high-conviction (>=70) calls hit {hi_acc:.0f}% ({len(hi)} of them).")
    lines.append("Use this to stay honest about your conviction — do not repeat systematic errors.")
    return "CALIBRATION FEEDBACK:\n" + '\n'.join(lines) + "\n\n"


def make_forecast(provider, model, feedback=False):
    sig = build_signals()
    news = get_news()
    news_txt = '\n'.join(f"- {n['headline']} ({','.join(n['tickers'][:3])})" for n in news[:25]) or "(no headlines)"
    fb = build_feedback() if feedback else ''
    prompt = fb + PROMPT.format(signals=json.dumps(sig, indent=2), news=news_txt)
    raw = call_llm(prompt, provider, model)
    parse_ok = True
    try:
        dec = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                dec = json.loads(m.group(0))
            except Exception:
                parse_ok = False; dec = {'call': 'NEUTRAL', 'conviction': 0, 'rationale': 'parse-fail'}
        else:
            parse_ok = False; dec = {'call': 'NEUTRAL', 'conviction': 0, 'rationale': 'parse-fail'}
    rec = {'date': now_et().strftime('%Y-%m-%d'), 'ts_forecast': now_et().isoformat(timespec='seconds'),
           'provider': provider, 'model': model, 'signals': sig,
           'news': [n['headline'] for n in news[:25]],          # full headline set the model saw
           'feedback_used': bool(fb),
           'decision': dec, 'raw_response': raw, 'parse_ok': parse_ok,   # raw LLM text for reasoning audits
           'prompt': prompt,                                            # exact prompt sent (full audit trail)
           'actual': None, 'hit': None}
    return rec


# ---------------- log + verify ----------------
def append(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def load():
    if not os.path.exists(LOG):
        return []
    return [json.loads(l) for l in open(LOG) if l.strip()]


def verify(email=True):
    recs = load()
    if not recs:
        print("[*] No forecasts logged yet."); return
    df = _bars(REF, lookback=60)
    today = pd.Timestamp(now_et().date())
    changed = 0
    for rec in recs:
        # 1) fill actual outcome if the session has closed and it's missing
        if rec.get('actual') is None:
            d = pd.Timestamp(rec['date'])
            if d not in df.index:
                continue
            if d == today and now_et().hour < 16:
                continue                               # session not closed yet
            bar = df.loc[d]
            intraday = bar['Close'] / bar['Open'] - 1
            rec['actual'] = {'open': round(float(bar['Open']), 2), 'close': round(float(bar['Close']), 2),
                             'intraday_pct': round(float(intraday) * 100, 2),
                             'day': 'BULL' if intraday > 0 else 'BEAR'}
            changed += 1
        # 2) (re)derive scores from actual — backfills lean_hit onto older records too
        if rec.get('actual'):
            act = rec['actual']['day']
            call = rec['decision'].get('call', 'NEUTRAL')
            lean = rec['decision'].get('lean')
            new_hit = None if call == 'NEUTRAL' else bool(call == act)           # only when it formally commits
            new_lean = bool(lean == act) if lean in ('BULL', 'BEAR') else None   # lean always commits -> always scorable
            if rec.get('hit') != new_hit or rec.get('lean_hit') != new_lean:
                rec['hit'] = new_hit; rec['lean_hit'] = new_lean; changed += 1
    if changed:
        with open(LOG, 'w') as f:
            for rec in recs:
                f.write(json.dumps(rec) + '\n')
    txt = report(recs)
    if email:
        today = now_et().strftime('%Y-%m-%d')
        verified_today = [r for r in recs if r.get('date') == today and r.get('actual')]
        head = ''
        if verified_today:
            r = verified_today[-1]
            lh = 'LEAN HIT' if r.get('lean_hit') else 'LEAN MISS' if r.get('lean_hit') is not None else 'n/a'
            head = (f"TODAY {today}: call {r['decision'].get('call')}/lean {r['decision'].get('lean','?')} "
                    f"(conv {r['decision'].get('conviction')}) -> actual {r['actual']['day']} "
                    f"({r['actual']['intraday_pct']:+.2f}%) -> {lh}\n\n")
        send_email(f"Oracle verdict {today}", head + txt)


def send_email(subject, body):
    u = getattr(cfg, 'GMAIL_USER', None); p = getattr(cfg, 'GMAIL_APP_PASSWORD', None)
    if not (u and p):
        print("[!] email skipped — GMAIL_USER/GMAIL_APP_PASSWORD not set."); return False
    try:
        m = MIMEText(body); m['Subject'] = subject; m['From'] = u; m['To'] = RECIPIENT
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(u, p); s.send_message(m)
        print(f"[*] emailed -> {RECIPIENT}: {subject}")
        return True
    except Exception as e:
        print(f"[!] email failed: {e}"); return False


def report(recs=None):
    """Build the scorecard text, print it, and return it (so it can be emailed).
    PRIMARY metric = LEAN accuracy (the lean always commits BULL/BEAR, so every verified day counts).
    Formal-call accuracy is shown as a secondary 'when it chose to bet' stat."""
    recs = recs if recs is not None else load()
    leans = [r for r in recs if r.get('lean_hit') is not None]     # every verified day (lean always commits)
    out = [f"=== ORACLE FORECAST SCORECARD ===  ({len(recs)} logged, {len(leans)} verified)"]
    if not leans:
        out.append("No verified days yet. base rate to beat: 54.5% BULL days.")
        txt = '\n'.join(out); print('\n' + txt); return txt
    lh = sum(r['lean_hit'] for r in leans)
    out.append(f"Directional LEAN accuracy: {lh}/{len(leans)} = {lh/len(leans)*100:.0f}%   (base rate 54.5%)")
    # --- degeneracy check: is the agent actually predicting, or just parroting one answer? ---
    n_bull = sum(1 for r in leans if r['decision'].get('lean') == 'BULL')
    n_bear = len(leans) - n_bull
    always_bull = sum(1 for r in leans if r['actual']['day'] == 'BULL') / len(leans) * 100   # score of a dumb always-BULL bot
    out.append(f"  lean mix: {n_bull} BULL / {n_bear} BEAR   |   an ALWAYS-BULL bot would score {always_bull:.0f}% here")
    if n_bear == 0:
        out.append("  ⚠️  agent has NEVER called BEAR — it carries no information beyond the base rate (constant output).")
    convs = [r['decision'].get('conviction', 0) for r in leans]
    if len(set(convs)) == 1:
        out.append(f"  ⚠️  conviction is STUCK at {convs[0]} every day — the model is flatlining, not reasoning.")
    calls = [r for r in recs if r.get('hit') is not None]
    if calls:
        ch = sum(r['hit'] for r in calls)
        out.append(f"When it COMMITTED (non-NEUTRAL): {ch}/{len(calls)} = {ch/len(calls)*100:.0f}%   | stood aside {len(leans)-len(calls)}")
    else:
        out.append(f"Formal calls: 0 committed (all NEUTRAL so far) — lean is the read to watch.")
    out.append(f"{'conviction':<14}{'n':>4}{'lean acc':>9}")
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 101)]:
        b = [r for r in leans if lo <= r['decision'].get('conviction', 0) < hi]
        if b:
            out.append(f"  {str(lo)+'-'+str(hi-1):<12}{len(b):>4}{sum(x['lean_hit'] for x in b)/len(b)*100:>8.0f}%")
    out.append(f"{'provider/model':<28}{'n':>4}{'lean acc':>9}")
    bykey = {}
    for r in leans:
        bykey.setdefault(f"{r.get('provider','?')}/{r.get('model','?')}", []).append(r)
    for k, b in sorted(bykey.items()):
        out.append(f"  {k:<26}{len(b):>4}{sum(x['lean_hit'] for x in b)/len(b)*100:>8.0f}%")
    out.append(f"NEUTRAL (formally stood aside): {sum(1 for r in recs if r['decision'].get('call') == 'NEUTRAL')}")
    last = recs[-1]
    a = last.get('actual')
    if a:
        lh_str = 'LEAN HIT' if last.get('lean_hit') else 'LEAN MISS' if last.get('lean_hit') is not None else 'n/a'
        outcome = f"actual {a['day']} ({a['intraday_pct']:+.2f}%)  ->  {lh_str}"
    else:
        outcome = 'pending verification'
    out.append(f"\nlast: {last['date']} call {last['decision'].get('call')}/lean {last['decision'].get('lean','?')} "
               f"(conv {last['decision'].get('conviction')})  ->  {outcome}")
    txt = '\n'.join(out); print('\n' + txt); return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forecast', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--provider', default='ollama', choices=['ollama', 'gemini', 'claude'])
    ap.add_argument('--model', default=None)
    ap.add_argument('--force', action='store_true', help='bypass the weekend/holiday guard (testing)')
    ap.add_argument('--dry_run', action='store_true', help='forecast but print only, do not append to log')
    ap.add_argument('--no_email', action='store_true', help='suppress the email send')
    ap.add_argument('--test_email', action='store_true', help='send a test email and exit')
    ap.add_argument('--feedback', action='store_true',
                    help=f'inject in-context track-record feedback (auto-dormant until {MIN_FEEDBACK_DAYS} verified days)')
    ap.add_argument('--show', action='store_true', help='pretty-print the last logged forecast (incl. raw response) and exit')
    a = ap.parse_args()
    if a.show:
        recs = load()
        if not recs:
            print("(no forecasts logged yet)"); return
        print(json.dumps(recs[-1], indent=2)); return
    if a.test_email:
        ok = send_email('Oracle test — email channel live',
                        "This is a test from oracle_forecast.py.\n\n"
                        "If you're reading this, daily forecasts (9:25am) and EOD verdicts (4:05pm) will "
                        f"arrive here at {RECIPIENT}.\n\nNo capital is at risk — this is the prediction "
                        "track-record experiment.")
        return
    if a.forecast:
        if not a.force and not is_trading_day():
            print(f"[*] {now_et():%Y-%m-%d} is not an NYSE trading day — no forecast."); return
        default_models = {'ollama': getattr(cfg, 'NEWS_AGENT_MODEL', 'qwen2.5:32b'),
                          'gemini': 'gemini-2.0-flash',
                          'claude': 'claude-opus-4-8'}   # best reasoning for the daily call; override w/ --model
        model = a.model or default_models[a.provider]
        try:
            rec = make_forecast(a.provider, model, feedback=a.feedback)
        except Exception as e:
            err = f"Oracle forecast FAILED {now_et():%Y-%m-%d %H:%M} ET ({a.provider}/{model}): {e}"
            print(f"[!] {err}")
            if not a.no_email:
                send_email(f"Oracle ERROR {now_et():%Y-%m-%d}", err + "\n\n(no call logged today — check oracle.log)")
            raise SystemExit(1)
        print(json.dumps(rec['decision'], indent=2))
        print(f"\nsignals: trend={rec['signals']['trend_stack']} gap={rec['signals']['gap_pct']}% "
              f"rsi={rec['signals']['rsi14']} 50d={'+' if rec['signals']['above_50d'] else '-'} "
              f"150d={'+' if rec['signals']['above_150d'] else '-'} vol={rec['signals']['realized_vol_20d']}")
        if not a.dry_run:
            append(rec); print(f"\n[*] Logged to {LOG}")
            if not a.no_email:
                s = rec['signals']; dec = rec['decision']
                heads = '\n'.join(f"  • {h}" for h in rec.get('news', [])[:7]) or "  (none)"
                body = (f"Oracle morning call — {rec['date']}  ({a.provider}/{model})\n\n"
                        f"CALL: {dec.get('call')}   |   lean {dec.get('lean','?')}   |   conviction {dec.get('conviction')}\n"
                        f"KEY DRIVER: {dec.get('key_driver','—')}\n\n"
                        f"rationale: {dec.get('rationale','')}\n"
                        f"invalidation: {dec.get('invalidation','')}\n\n"
                        f"matrix: {json.dumps(dec.get('matrix', {}))}\n"
                        f"signals: trend={s['trend_stack']} gap={s['gap_pct']}% rsi={s['rsi14']} "
                        f"mom5d={s['mom_5d']}% 50d={'+' if s['above_50d'] else '-'} "
                        f"150d={'+' if s['above_150d'] else '-'} vol={s['realized_vol_20d']}\n\n"
                        f"top headlines it weighed:\n{heads}\n\n(verdict email follows at EOD)")
                send_email(f"Oracle call {rec['date']}: {dec.get('call')}/{dec.get('lean','?')} (conv {dec.get('conviction')})", body)
        else:
            print("\n[*] DRY RUN — not logged.")
    elif a.verify:
        verify(email=not a.no_email)
    elif a.report:
        report()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
