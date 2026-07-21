#!/usr/bin/env python3
"""
dip_report.py — 9:25 ET PRE-MARKET email for the dip-reversal satellite.

Runs the 60-day reversal screener (now trend-filtered: downtrends/knives score low), and per candidate:
a 60-day CHART, overnight/pre-market move, Alpaca news, quick financials, and a local qwen2.5:32b
NEWS SCORE (0-100, forced number — no "neutral"). Two numbers per name:
  SETUP score  = deterministic TA pattern-fit (near a dip + mean-reverting + recent bounce, downtrend-penalized)
  NEWS score   = qwen read of whether recent news supports a bounce (100) or is a knife (0)
Emails a formatted report (no SMS). Regime-gated. Idea/paper stage, separate from QC4.
"""
import io, requests, smtplib, warnings, logging
from datetime import datetime
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import alphahconfig as cfg
import dip_scanner as ds

warnings.filterwarnings('ignore'); logging.getLogger('yfinance').setLevel(logging.CRITICAL)
ET = ZoneInfo('America/New_York')
H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
LOOKBACK = 30
TO = [getattr(cfg, 'ALERT_EMAIL', '')]
MAX_CARDS = 8
WATCH = []                                        # merit-based: no pinned names; setups surface on their own
QWEN = 'qwen2.5:32b'


def snapshots(syms):
    try:
        j = requests.get("https://data.alpaca.markets/v2/stocks/snapshots", headers=H,
                         params={'symbols': ','.join(syms), 'feed': 'iex'}, timeout=15).json()
        return j.get('snapshots', j)
    except Exception:
        return {}


def news(sym, limit=5):
    try:
        j = requests.get("https://data.alpaca.markets/v1beta1/news", headers=H,
                         params={'symbols': sym, 'limit': limit}, timeout=15).json()
        return [(a.get('headline', ''), a.get('created_at', '')[:10]) for a in j.get('news', [])]
    except Exception:
        return []


def financials(sym):
    out = {}
    try:
        fi = yf.Ticker(sym).fast_info
        out = dict(mcap=fi.get('marketCap'), hi=fi.get('yearHigh'), lo=fi.get('yearLow'))
    except Exception:
        pass
    return out


def qwen(prompt):
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": QWEN, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.3, "num_predict": 160}}, timeout=180)
    return r.json().get('response', '').strip()


def news_score(name, ta, overnight, heads, fin):
    prompt = f"""You are a decisive trading analyst. A screener flagged {name} as a possible DIP-REVERSAL.
Score, from 0 to 100, whether RECENT NEWS supports a near-term BOUNCE. 100 = clear positive catalyst /
supportive news. 0 = real bad news, a falling knife, avoid. You MUST give a number — never say neutral.

Overnight/pre-market: {overnight:+.1f}% vs prior close.
Recent dips historically bounced {ta['b_avg']:+.1f}% over 3 days.
HEADLINES:
{chr(10).join('- ' + h[0] for h in heads) if heads else '(no recent headlines)'}

Reply EXACTLY:
NEWS_SCORE: <0-100 integer>
WHY: <one sentence citing a specific headline or the lack of one>"""
    try:
        txt = qwen(prompt)
    except Exception as e:
        return 50, f"(qwen unavailable: {e})"
    sc, why = 50, txt.strip()
    for line in txt.splitlines():
        u = line.upper()
        if 'NEWS_SCORE' in u:
            digs = ''.join(ch for ch in line.split(':', 1)[-1] if ch.isdigit())
            if digs:
                sc = max(0, min(100, int(digs)))
        elif u.startswith('WHY'):
            why = line.split(':', 1)[1].strip()
    return sc, why


def chart_png(name, df):
    c = df['c'].values[-LOOKBACK:]
    fig, ax = plt.subplots(figsize=(5.2, 1.7))
    ax.plot(range(len(c)), c, color='#1f77b4', lw=1.6)
    ax.axhline(c.max(), color='#c0c0c0', ls=':', lw=0.8)
    ax.axhline(c.min(), color='#c0c0c0', ls=':', lw=0.8)
    ax.scatter([len(c) - 1], [c[-1]], color='#d62728', s=28, zorder=5)
    ax.margins(x=0.01); ax.axis('off')
    ax.set_title(f"{name}  ·  {LOOKBACK}d  (${c[-1]:.2f})", fontsize=9, loc='left')
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', dpi=110); plt.close(fig)
    return buf.getvalue()


def color(v):
    return '#2e7d32' if v >= 65 else '#f9a825' if v >= 50 else '#c62828'


def money(v):
    if not v:
        return '—'
    return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"


def main():
    reg, df, data = ds.scan(ds.UNIV, LOOKBACK)
    d = datetime.now(ET).strftime('%A %b %d, %Y')
    if df.empty:
        cards = df
    else:
        setups = df[df.setup]
        watch = df[df.name.isin(WATCH) & ~df.setup]
        cards = pd.concat([setups, watch]).drop_duplicates('name')
        if len(cards) < 4:                               # thin day -> top-fit names fill the rest
            extra = df[~df.name.isin(cards.name)].head(4 - len(cards))
            cards = pd.concat([cards, extra])
        cards = cards.head(MAX_CARDS)

    snaps = snapshots(list(cards.name) + ['QQQ'])

    def overnight(sym, prev):
        lt = (snaps.get(sym, {}).get('latestTrade') or {}).get('p')
        return (lt / prev - 1) * 100 if (lt and prev) else 0.0

    qov = overnight('QQQ', reg['qqq'])
    regcolor = '#2e7d32' if reg['regime'] == 'BULL' else '#c62828'
    reg_line = (f"REGIME {reg['regime']} · QQQ {reg['qqq']:.0f} vs 150d {reg['sma']:.0f} · overnight {qov:+.1f}% — "
                + ("chop-in-uptrend, dip-buys valid" if reg['regime'] == 'BULL'
                   else "TREND BROKEN, dip-buys = knife-catch, STAND DOWN"))

    charts = {}
    parts = [f"""<html><body style="font-family:Arial,sans-serif;font-size:14px;color:#222;max-width:640px">
<h2 style="margin-bottom:2px">Dip-Reversal Pre-Market Report</h2>
<div style="color:#666">{d} · 9:25 AM ET · 60-day window</div>
<div style="background:{regcolor};color:#fff;padding:8px 12px;border-radius:6px;margin:10px 0">{reg_line}</div>
<div style="color:#666;font-size:12px">SETUP = TA pattern-fit (0-100, downtrends penalized). NEWS = qwen32b read of catalysts (0-100). Not a forecast — use stops.</div>"""]

    if cards.empty:
        parts.append("<p><b>No candidates today.</b></p>")
    for _, r in cards.iterrows():
        n = r['name']
        ov = overnight(n, r.price)
        heads = news(n)
        fin = financials(n)
        nsc, why = news_score(n, r, ov, heads, fin)
        if n in data:
            charts[n] = chart_png(n, data[n])
        fit = int(r.fit)
        ovc = '#2e7d32' if ov >= 0 else '#c62828'
        tag = '★ SETUP' if r.setup else 'watch'
        hd = "".join(f"<li>{h[0]} <span style='color:#999'>({h[1]})</span></li>" for h in heads[:3]) or "<li style='color:#999'>no recent headlines</li>"
        parts.append(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:18px;font-weight:bold">{n} <span style="font-size:12px;color:#888">{tag}</span></span>
    <span>
      <span style="background:{color(fit)};color:#fff;padding:3px 9px;border-radius:12px;font-weight:bold">SETUP {fit}</span>
      <span style="background:{color(nsc)};color:#fff;padding:3px 9px;border-radius:12px;font-weight:bold">NEWS {nsc}</span>
    </span>
  </div>
  <img src="cid:chart_{n}" width="480" style="display:block;margin:8px 0"/>
  <div style="color:#333">
    ${r.price:.2f} · overnight <b style="color:{ovc}">{ov:+.1f}%</b> · range pos <b>{r.rngpos:.0%}</b> (0=low) ·
    10d trend <b>{r.trend10*100:+.0f}%</b> · dips bounced <b>{r.b_avg:+.1f}%/3d</b> ({r.b_win}/{r.b_n}) · to top <b>{r.up_top:+.0f}%</b>
  </div>
  <div style="color:#555;font-size:13px">mkt cap {money(fin.get('mcap'))} · 52wk ${fin.get('lo') or '—'}–${fin.get('hi') or '—'}</div>
  <div style="margin-top:6px"><b>qwen:</b> {why}</div>
  <ul style="margin:6px 0;padding-left:18px;font-size:12px;color:#444">{hd}</ul>
</div>""")

    parts.append('<p style="color:#999;font-size:11px">Automated · dip-reversal satellite (idea/paper) · separate from QC4.</p></body></html>')

    root = MIMEMultipart('related')
    root['Subject'] = f"Dip-Reversal Pre-Market — {datetime.now(ET).strftime('%b %d')}"
    root['From'] = cfg.GMAIL_USER; root['To'] = ', '.join(TO)
    root.attach(MIMEText("".join(parts), 'html'))
    for n, png in charts.items():
        img = MIMEImage(png); img.add_header('Content-ID', f'<chart_{n}>')
        img.add_header('Content-Disposition', 'inline'); root.attach(img)
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465); s.login(cfg.GMAIL_USER, cfg.GMAIL_APP_PASSWORD)
    s.sendmail(cfg.GMAIL_USER, TO, root.as_string()); s.quit()
    setups = int(df.setup.sum()) if not df.empty else 0
    print(f"sent: {len(cards)} cards ({setups} setups), {len(charts)} charts, regime {reg['regime']}")


if __name__ == '__main__':
    main()
