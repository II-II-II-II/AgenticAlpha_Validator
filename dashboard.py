#!/usr/bin/env python3
"""
dashboard.py — QC4 live web dashboard (local Flask server).

Single-view QC4 dashboard: current allocation + portfolio value · QQQ / VIX / TQQQ moving averages
(live price via Yahoo incl. pre/post-market, prior close from real daily bars) · top market news ·
a live "morning read" panel · a buy/sell trade chart · and a performance chart toggling between the
live paper NAV and $10k-from-Jan backtests.

Prices: live via Yahoo (matches broker, all sessions); heavy Alpaca pulls cached 15 min. Binds 0.0.0.0
for phone access on the LAN. (Options/dip experiments removed 2026-07-21 — see research/graveyard/.)
"""
import json, time, os, sys, requests, warnings, logging
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, Response
import alphahconfig as cfg
from letf_strategy import build_weights

warnings.filterwarnings('ignore'); logging.getLogger('yfinance').setLevel(logging.CRITICAL)
H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
DATA = 'https://data.alpaca.markets'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'research'))
import macro as macromod         # objective macro backdrop (market-based + FRED hard data)
PORT = 8787
TTL = 900                        # 15-min cache on Alpaca-heavy calls
app = Flask(__name__)
_cache = {}


def cached(key, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < TTL:
        return _cache[key][1]
    val = fn(); _cache[key] = (now, val); return val


# ---------------- Alpaca helpers ----------------
def daily_bars(syms, start):
    out, tok = {}, None
    while True:
        p = {'symbols': ','.join(syms), 'timeframe': '1Day', 'start': start,
             'limit': 10000, 'feed': 'iex', 'adjustment': 'all'}
        if tok:
            p['page_token'] = tok
        j = requests.get(f"{DATA}/v2/stocks/bars", headers=H, params=p, timeout=25).json()
        for s, b in j.get('bars', {}).items():
            out.setdefault(s, []).extend(b)
        tok = j.get('next_page_token')
        if not tok:
            break
    return {s: pd.DataFrame(b) for s, b in out.items() if b}


def latest(syms):
    j = requests.get(f"{DATA}/v2/stocks/snapshots", headers=H,
                     params={'symbols': ','.join(syms), 'feed': 'iex'}, timeout=15).json()
    out = {}
    for s, v in j.items():
        lt = (v.get('latestTrade') or {}).get('p')
        pc = (v.get('prevDailyBar') or {}).get('c')
        db = v.get('dailyBar') or {}
        out[s] = dict(price=lt, prev=pc, open=db.get('o'))
    return out


def yahoo_live(sym):
    """Up-to-the-minute price incl pre/post market from Yahoo (near real-time, matches broker). Timestamped."""
    try:
        j = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}',
                         params={'includePrePost': 'true', 'interval': '1m', 'range': '1d'},
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).json()
        res = j['chart']['result'][0]; m = res['meta']
        ts = res.get('timestamp') or []
        closes = (res.get('indicators', {}).get('quote', [{}])[0].get('close')) or []
        price = when = None
        for i in range(len(ts) - 1, -1, -1):
            if i < len(closes) and closes[i] is not None:
                price = round(float(closes[i]), 2)
                when = datetime.fromtimestamp(ts[i], timezone.utc).astimezone().strftime('%H:%M:%S %Z')
                break
        if price is None and m.get('regularMarketPrice'):
            price = round(float(m['regularMarketPrice']), 2)
            when = datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S %Z')
        prev = m.get('previousClose') or m.get('chartPreviousClose')
        return dict(price=price, ts=when, state=m.get('marketState'),
                    prev=round(float(prev), 2) if prev else None)
    except Exception as e:
        return dict(price=None, ts=None, state=None, prev=None, error=str(e))


def rsi(series, n=14):
    d = series.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return float((100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1]))) if dn.iloc[-1] else 100.0


# ---------------- section builders ----------------
def sec_alloc():
    st = json.load(open(os.path.join(HERE, 'etf_paper_state.json')))
    pos = {k: v for k, v in st['positions'].items() if v}
    px = latest(list(pos) or ['TQQQ'])
    holds = []
    total = st['cash']
    for tk, sh in pos.items():
        pr = px.get(tk, {}).get('price') or 0
        val = sh * pr; total += val
        holds.append(dict(ticker=tk, shares=sh, price=round(pr, 2), value=round(val, 2)))
    for h in holds:
        h['pct'] = round(h['value'] / total * 100, 1) if total else 0
    return dict(holdings=holds, cash=round(st['cash'], 2), nav=round(total, 2),
                start_cap=st.get('start_cap', 6000), inception=st.get('inception'))


def sec_mas():
    q = daily_bars(['QQQ'], (datetime.now(timezone.utc) - timedelta(days=320)).date().isoformat())['QQQ']
    c = q['c']
    price = latest(['QQQ'])['QQQ']['price'] or float(c.iloc[-1])
    mas = []
    for n in (7, 14, 30, 90, 150):
        v = float(c.tail(n).mean())
        mas.append(dict(period=n, value=round(v, 2), above=price > v,
                        dist=round((price / v - 1) * 100, 2)))
    return dict(price=round(price, 2), mas=mas,
                regime='BULL' if price > mas[-1]['value'] else 'BEAR', rsi=round(rsi(c), 1),
                series=[round(x, 2) for x in c.tail(120).tolist()],
                sma_lines={str(n): [round(float(c.iloc[max(0, i-n+1):i+1].mean()), 2)
                           for i in range(len(c)-120, len(c))] for n in (30, 150)})


def mas_readout(sym, periods=(7, 14, 30, 90, 150), price=None):
    q = daily_bars([sym], (datetime.now(timezone.utc) - timedelta(days=320)).date().isoformat())[sym]
    c = q['c']
    if price is None:
        price = latest([sym])[sym]['price'] or float(c.iloc[-1])
    mas = []
    for n in periods:
        v = float(c.tail(n).mean())
        mas.append(dict(period=n, value=round(v, 2), above=price > v, dist=round((price / v - 1) * 100, 2)))
    return dict(price=round(price, 2), mas=mas)


def prior_close(sym):
    """Reliable prior-session close = last COMPLETED daily bar before today (Alpaca SIP).
    NEVER trust Yahoo meta previousClose/chartPreviousClose — they return None or a range-dependent stale value."""
    r = requests.get(f"{DATA}/v2/stocks/{sym}/bars", headers=H,
                     params={'timeframe': '1Day', 'start': (datetime.now(timezone.utc) - timedelta(days=12)).date().isoformat(),
                             'feed': 'sip', 'adjustment': 'all', 'limit': 12}, timeout=15).json().get('bars', [])
    today = datetime.now(timezone.utc).date().isoformat()
    prior = [b for b in r if b['t'][:10] < today]     # exclude today's (possibly-forming) bar
    return prior[-1]['c'] if prior else (r[-2]['c'] if len(r) >= 2 else None)


def tqqq_live_price():
    """fresh Yahoo live price + timestamp + pullback vs the REAL prior close (from Alpaca daily bars)."""
    yl = yahoo_live('TQQQ')
    prev = prior_close('TQQQ')
    yl['prev'] = round(prev, 2) if prev else None
    yl['chg_prev'] = round((yl['price'] / prev - 1) * 100, 2) if (yl.get('price') and prev) else None
    return yl


def sec_tqqq():
    yl = tqqq_live_price()
    r = mas_readout('TQQQ', (7, 14, 30, 90, 150), price=yl.get('price'))
    r['price_ts'] = yl.get('ts'); r['state'] = yl.get('state'); r['chg_prev'] = yl.get('chg_prev')
    op = latest(['TQQQ'])['TQQQ'].get('open')
    r['chg_open'] = round((r['price'] / op - 1) * 100, 2) if (r.get('price') and op) else None
    return r


def sec_vix():
    # VIX (^VIX) isn't on Alpaca's stock API — pull via yahooquery, compute its own 7/14/30/90d averages
    try:
        from yahooquery import Ticker
        tk = Ticker('^VIX')
        h = tk.history(period='7mo').reset_index()
        closes = [float(x) for x in h['close'].tolist() if pd.notna(x)]
        cur = closes[-1]; ts = None
        try:                                   # overlay the LIVE intraday value (pre/regular/post aware)
            pr = tk.price['^VIX']; state = (pr.get('marketState') or '').upper()
            live = pr.get('regularMarketPrice')
            if 'PRE' in state and pr.get('preMarketPrice'): live = pr['preMarketPrice']
            elif 'POST' in state and pr.get('postMarketPrice'): live = pr['postMarketPrice']
            if live: cur = round(float(live), 2); ts = datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S %Z')
        except Exception:
            pass
        mas = []
        for n in (7, 14, 30, 90):
            v = sum(closes[-n:]) / min(len(closes), n)
            mas.append(dict(period=n, value=round(v, 2), above=cur > v, dist=round((cur / v - 1) * 100, 2)))
        return dict(price=round(cur, 2), ts=ts, mas=mas, gate=cur >= 18)
    except Exception as e:
        return dict(price=None, mas=[], gate=False, error=str(e))


def sec_macro():
    """Objective macro backdrop: market-based (Yahoo, real-time) + FRED hard data + credit regime.
    Its own async endpoint so a slow/unreachable FRED never blocks the main dashboard.
    Persists to datalake/ so the Oracle agents read the SAME data the dashboard shows."""
    snap = macromod.snapshot()
    try:
        macromod.save_snapshot(snap)
    except Exception:
        pass
    return snap


def sec_news():
    def pull(params):
        try:
            return requests.get(f"{DATA}/v1beta1/news", headers=H, params=params, timeout=15).json().get('news', [])
        except Exception:
            return []
    arts = pull({'symbols': 'SPY,QQQ,DIA', 'limit': 15}) + pull({'limit': 15})
    seen, out = set(), []
    for a in sorted(arts, key=lambda x: x.get('created_at', ''), reverse=True):
        h = a.get('headline', '')
        if h and h not in seen:
            seen.add(h)
            out.append(dict(headline=h, source=a.get('source', ''),
                            when=a.get('created_at', '')[:16].replace('T', ' '), url=a.get('url', '')))
        if len(out) >= 10:
            break
    return out


def sec_morning():
    m = cached('mas', sec_mas)
    return dict(regime=m['regime'], rsi=m['rsi'], qqq=m['price'],
                sma150=m['mas'][-1]['value'], above150=m['mas'][-1]['above'])


def sec_trades():
    p = os.path.join(HERE, 'etf_paper_trades.csv')
    if not os.path.exists(p):
        return dict(price=[], dates=[], trades=[])
    tr = pd.read_csv(p)
    start = (pd.Timestamp(tr.date.min()) - pd.Timedelta(days=3)).date().isoformat()
    q = daily_bars(['TQQQ'], start).get('TQQQ', pd.DataFrame())
    dates = [t[:10] for t in q['t']] if len(q) else []
    price = [round(x, 2) for x in q['c'].tolist()] if len(q) else []
    trades = [dict(date=r.date, ticker=r.ticker, side=r.side, price=round(r.price, 2))
              for r in tr.itertuples()]
    return dict(dates=dates, price=price, trades=trades)


def sec_nav():
    p = os.path.join(HERE, 'etf_paper_nav.csv')
    if not os.path.exists(p):
        return dict(dates=[], nav=[])
    df = pd.read_csv(p, header=None)
    df[1] = pd.to_numeric(df[1], errors='coerce')
    df = df.dropna(subset=[1])                       # drop any header row
    return dict(dates=[str(x)[:10] for x in df[0].tolist()],
                nav=[round(float(x), 2) for x in df[1].tolist()])


def qc4_backtest(start_date='2026-01-02', capital=10000, warmup='2025-05-01'):
    syms = ['QQQ', 'TQQQ', 'SQQQ', 'TLT', 'SHY', 'SPY']
    b = daily_bars(syms, warmup)
    PX = pd.DataFrame({s: b[s].set_index('t')['c'] for s in syms if s in b})
    OP = pd.DataFrame({s: b[s].set_index('t')['o'] for s in syms if s in b})
    PX.index = pd.to_datetime(PX.index); OP.index = pd.to_datetime(OP.index)
    ORET = OP.shift(-1) / OP - 1.0
    W = build_weights(PX, sma_bull=150, sma_slow=152, rsi_lo=35, rsi_bear=40, s2_mode='rsi')
    turn = W.diff().abs().sum(axis=1).fillna(0); COST = 5 / 1e4
    cash_w = (1 - W[['TQQQ', 'SQQQ', 'TLT']].sum(axis=1)).clip(lower=0)
    shy = (cash_w.shift(1) * ORET['SHY']).fillna(0)
    qc4 = ((W.shift(1) * ORET).sum(axis=1) - turn.shift(1).fillna(0) * COST).fillna(0) + shy
    r = pd.DataFrame({'QC4': qc4, 'SPY': ORET['SPY']}).dropna()
    tz = getattr(r.index, 'tz', None)
    r = r[r.index >= pd.Timestamp(start_date, tz=tz)]
    eq = (1 + r).cumprod() * capital
    q = eq['QC4']
    dd = float((q / q.cummax() - 1).min() * 100)
    yrs = max(len(q) / 252, 0.1); cagr = float((q.iloc[-1] / capital) ** (1 / yrs) - 1) * 100
    return dict(dates=[d.date().isoformat() for d in eq.index],
                qc4=[round(x, 2) for x in q.tolist()],
                spy=[round(x, 2) for x in eq['SPY'].tolist()],
                final_qc4=round(q.iloc[-1], 0), final_spy=round(eq['SPY'].iloc[-1], 0),
                total=round((q.iloc[-1] / capital - 1) * 100, 1), cagr=round(cagr, 1), max_dd=round(dd, 1))


@app.route('/api/tqqqlive')
def api_tqqqlive():
    return jsonify(tqqq_live_price())        # uncached — fresh Yahoo pull for the refresh button


@app.route('/api/macro')
def api_macro():
    return jsonify(cached('macro', sec_macro))   # own endpoint; a slow FRED can't block /api/data


@app.route('/api/data')
def api():
    return jsonify(dict(
        updated=datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
        alloc=cached('alloc', sec_alloc), mas=cached('mas', sec_mas),
        news=cached('news', sec_news), morning=sec_morning(),
        vix=cached('vix', sec_vix), tqqq=cached('tqqq', sec_tqqq),
        trades=cached('trades', sec_trades), nav=sec_nav(),
        backtest=cached('bt', lambda: qc4_backtest('2026-01-02', 10000, '2025-05-01')),
        bt2024=cached('bt24', lambda: qc4_backtest('2024-01-02', 10000, '2023-06-01'))))


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>QC4 Dashboard</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{font-family:-apple-system,Arial,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:#888;font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:#181b22;border:1px solid #262b36;border-radius:10px;padding:16px}
.card h2{font-size:14px;margin:0 0 12px;color:#9fb3c8;text-transform:uppercase;letter-spacing:.5px}
.big{font-size:34px;font-weight:700}.g{color:#3ecf7a}.r{color:#e5534b}.mut{color:#888}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:5px 4px;text-align:left;border-bottom:1px solid #262b36}
.pill{padding:2px 8px;border-radius:10px;font-weight:700;font-size:12px}
.news a{color:#cfe;text-decoration:none;font-size:13px}.news li{margin-bottom:9px}
button{background:#2c6fbb;color:#fff;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font-weight:600}
button.off{background:#333}
.tabs{margin:6px 0 14px}.tabs button{margin-right:8px;background:#20242c;font-size:15px}.tabs button.act{background:#2c6fbb}
.alpha{background:#3a2a00;color:#e5a04b;padding:2px 7px;border-radius:10px;font-size:11px;margin-left:7px}
</style></head><body>
<h1>QC4 Live Dashboard</h1><div class="sub" id="upd">loading…</div>
<div id="tab-qc4">
<div class="grid">
  <div class="card"><h2>Portfolio</h2><div class="big" id="nav">—</div><div id="navsub" class="mut"></div>
    <table id="holds"></table></div>
  <div class="card"><h2>Nasdaq (QQQ) — moving averages</h2><div class="big" id="qqq">—</div>
    <div id="regime" style="margin:6px 0"></div><table id="matab"></table></div>
  <div class="card"><h2>VIX — moving averages</h2><div class="big" id="vix">—</div>
    <div id="vixgate" style="margin:6px 0"></div>
    <div id="vixts" class="mut" style="font-size:11px;margin-bottom:6px"></div>
    <table id="vixtab"></table></div>
  <div class="card"><h2>TQQQ — moving averages
      <button onclick="refreshTqqq()" style="float:right;padding:3px 10px;font-size:12px">↻ Yahoo</button></h2>
    <div class="big" id="tqqq">—</div>
    <div id="tqsub" class="mut" style="margin:6px 0"></div>
    <div id="tqts" class="mut" style="font-size:11px;margin-bottom:6px"></div>
    <table id="tqtab"></table></div>
  <div class="card"><h2>Morning read (live)</h2><div id="morning"></div></div>
  <div class="card news"><h2>Top news</h2><ul id="news"></ul></div>
</div>
<div class="card" style="margin-top:16px">
  <h2>Macro backdrop — zooming out
    <span class="mut" style="font-size:11px;text-transform:none;letter-spacing:0;font-weight:400">· context, not a timing signal (macro is already priced in)</span></h2>
  <div id="macro" class="mut">loading…</div></div>
<div class="grid" style="margin-top:16px">
  <div class="card"><h2>QQQ price vs 30 &amp; 150-day</h2><canvas id="maChart" height="150"></canvas></div>
  <div class="card"><h2>Trades — buys &amp; sells</h2><canvas id="trChart" height="150"></canvas></div>
</div>
<div class="card" style="margin-top:16px">
  <h2>Performance</h2>
  <button id="bBt24" onclick="setMode('bt24')">$10k from Jan 2, 2024</button>
  <button id="bBt" class="off" onclick="setMode('bt')">$10k from Jan 2, 2026</button>
  <button id="bLive" class="off" onclick="setMode('live')">Live paper (since June)</button>
  <div id="perfsub" class="mut" style="margin:8px 0"></div>
  <canvas id="perfChart" height="90"></canvas></div>
</div>
<script>
let D=null, mode='bt24', charts={};
function fmt(n){return '$'+Number(n).toLocaleString(undefined,{maximumFractionDigits:0})}
async function refreshTqqq(){
  const el=document.getElementById('tqts'); el.textContent='refreshing from Yahoo…';
  try{
    const t=await (await fetch('/api/tqqqlive')).json();
    if(t.price!=null){
      document.getElementById('tqqq').textContent='$'+t.price;
      let sub=[]; if(t.chg_prev!=null) sub.push(`<span class="${t.chg_prev<0?'r':'g'}">${t.chg_prev>0?'+':''}${t.chg_prev}% vs prev close</span>`);
      document.getElementById('tqsub').innerHTML=sub.join(' · ');
      el.textContent='Yahoo '+(t.state||'')+' · pulled '+t.ts;
    } else el.textContent='Yahoo pull failed';
  }catch(e){el.textContent='refresh error';}
}
async function load(){
  D=await (await fetch('/api/data')).json();
  document.getElementById('upd').textContent='updated '+D.updated+' · auto-refresh 15 min';
  // portfolio
  const a=D.alloc; document.getElementById('nav').textContent=fmt(a.nav);
  document.getElementById('navsub').textContent='cash '+fmt(a.cash)+' · start '+fmt(a.start_cap)+' ('+a.inception+')';
  document.getElementById('holds').innerHTML='<tr><th>Holding</th><th>Shares</th><th>Price</th><th>Value</th><th>%</th></tr>'+
    a.holdings.map(h=>`<tr><td>${h.ticker}</td><td>${h.shares}</td><td>$${h.price}</td><td>${fmt(h.value)}</td><td>${h.pct}%</td></tr>`).join('');
  // MAs
  const m=D.mas; document.getElementById('qqq').textContent='$'+m.price;
  document.getElementById('regime').innerHTML=`<span class="pill ${m.regime=='BULL'?'g':'r'}" style="background:#20242c">${m.regime}</span> <span class="mut">RSI ${m.rsi}</span>`;
  document.getElementById('matab').innerHTML='<tr><th>MA</th><th>Value</th><th>Price is</th><th>Dist</th></tr>'+
    m.mas.map(x=>`<tr><td>${x.period}d</td><td>$${x.value}</td><td class="${x.above?'g':'r'}">${x.above?'ABOVE':'below'}</td><td class="${x.above?'g':'r'}">${x.dist>0?'+':''}${x.dist}%</td></tr>`).join('');
  // VIX moving averages
  const v=D.vix; if(v&&v.price!=null){
    document.getElementById('vix').textContent=v.price;
    document.getElementById('vixgate').innerHTML=v.gate
      ?'<span class="pill" style="background:#3a2a00;color:#e5a04b">VIX ≥18 — dip-buy regime ON</span>'
      :'<span class="pill mut" style="background:#20242c">VIX &lt;18 — calm</span>';
    document.getElementById('vixts').textContent=v.ts?('Yahoo live · '+v.ts):'daily close (market closed)';
    document.getElementById('vixtab').innerHTML='<tr><th>MA</th><th>Value</th><th>VIX is</th><th>Dist</th></tr>'+
      v.mas.map(x=>`<tr><td>${x.period}d</td><td>${x.value}</td><td class="${x.above?'r':'g'}">${x.above?'ABOVE':'below'}</td><td class="${x.above?'r':'g'}">${x.dist>0?'+':''}${x.dist}%</td></tr>`).join('');
  }
  // TQQQ moving averages + today's pullback (the buy trigger)
  const tq=D.tqqq; if(tq&&tq.price!=null){
    document.getElementById('tqqq').textContent='$'+tq.price;
    let sub=[];
    if(tq.chg_prev!=null) sub.push(`<span class="${tq.chg_prev<0?'r':'g'}">${tq.chg_prev>0?'+':''}${tq.chg_prev}% vs prev close</span>`);
    if(tq.chg_open!=null) sub.push(`<span class="${tq.chg_open<0?'r':'g'}">${tq.chg_open>0?'+':''}${tq.chg_open}% vs open</span>`);
    document.getElementById('tqsub').innerHTML=sub.join(' · ');
    document.getElementById('tqts').textContent=tq.price_ts?('Yahoo '+(tq.state||'')+' · price @ '+tq.price_ts):'';
    document.getElementById('tqtab').innerHTML='<tr><th>MA</th><th>Value</th><th>Price is</th><th>Dist</th></tr>'+
      tq.mas.map(x=>`<tr><td>${x.period}d</td><td>$${x.value}</td><td class="${x.above?'g':'r'}">${x.above?'ABOVE':'below'}</td><td class="${x.above?'g':'r'}">${x.dist>0?'+':''}${x.dist}%</td></tr>`).join('');
  }
  // morning
  const mo=D.morning; document.getElementById('morning').innerHTML=
    `<div class="big ${mo.regime=='BULL'?'g':'r'}">${mo.regime}</div>
     <table><tr><td>QQQ</td><td>$${mo.qqq}</td></tr><tr><td>150-day</td><td>$${mo.sma150}</td></tr>
     <tr><td>vs 150d</td><td class="${mo.above150?'g':'r'}">${mo.above150?'above — dip-buys valid':'below — defensive'}</td></tr>
     <tr><td>RSI(14)</td><td class="${mo.rsi<35?'g':''}"><b>${mo.rsi}</b>${mo.rsi<35?' · OVERSOLD — dip-buy ON':' · '+(mo.rsi-35).toFixed(0)+' pts from dip-buy (35)'}</td></tr></table>`;
  // news
  document.getElementById('news').innerHTML=D.news.map(n=>`<li><a href="${n.url}" target="_blank">${n.headline}</a><br><span class="mut">${n.source} · ${n.when}</span></li>`).join('');
  drawMA(); drawTrades(); setMode(mode);
}
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id),cfg);}
function drawMA(){const m=D.mas,L=m.series.map((_,i)=>i);
  mk('maChart',{type:'line',data:{labels:L,datasets:[
    {label:'QQQ',data:m.series,borderColor:'#4da3ff',pointRadius:0,borderWidth:2},
    {label:'30d',data:m.sma_lines['30'],borderColor:'#9fb3c8',pointRadius:0,borderWidth:1},
    {label:'150d',data:m.sma_lines['150'],borderColor:'#e5a04b',pointRadius:0,borderWidth:1}]},
    options:{plugins:{legend:{labels:{color:'#aaa'}}},scales:{x:{display:false},y:{ticks:{color:'#888'}}}}});}
function drawTrades(){const t=D.trades;const pts=t.trades.map(tr=>{const i=t.dates.indexOf(tr.date);return {x:i<0?null:i,y:tr.price,side:tr.side,tk:tr.ticker};}).filter(p=>p.x!==null);
  mk('trChart',{type:'line',data:{labels:t.dates.map((_,i)=>i),datasets:[
    {label:'TQQQ',data:t.price,borderColor:'#4da3ff',pointRadius:0,borderWidth:2},
    {label:'BUY',data:pts.filter(p=>p.side=='BUY').map(p=>({x:p.x,y:p.y})),showLine:false,pointBackgroundColor:'#3ecf7a',pointRadius:6},
    {label:'SELL',data:pts.filter(p=>p.side=='SELL').map(p=>({x:p.x,y:p.y})),showLine:false,pointBackgroundColor:'#e5534b',pointRadius:6}]},
    options:{plugins:{legend:{labels:{color:'#aaa'}}},scales:{x:{display:false},y:{ticks:{color:'#888'}}}}});}
function setMode(mm){mode=mm;
  document.getElementById('bLive').className=mm=='live'?'':'off';
  document.getElementById('bBt').className=mm=='bt'?'':'off';
  document.getElementById('bBt24').className=mm=='bt24'?'':'off';
  let labels,ds,sub;
  if(mm=='live'){const n=D.nav;labels=n.dates;ds=[{label:'NAV',data:n.nav,borderColor:'#3ecf7a',pointRadius:0,borderWidth:2}];
    sub='Live paper account since inception';}
  else{const b=(mm=='bt24')?D.bt2024:D.backtest;labels=b.dates;
    ds=[{label:'QC4',data:b.qc4,borderColor:'#3ecf7a',pointRadius:0,borderWidth:2},
        {label:'S&P 500',data:b.spy,borderColor:'#888',pointRadius:0,borderWidth:2}];
    const start=(mm=='bt24')?'Jan 2, 2024':'Jan 2, 2026';
    sub=`$10k in QC4 on ${start} → ${fmt(b.final_qc4)}  ·  +${b.total}% total · +${b.cagr}%/yr · worst DD ${b.max_dd}%   (S&P ${fmt(b.final_spy)})`;}
  document.getElementById('perfsub').textContent=sub;
  mk('perfChart',{type:'line',data:{labels,datasets:ds},options:{plugins:{legend:{labels:{color:'#aaa'}}},scales:{x:{display:false},y:{ticks:{color:'#888'}}}}});}
async function loadMacro(){
  try{
    const d=await (await fetch('/api/macro')).json();
    const dot=t=>t=='good'?'🟢':t=='bad'?'🔴':'⚪';
    const row=r=>`<tr><td>${dot(r.tone)} ${r.label}</td><td style="text-align:right"><b>${r.value}</b></td><td class="mut" style="font-size:12px">${r.trend}${r.asof?' · '+r.asof:''}</td></tr>`;
    const cr=d.credit_regime||{};
    const tc=cr.tone=='good'?'g':(cr.tone=='bad'?'r':'mut');
    let h='<div style="margin-bottom:14px">'
      +`<span class="pill ${tc}" style="background:#20242c;font-size:13px">CREDIT REGIME · ${cr.label||'—'}</span>`
      +(cr.detail?`<div class="mut" style="font-size:12px;margin-top:5px">${cr.detail}`
        +(cr.hy_percentile!=null?`  <span style="opacity:.7">(HY spread ${cr.hy_percentile}th pct · SPY 1mo ${cr.spx_1mo}%)</span>`:'')+`</div>`:'')
      +'</div>';
    h+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px">';
    h+='<div><div class="mut" style="margin-bottom:6px;font-size:11px">MARKET-BASED · real-time, forward-looking</div><table>'+d.market.map(row).join('')+'</table></div>';
    h+='<div><div class="mut" style="margin-bottom:6px;font-size:11px">HARD DATA · FRED (lagging — defines the regime)</div>';
    h+=d.fred_ok?('<table>'+d.hard.map(row).join('')+'</table>'):'<div class="mut" style="padding:8px 0">FRED not reachable from here. Populates when the dashboard runs on your Mac.</div>';
    h+='</div></div>';
    document.getElementById('macro').innerHTML=h;
  }catch(e){document.getElementById('macro').textContent='macro load error';}
}
load(); loadMacro(); setInterval(load, 900000); setInterval(loadMacro, 900000);
</script></body></html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, threaded=True)
