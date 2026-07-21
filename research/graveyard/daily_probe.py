#!/usr/bin/env python3
"""
daily_probe.py — point-in-time PRICE + NEWS -> same-session direction mechanical probe.

Purpose: before spending agent compute, find out mechanically whether {technical signals + news
sentiment}, using ONLY information available at the morning decision point, predict the day's
direction (open->close) better than the base rate. Validation-first: cheap test before the debate.

STRICT POINT-IN-TIME (no look-ahead):
  - technical signals: daily bars through T-1 close, plus T's OPEN (known at the bell)
  - news: only articles published BEFORE T's open (belt-and-suspenders: re-filtered in code)
  - outcome: sign(close_T - open_T), from the historical bar (the truth)

ROBUST LOGGING is the deliverable. Every day writes ONE self-contained JSONL record to
daily_probe_log.jsonl holding: all raw signal values, EVERY headline used (ts + symbols + text),
the sentiment breakdown, several candidate rule-predictions, and the outcome. Because we log the
RAW headlines and signals, the tape can be re-scored later with any method (LLM sentiment, new
rules) WITHOUT re-pulling data. Analyze later; capture everything now.
"""
import requests, json, math, statistics as st
from datetime import datetime, timezone, timedelta
import alphahconfig as cfg

H = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
LOG = 'daily_probe_log.jsonl'
N_DAYS = 30
# index-moving basket: QQQ + its megacap constituents (news that actually moves the index)
BASKET = 'QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,TSLA,COST,NFLX'

# crude, DETERMINISTIC finance headline lexicon (step-1 baseline; raw headlines are logged so we
# can re-score with an LLM later). Reproducible and free.
POS = {'surge','surges','soar','soars','jump','jumps','rally','rallies','beat','beats','tops','top',
       'upgrade','upgraded','buy','gain','gains','record','high','highs','strong','boost','boosts',
       'win','wins','higher','climb','climbs','rise','rises','rebound','optimism','bullish','outperform'}
NEG = {'plunge','plunges','drop','drops','fall','falls','sink','sinks','slump','slumps','miss','misses',
       'downgrade','downgraded','sell','cut','cuts','warn','warns','warning','lawsuit','probe','decline',
       'declines','weak','loss','losses','lower','tumble','tumbles','fear','fears','crash','recession',
       'slowdown','layoff','layoffs','bearish','underperform','selloff','sell-off'}


def bars(sym, start):
    out, tok = [], None
    while True:
        p = {'timeframe': '1Day', 'start': start, 'adjustment': 'all', 'limit': 10000, 'feed': 'sip'}
        if tok: p['page_token'] = tok
        r = requests.get(f'https://data.alpaca.markets/v2/stocks/{sym}/bars', headers=H, params=p, timeout=40).json()
        out += r.get('bars', []); tok = r.get('next_page_token')
        if not tok: break
    return out


def news_window(symbols, start_iso, end_iso):
    """Pull news in [start,end); then STRICTLY re-filter created_at < end_iso in code (no look-ahead)."""
    arts, tok = [], None
    while True:
        p = {'symbols': symbols, 'start': start_iso, 'end': end_iso, 'limit': 50, 'sort': 'desc'}
        if tok: p['page_token'] = tok
        r = requests.get('https://data.alpaca.markets/v1beta1/news', headers=H, params=p, timeout=25).json()
        arts += r.get('news', []); tok = r.get('next_page_token')
        if not tok or len(arts) >= 200: break
    return [a for a in arts if a['created_at'] < end_iso]   # belt-and-suspenders point-in-time guard


def rsi(cl, n=14):
    if len(cl) < n + 1: return None
    d = [cl[i] - cl[i-1] for i in range(1, len(cl))]
    g = sum(max(x, 0) for x in d[-n:]) / n; l = sum(max(-x, 0) for x in d[-n:]) / n
    return 100.0 if l == 0 else round(100 - 100 / (1 + g / l), 1)


def sma(cl, n): return round(sum(cl[-n:]) / n, 2) if len(cl) >= n else None


def score_headlines(arts):
    scored = []
    pos = neg = 0
    for a in arts:
        words = set(a['headline'].lower().replace(',', ' ').replace('.', ' ').split())
        p = len(words & POS); n = len(words & NEG)
        s = 'pos' if p > n else ('neg' if n > p else 'neu')
        pos += (s == 'pos'); neg += (s == 'neg')
        scored.append({'ts': a['created_at'], 'symbols': a.get('symbols', [])[:5],
                       'headline': a['headline'], 'pos_hits': p, 'neg_hits': n, 'label': s})
    net = pos - neg
    return scored, {'count': len(arts), 'pos': pos, 'neg': neg, 'net': net}


def main():
    hist = bars('QQQ', (datetime.now(timezone.utc) - timedelta(days=420)).strftime('%Y-%m-%d'))
    if len(hist) < 160:
        print("not enough history"); return
    test = hist[-N_DAYS:]
    records = []
    open(LOG, 'w').close()   # fresh run
    print(f"probing {len(test)} days: {test[0]['t'][:10]} -> {test[-1]['t'][:10]}\n")
    for b in test:
        i = hist.index(b)
        prior = hist[:i]                    # strictly through T-1 close (NO look-ahead)
        cl = [x['c'] for x in prior]
        openT, closeT = b['o'], b['c']
        date = b['t'][:10]
        try:
            # --- point-in-time technical signals (prior closes + today's open) ---
            s7, s50, s150 = sma(cl, 7), sma(cl, 50), sma(cl, 150)
            sig = {
                'open': round(openT, 2), 'prev_close': round(cl[-1], 2),
                'gap_pct': round((openT / cl[-1] - 1) * 100, 2),
                'sma7': s7, 'sma50': s50, 'sma150': s150,
                'above_50d': openT > s50 if s50 else None,
                'above_150d': openT > s150 if s150 else None,
                'dist_150d_pct': round((openT / s150 - 1) * 100, 2) if s150 else None,
                'rsi14': rsi(cl), 'mom_1d': round((cl[-1] / cl[-2] - 1) * 100, 2),
                'mom_5d': round((cl[-1] / cl[-6] - 1) * 100, 2) if len(cl) > 6 else None,
                'realized_vol_20d': round(st.pstdev([math.log(cl[k]/cl[k-1]) for k in range(len(cl)-20, len(cl))]) * math.sqrt(252) * 100, 1) if len(cl) > 21 else None,
            }
            # --- point-in-time news (24h ending at today's OPEN, 13:30 UTC ~ 9:30 ET) ---
            end_iso = f"{date}T13:30:00Z"
            start_iso = (datetime.fromisoformat(date) - timedelta(days=1)).strftime('%Y-%m-%dT13:30:00Z')
            try:
                arts = news_window(BASKET, start_iso, end_iso)
                headlines, nsum = score_headlines(arts)
                news_err = None
            except Exception as e:
                headlines, nsum, news_err = [], {'count': 0, 'pos': 0, 'neg': 0, 'net': 0}, str(e)

            # --- candidate deterministic predictions (each scored separately later) ---
            regime_up = sig['above_150d']
            preds = {
                'gap_follow':  'BULL' if sig['gap_pct'] > 0 else 'BEAR',
                'trend':       'BULL' if regime_up and (sig['above_50d']) else 'BEAR',
                'rsi_revert':  ('BULL' if sig['rsi14'] and sig['rsi14'] < 40 else ('BEAR' if sig['rsi14'] and sig['rsi14'] > 60 else ('BULL' if regime_up else 'BEAR'))),
                'news':        ('BULL' if nsum['net'] > 0 else ('BEAR' if nsum['net'] < 0 else 'NEUTRAL')),
                'always_bull': 'BULL',
            }
            votes = [preds['trend'], preds['news'], preds['rsi_revert']]
            b_ = votes.count('BULL'); r_ = votes.count('BEAR')
            preds['combined'] = 'BULL' if b_ >= r_ else 'BEAR'

            # --- outcome (the truth) ---
            ret = (closeT / openT - 1) * 100
            actual = 'BULL' if ret > 0 else 'BEAR'
            hits = {k: (v == actual) for k, v in preds.items() if v in ('BULL', 'BEAR')}

            rec = {'date': date, 'decision_ts': end_iso,
                   'signals': sig,
                   'news': {**nsum, 'error': news_err, 'headlines': headlines},
                   'predictions': preds,
                   'outcome': {'open': round(openT, 2), 'close': round(closeT, 2),
                               'ret_pct': round(ret, 2), 'actual': actual},
                   'hits': hits}
        except Exception as e:
            rec = {'date': date, 'error': f"{type(e).__name__}: {e}"}
        records.append(rec)
        open(LOG, 'a').write(json.dumps(rec) + '\n')
        if 'error' in rec:
            print(f"{date}  ERROR {rec['error']}")
        else:
            print(f"{date}  ret {rec['outcome']['ret_pct']:+5.2f}% ({actual:4}) | news {nsum['count']:2d} net{nsum['net']:+d} | "
                  f"trend {preds['trend']:4} news {preds['news']:7} comb {preds['combined']:4}")

    # --- summary (the LOG is the deliverable; this is just a pulse check) ---
    good = [r for r in records if 'outcome' in r]
    base = sum(r['outcome']['actual'] == 'BULL' for r in good) / len(good) * 100
    print(f"\n=== {len(good)} days | BASE RATE {base:.1f}% BULL (always-bull baseline) ===")
    print(f"{'rule':<14}{'accuracy':>10}{'n':>6}")
    for rule in ['gap_follow', 'trend', 'rsi_revert', 'news', 'combined']:
        hh = [r['hits'][rule] for r in good if rule in r.get('hits', {})]
        if hh: print(f"{rule:<14}{sum(hh)/len(hh)*100:>9.1f}%{len(hh):>6}")
    print(f"\nfull tape -> {LOG}  ({len(records)} records; raw headlines + signals captured for later re-scoring)")


if __name__ == '__main__':
    main()
