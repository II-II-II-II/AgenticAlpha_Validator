#!/usr/bin/env python3
"""
vol_paper_log.py — forward paper-log + ACTIVE management for the CHEAP-GAMMA buy-vol scanner.

Instrument: same-strike ATM LONG STRADDLE (call+put), a pure volatility bet (no direction call).
The thesis is "implied < realized" — if the scanner is right, the actual move should exceed the
premium we paid.

We do NOT just hold to expiry. Every open position is walked forward on DAILY High/Low bars and the
straddle is repriced via Black-Scholes (a long straddle is monotonic in |S-K|, so daily H/L bound its
whole intraday range -> intraday exits are detectable with NO 1-min feed). Exit on:
  - TARGET  : straddle value up >= +60%  -> take the profit (this is how you actually trade long vol)
  - STOP    : value down <= -50%          -> cut it
  - EXPIRY  : reached expiration           -> settle at intrinsic |S_exp - K|
Fills assume a resting limit AT the target/stop level; entry pays mid + 4% (you never get the mid).

Every trade logs a BUY and a SELL leg so it reads like a broker statement:
  python vol_paper_log.py --log [--top 5] [--limit N | TICKERS...]   # snapshot today's top signals (BUY)
  python vol_paper_log.py --manage                                    # mark opens, take profits/stops/expiry (SELL)
  python vol_paper_log.py --ledger                                    # human-readable BUY->SELL per trade
  python vol_paper_log.py --report                                    # expectancy over closed trades
  python vol_paper_log.py --show                                      # raw jsonl

Nothing here touches real money or QC4. Goal: >=30 closed trades, positive expectancy AFTER costs.
"""
import argparse, json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import norm

import vol_target_scanner as vs
import alphahconfig as cfg

R_FREE = 0.04

ET = ZoneInfo('America/New_York')
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'vol_paper_trades.jsonl')

SLIP = 0.04           # per-side slippage as fraction of straddle premium (conservative)
TARGET = 0.60         # take profit at +60% on the premium
STOP = -0.50          # cut at -50%


def today_et():
    return datetime.now(ET).date()


def load():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_all(rows):
    with open(LOG, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def append(row):
    with open(LOG, 'a') as f:
        f.write(json.dumps(row) + '\n')


def _bs_straddle(S, K, iv, T):
    """Black-Scholes value of a same-strike call+put (a straddle)."""
    if T <= 0:
        return abs(S - K)
    srt = iv * np.sqrt(T)
    d1 = (np.log(S / K) + (R_FREE + 0.5 * iv ** 2) * T) / srt
    d2 = d1 - srt
    disc = np.exp(-R_FREE * T)
    call = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    put = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def _day_extremes(S_hi, S_lo, K, iv, T):
    """Straddle value at the day's price extremes -> (intraday_max, intraday_min).
    A long straddle rises with |S-K|, so its intraday MAX is at the further extreme and its MIN is at
    the price nearest K. Daily High/Low therefore bound the whole intraday range (no 1-min feed needed)."""
    v_hi, v_lo = _bs_straddle(S_hi, K, iv, T), _bs_straddle(S_lo, K, iv, T)
    vmax = max(v_hi, v_lo)
    vmin = _bs_straddle(K, K, iv, T) if S_lo <= K <= S_hi else min(v_hi, v_lo)
    return vmax, vmin


def do_log(names, top):
    print(f"scanning {len(names)} names...")
    liq, df = vs.rank(names, progress=True)
    if liq.empty:
        print("no liquid candidates today — nothing logged."); return
    open_keys = {(r['name'], r['exp']) for r in load() if r['status'] == 'OPEN'}
    d = today_et().isoformat()

    def emit(r, mode):
        if (r['name'], r['exp']) in open_keys:                 # don't double up the same position
            return 0
        entry_fill = float(r['straddle'] * (1 + SLIP))
        row = dict(
            status='OPEN', action='BUY', signal_mode=mode, scan_date=d,
            logged_at=datetime.now(ET).isoformat(timespec='seconds'),
            name=r['name'], exp=r['exp'], dte=int(r['dte']),
            strike=float(r['strike']), spot_in=float(r['spot']),
            call_in=float(r['call_mid']), put_in=float(r['put_mid']),
            straddle_mid_in=float(r['straddle']), entry_fill=entry_fill,
            atm_iv=float(r['atm_iv']), rv20=float(r['rv20']), iv_rv=float(r['iv_rv']),
            iv_pctile=float(r['iv_pctile']), move_ratio=float(r['move_ratio']),
            gex=None if pd.isna(r['gex']) else float(r['gex']), score=float(r['score']),
            # --- #1 trigger metadata ---
            trigger=r['trigger'], primed=bool(r['primed']), cheap=bool(r['cheap']),
            earn_days=None if pd.isna(r['earn_days']) else int(r['earn_days']),
            passes_trigger=bool(r['passes_trigger']),
            # exit fields, filled by --manage:
            exit_date=None, exit_reason=None, exit_value=None,
            spot_exit=None, pnl=None, pnl_pct=None, win=None, held_days=None,
        )
        append(row); open_keys.add((r['name'], r['exp']))
        print(f"  BUY[{mode:<10}] {r['name']:<6} exp {r['exp']}  K={r['strike']:.1f}  "
              f"straddle ${r['straddle']:.2f}->${entry_fill:.2f}  trig={r['trigger'] or '-':<8} "
              f"moveR {r['move_ratio']:.2f}  score {r['score']:.2f}")
        return 1

    primed = liq[liq.passes_trigger].head(top)                 # PRIMARY: cheap + primed/catalyst
    n1 = sum(emit(r, 'trigger') for _, r in primed.iterrows())
    ctrl = liq[(~liq.passes_trigger) & liq.cheap].head(3)      # CONTROL: cheap but NOT primed (A/B)
    n2 = sum(emit(r, 'cheap_ctrl') for _, r in ctrl.iterrows())
    if primed.empty:
        print("  note: NO names passed the trigger gate today (cheap + primed/catalyst) — a real 'no-setup' day.")
    print(f"logged {n1} trigger + {n2} control." if (n1 + n2) else "  (nothing new to log)")


def _close(r, reason, exit_value, spot_exit=None, on=None):
    entry = r['entry_fill']
    exit_d = on or today_et()
    pnl = exit_value - entry
    r.update(status='CLOSED', exit_date=exit_d.isoformat(), exit_reason=reason,
             exit_value=float(exit_value), spot_exit=spot_exit,
             pnl=float(pnl), pnl_pct=float(pnl / entry * 100), win=bool(pnl > 0),
             held_days=(exit_d - pd.Timestamp(r['scan_date']).date()).days)
    print(f"  SELL {r['name']:<6} {reason:<7} recv ${exit_value:.2f} vs paid ${entry:.2f}  "
          f"= {pnl:+.2f} ({r['pnl_pct']:+.0f}%)  [{exit_d} held {r['held_days']}d]")


def do_manage():
    """
    Walk each OPEN trade forward day-by-day on DAILY High/Low (free, no membership) and reprice the
    straddle via Black-Scholes (entry IV frozen, T decaying). Because a long straddle is monotonic in
    |S-K|, the day's High/Low bound its intraday range -> we detect an intraday +60% target or -50%
    stop from daily data alone. Fills assume a resting limit AT the level (poll-timing-independent,
    conservative). Time domain: the ENTRY day is skipped (its high may predate our fill = look-ahead).
    """
    rows = load()
    acted = 0
    for r in rows:
        if r['status'] != 'OPEN':
            continue
        exp = pd.Timestamp(r['exp']).date()
        entry_day = pd.Timestamp(r['scan_date']).date()
        K, iv, entry = r['strike'], r['atm_iv'], r['entry_fill']
        tgt_val, stop_val = entry * (1 + TARGET), entry * (1 + STOP)
        try:
            ohlc = yf.Ticker(r['name']).history(start=str(entry_day),
                                                end=str(exp + pd.Timedelta(days=3)),
                                                auto_adjust=True)[['High', 'Low', 'Close']].dropna()
        except Exception:
            continue
        if ohlc.empty:
            continue
        cutoff = min(today_et(), exp)
        ohlc = ohlc[[ts.date() <= cutoff for ts in ohlc.index]]
        for ts, bar in ohlc.iterrows():
            d = ts.date()
            if d <= entry_day:                      # skip entry day (no intraday look-ahead)
                continue
            if d >= exp:                            # expiry -> settle intrinsic at that close
                _close(r, 'EXPIRY', abs(float(bar.Close) - K), spot_exit=float(bar.Close), on=d)
                acted += 1; break
            T = max((exp - d).days, 0) / 365
            vmax, vmin = _day_extremes(float(bar.High), float(bar.Low), K, iv, T)
            if vmax >= tgt_val:                     # intraday touched the profit target
                _close(r, 'TARGET', tgt_val, spot_exit=float(bar.Close), on=d); acted += 1; break
            if vmin <= stop_val:                    # intraday touched the stop
                _close(r, 'STOP', stop_val, spot_exit=float(bar.Close), on=d); acted += 1; break
        # else: still open
    save_all(rows)
    print(f"managed: {acted} exit(s); "
          f"{sum(1 for r in rows if r['status']=='OPEN')} still open.")


def do_ledger():
    rows = load()
    if not rows:
        print("no trades yet."); return
    print(f"{'name':<6}{'BUY date':<12}{'K':>8}{'paid':>8}   {'SELL date':<12}{'why':<8}{'recv':>8}{'P/L%':>8}  held")
    print('-' * 88)
    for r in rows:
        buy = f"{r['name']:<6}{r['scan_date']:<12}{r['strike']:>8.1f}{r['entry_fill']:>8.2f}"
        if r['status'] == 'OPEN':
            print(f"{buy}   {'(open)':<12}{'':<8}{'':>8}{'':>8}")
        else:
            print(f"{buy}   {r['exit_date']:<12}{r['exit_reason']:<8}{r['exit_value']:>8.2f}"
                  f"{r['pnl_pct']:>+8.0f}  {r['held_days']}d")


def do_report():
    rows = load()
    df = pd.DataFrame([r for r in rows if r['status'] == 'CLOSED'])
    n_open = sum(1 for r in rows if r['status'] == 'OPEN')
    if df.empty:
        print(f"no closed trades yet. ({n_open} open)"); return
    n = len(df); wins = int(df.win.sum()); wr = wins / n * 100
    exp_pct = df.pnl_pct.mean(); tot = df.pnl.sum()
    aw = df[df.win].pnl_pct.mean() if wins else 0
    al = df[~df.win].pnl_pct.mean() if wins < n else 0
    print(f"=== BUY-CHEAP-GAMMA paper results  ({n} closed, {n_open} open) ===")
    print(f"  win rate:        {wr:.0f}%  ({wins}/{n})")
    print(f"  expectancy:      {exp_pct:+.1f}% per trade  (after {SLIP*100:.0f}%/side slippage)")
    print(f"  avg win / loss:  {aw:+.0f}%  /  {al:+.0f}%")
    print(f"  total P/L:       ${tot:+.2f}  (1 straddle each)")
    print(f"  exit mix:        " + ", ".join(f"{k} {v}" for k, v in df.exit_reason.value_counts().items()))
    print(f"  verdict:         {'EDGE (positive after costs)' if exp_pct > 0 else 'NO EDGE (theta wins)'}"
          f"   [need >=30; have {n}]")
    cheap = df[df.move_ratio < 1]; rich = df[df.move_ratio >= 1]
    if len(cheap) and len(rich):
        print(f"  moveR<1 (cheap): {cheap.pnl_pct.mean():+.1f}% exp, {cheap.win.mean()*100:.0f}% win (n={len(cheap)})")
        print(f"  moveR>=1(rich):  {rich.pnl_pct.mean():+.1f}% exp, {rich.win.mean()*100:.0f}% win (n={len(rich)})")
    # ---- #1 the A/B: does the trigger gate beat cheap-only? ----
    if 'signal_mode' in df.columns:
        print("  --- A/B (trigger gate vs cheap-only control) ---")
        for m in ['trigger', 'cheap_ctrl']:
            sub = df[df.signal_mode == m]
            if len(sub):
                print(f"  [{m:<10}]  {sub.pnl_pct.mean():+.1f}% exp, {sub.win.mean()*100:.0f}% win  (n={len(sub)})")


def do_notify():
    """Morning summary of the straddle book -> SMS + email (mirrors the QC4 text)."""
    do_manage()                                   # process overnight exits first
    import smtplib
    from email.mime.text import MIMEText
    rows = load()
    op = [r for r in rows if r['status'] == 'OPEN']
    cl = [r for r in rows if r['status'] == 'CLOSED']
    d = today_et().isoformat()
    new_today = [r for r in op if r['scan_date'] == d]
    exits_today = [r for r in cl if r.get('exit_date') == d]
    realized = sum(r.get('pnl', 0) or 0 for r in cl)

    marks = []
    for r in op:
        try:
            spot = float(yf.Ticker(r['name']).history(period='1d')['Close'].iloc[-1])
            T = max((pd.Timestamp(r['exp']).date() - today_et()).days, 0) / 365
            val = _bs_straddle(spot, r['strike'], r['atm_iv'], T)
            marks.append((r['name'], (val / r['entry_fill'] - 1) * 100, (val - r['entry_fill']) * 100))
        except Exception:
            pass
    marks.sort(key=lambda m: -m[1])
    open_unreal = sum(m[2] for m in marks)

    L = [f"STRADDLE PAPER {d}",
         f"open {len(op)} | closed {len(cl)} | realized ${realized:+.0f}"]
    if new_today:
        L.append("NEW: " + ", ".join(f"{r['name']}({r.get('trigger') or r.get('signal_mode')})" for r in new_today))
    L.append("EXITS: " + (", ".join(f"{r['name']} {r['exit_reason']} {r['pnl_pct']:+.0f}%" for r in exits_today)
                          if exits_today else "none"))
    if marks:
        L.append(f"open unrealized ${open_unreal:+.0f}")
        L.append(f"best {marks[0][0]} {marks[0][1]:+.0f}% | worst {marks[-1][0]} {marks[-1][1]:+.0f}%")
        L.append(" ".join(f"{m[0]}{m[1]:+.0f}%" for m in marks))
    body = "\n".join(L)
    print(body)
    try:
        to = [cfg.PHONE_GATEWAY, getattr(cfg, 'ALERT_EMAIL', '')]
        msg = MIMEText(body); msg['Subject'] = 'STRADDLE-BOT'
        msg['From'] = cfg.GMAIL_USER; msg['To'] = ', '.join(to)
        s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        s.login(cfg.GMAIL_USER, cfg.GMAIL_APP_PASSWORD)
        s.sendmail(cfg.GMAIL_USER, to, msg.as_string()); s.quit()
        print("[sent]")
    except Exception as e:
        print(f"[send failed: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tickers', nargs='*')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--manage', action='store_true')
    ap.add_argument('--ledger', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--notify', action='store_true')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    if a.notify:
        do_notify()
    if a.manage:
        do_manage()
    if a.ledger:
        do_ledger()
    if a.report:
        do_report()
    if a.show:
        for r in load():
            print(json.dumps(r))
    if a.log or (not any([a.notify, a.manage, a.ledger, a.report, a.show])):
        names = [t.upper() for t in a.tickers] or vs.UNIVERSE
        if a.limit:
            names = names[:a.limit]
        do_log(names, a.top)


if __name__ == '__main__':
    main()
