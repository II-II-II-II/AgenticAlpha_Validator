#!/usr/bin/env python3
"""
ETF LETF Ensemble — Live Paper-Trading Engine
=============================================
Daily, MOO-style paper trader for the validated LETF regime ensemble
(see letf_baseline_model.md). Mirrors the grandnagus.py setup: internal fill
simulation, SMS via Gmail->phone gateway, CSV logging. SAME brain as the backtest —
it imports build_weights() from letf_strategy.py so the live signal cannot drift.

FLOW (run ~9:35am ET each trading day, via cron):
  1. Market-open check (Alpaca clock) — skip holidays/weekends.
  2. Pull QQQ daily closes (signal) through the last CLOSED bar + today's OPENs (fills).
  3. build_weights(QQQ) -> target weights {TQQQ, SQQQ, TLT, cash}.  [live 150d/35/40]
     v4: the cash sleeve is parked in SHY (1-3yr Treasuries) instead of 0%-yield cash.
  4. Mark portfolio to today's open -> NAV; compute whole-share rebalance orders.
  5. Simulate fills at today's open; persist state.
  6. Log fills (etf_paper_trades.csv) + daily NAV/target (etf_paper_nav.csv).
  7. One SMS: regime, NAV, target, orders, holdings.

Backtest parity: same build_weights + fill-at-open. reconcile.py diffs live vs replay.

Usage:
  python etf_live_trader.py                 # one live cycle (for cron)
  python etf_live_trader.py --dry_run       # compute + print, no state/SMS change
  python etf_live_trader.py --force         # bypass market-open / once-per-day guards (testing)
  python etf_live_trader.py --init 6000     # set/reset starting capital
"""
import os, json, csv, smtplib, argparse
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import requests
import pandas as pd
import alphahconfig as cfg
from letf_strategy import build_weights

REF = 'QQQ'                       # regime reference
TRADED = ['TQQQ', 'SQQQ', 'TLT']  # signal instruments (SQQQ bought long — no shorting)
CASH_PROXY = 'SHY'                # v4: idle cash sleeve parked in 1-3yr Treasuries (earns ~T-bill yield vs 0% cash)
HELD = TRADED + [CASH_PROXY]      # everything we actually buy / hold / price / mark
SMA_BULL, SMA_SLOW, RSI_LO, RSI_BEAR = 150, 152, 35, 40   # v2: walk-forward-validated config (2026-06-21)
GAP_FADE_THR = 0.02   # v3: panic-fade overlay — QQQ open gap <= -2% -> FULL-BOOK 100% TQQQ until ~11am
STATE_FILE = 'etf_paper_state.json'
TRADE_LOG = 'etf_paper_trades.csv'
NAV_LOG = 'etf_paper_nav.csv'
DEFAULT_CAPITAL = 6000.0
DATA = 'https://data.alpaca.markets/v2/stocks'
PAPER = 'https://paper-api.alpaca.markets'


# ---------------- notifications (mirrors grandnagus.send_sms) ----------------
def send_sms(body):
    try:
        u = getattr(cfg, 'GMAIL_USER', None); p = getattr(cfg, 'GMAIL_APP_PASSWORD', None)
        to = getattr(cfg, 'PHONE_GATEWAY', None)
        if not (u and p and to):
            return
        msg = MIMEText(body); msg['Subject'] = 'ETF-BOT'; msg['From'] = u; msg['To'] = to
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(u, p); s.send_message(msg)
    except Exception as e:
        print(f"[!] SMS failure: {e}")


# ---------------- Alpaca ----------------
def _hdr():
    return {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY,
            'accept': 'application/json'}


def market_is_open():
    try:
        r = requests.get(f'{PAPER}/v2/clock', headers=_hdr(), timeout=15)
        return bool(r.json().get('is_open')) if r.status_code == 200 else False
    except Exception:
        return False


def daily_closes(symbol, lookback=400):
    start = (datetime.utcnow() - timedelta(days=lookback)).strftime('%Y-%m-%d')
    params = {'symbols': symbol, 'timeframe': '1Day', 'start': start, 'limit': 10000,
              'feed': 'sip', 'adjustment': 'all'}
    r = requests.get(f'{DATA}/bars', headers=_hdr(), params=params, timeout=30)
    if r.status_code == 403:
        params['feed'] = 'iex'; r = requests.get(f'{DATA}/bars', headers=_hdr(), params=params, timeout=30)
    bars = r.json().get('bars', {}).get(symbol, [])
    s = pd.Series({pd.Timestamp(b['t']).tz_convert('US/Eastern').tz_localize(None).normalize(): b['c'] for b in bars})
    today = pd.Timestamp(datetime.now().date())
    return s[s.index < today].sort_index()          # only CLOSED bars (drop today's partial)


def todays_opens(symbols):
    """Today's official open per symbol (falls back to prev close if not yet available)."""
    params = {'symbols': ','.join(symbols), 'feed': 'sip'}
    r = requests.get(f'{DATA}/snapshots', headers=_hdr(), params=params, timeout=20)
    if r.status_code == 403:
        params['feed'] = 'iex'; r = requests.get(f'{DATA}/snapshots', headers=_hdr(), params=params, timeout=20)
    snaps = r.json().get('snapshots', r.json())
    out = {}
    for sym in symbols:
        s = snaps.get(sym, {})
        o = (s.get('dailyBar') or {}).get('o')
        if not o:
            o = (s.get('prevDailyBar') or {}).get('c')      # fallback (e.g. market closed)
        out[sym] = float(o) if o else None
    return out


def current_prices(symbols):
    """Latest trade price per symbol — used for the ~11am fade revert (intraday, not the open)."""
    params = {'symbols': ','.join(symbols), 'feed': 'sip'}
    r = requests.get(f'{DATA}/snapshots', headers=_hdr(), params=params, timeout=20)
    if r.status_code == 403:
        params['feed'] = 'iex'; r = requests.get(f'{DATA}/snapshots', headers=_hdr(), params=params, timeout=20)
    snaps = r.json().get('snapshots', r.json())
    out = {}
    for sym in symbols:
        s = snaps.get(sym, {})
        p = (s.get('latestTrade') or {}).get('p') or (s.get('dailyBar') or {}).get('c')
        out[sym] = float(p) if p else None
    return out


# ---------------- state + logging ----------------
def load_state(init_capital=None):
    if init_capital is not None or not os.path.exists(STATE_FILE):
        cap = init_capital if init_capital is not None else DEFAULT_CAPITAL
        return {'cash': cap, 'positions': {t: 0 for t in TRADED}, 'start_cap': cap,
                'inception': datetime.now().strftime('%Y-%m-%d'), 'last_run': None, 'prev_nav': cap,
                'fade_active': False, 'fade_revert_target': None}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(st):
    with open(STATE_FILE, 'w') as f:
        json.dump(st, f, indent=2)


def log_trade(row):
    exists = os.path.isfile(TRADE_LOG)
    with open(TRADE_LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


def log_nav(row):
    exists = os.path.isfile(NAV_LOG)
    with open(NAV_LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------- signal ----------------
def target_weights(qqq):
    px = pd.DataFrame({REF: qqq})
    for c in TRADED:
        px[c] = 0.0                       # placeholder cols (signal only reads QQQ)
    W = build_weights(px, sma_bull=SMA_BULL, sma_slow=SMA_SLOW, rsi_lo=RSI_LO, rsi_bear=RSI_BEAR, s2_mode='rsi')
    last = W.iloc[-1]
    tw = {t: round(float(last[t]), 4) for t in TRADED}
    cash_w = round(1 - sum(tw.values()), 4)
    tw[CASH_PROXY] = cash_w        # v4: park the idle sleeve in SHY instead of 0%-yield cash
    tw['CASH'] = 0.0               # only whole-share rounding residual stays as literal cash
    bull = qqq.iloc[-1] > qqq.rolling(SMA_BULL).mean().iloc[-1]
    regime = 'BULL' if bull else 'BEAR'
    return tw, regime


def _rebalance(pos, cash, tw, prices):
    """Whole-share rebalance to target weights tw at given prices. Returns (orders, new_pos, new_cash, nav)."""
    nav = cash + sum(pos.get(t, 0) * prices[t] for t in HELD)
    orders, new_pos, new_cash = [], dict(pos), cash
    for t in HELD:
        tgt_sh = int((tw.get(t, 0.0) * nav) // prices[t])
        delta = tgt_sh - pos.get(t, 0)
        if delta != 0:
            orders.append((t, delta, prices[t])); new_cash -= delta * prices[t]; new_pos[t] = tgt_sh
    return orders, new_pos, round(new_cash, 2), nav


def _fmt_orders(orders):
    return '; '.join(f"{'BUY' if d > 0 else 'SELL'} {abs(d)} {t} @ ${p:.2f}" for t, d, p in orders) or 'no change'


# ---------------- main cycle ----------------
def run_cycle(args):
    today = datetime.now().strftime('%Y-%m-%d')
    if not args.force and not market_is_open():
        print(f"[*] {today}: market closed — skipping."); return
    st = load_state(args.init)

    # ===== REVERT PHASE (~11am): unwind the full-book fade back to the ensemble target =====
    if args.revert:
        if not st.get('fade_active'):
            print(f"[*] {today}: no active panic-fade to revert."); return
        rev = st.get('fade_revert_target') or {}
        prices = current_prices(HELD)
        if any(prices.get(t) is None for t in HELD):
            print(f"[!] Missing prices for revert: {prices}"); return
        pos = {t: int(st['positions'].get(t, 0)) for t in HELD}
        orders, new_pos, new_cash, nav = _rebalance(pos, float(st['cash']), rev, prices)
        hold = ', '.join(f"{new_pos[t]} {t}" for t in HELD if new_pos[t]) or 'all cash'
        body = (f"ETF-BOT {today} — FADE REVERT (~11am)\nUnwinding 100% TQQQ -> ensemble target\n"
                f"NAV ${nav:,.0f}\nOrders: {_fmt_orders(orders)}\nHolding: {hold}, ${new_cash:,.0f} cash")
        print(body)
        if args.dry_run:
            print("\n[*] DRY RUN — revert (no changes)."); return
        for t, d, p in orders:
            log_trade({'date': today, 'ticker': t, 'side': 'BUY' if d > 0 else 'SELL', 'shares': abs(d),
                       'price': round(p, 4), 'value': round(abs(d) * p, 2), 'regime': 'FADE_REVERT'})
        st.update({'cash': new_cash, 'positions': new_pos, 'fade_active': False, 'fade_revert_target': None})
        save_state(st); send_sms(body)
        print("\n[*] Fade reverted. State saved."); return

    # ===== OPEN PHASE (~9:35am) =====
    if not args.force and st.get('last_run') == today and not args.dry_run:
        print(f"[*] Already ran today ({today}). Use --force to re-run."); return
    qqq = daily_closes(REF)
    if len(qqq) < SMA_SLOW + 5:
        print(f"[!] Not enough QQQ history ({len(qqq)})."); return
    tw, regime = target_weights(qqq)
    opens = todays_opens(HELD + [REF])
    if any(opens.get(t) is None for t in HELD + [REF]):
        print(f"[!] Missing opens: {opens}"); return

    # ---- v3 panic-fade overlay (full-book): QQQ open gap <= -2% -> 100% TQQQ until ~11am ----
    gap = opens[REF] / qqq.iloc[-1] - 1
    fade = gap <= -GAP_FADE_THR
    fade_base = dict(tw)
    if fade:
        tw = {'TQQQ': 1.0, 'SQQQ': 0.0, 'TLT': 0.0, CASH_PROXY: 0.0, 'CASH': 0.0}

    pos = {t: int(st['positions'].get(t, 0)) for t in HELD}
    orders, new_pos, new_cash, nav_open = _rebalance(pos, float(st['cash']), tw, opens)
    day_ret = nav_open / st.get('prev_nav', nav_open) - 1 if st.get('prev_nav') else 0.0
    total_ret = nav_open / st['start_cap'] - 1
    tag = 'PANIC_FADE' if fade else regime

    hold_str = ', '.join(f"{new_pos[t]} {t}" for t in HELD if new_pos[t]) or 'all cash'
    tgt_str = ', '.join(f"{int(tw[k]*100)}% {k}" for k in (*TRADED, CASH_PROXY, 'CASH') if tw.get(k, 0) > 0)
    fade_str = f"\n*** PANIC FADE: QQQ gap {gap*100:.1f}% -> FULL-BOOK 100% TQQQ (revert ~11am) ***" if fade else ""
    body = (f"ETF-BOT {today}\nRegime: {regime}{fade_str}\nNAV ${nav_open:,.0f} ({total_ret*100:+.1f}% since "
            f"{st['inception']})\nTarget: {tgt_str}\nOrders: {_fmt_orders(orders)}\nHolding: {hold_str}, ${new_cash:,.0f} cash")
    print(body)
    if args.dry_run:
        print("\n[*] DRY RUN — no state/log/SMS changes."); return

    for t, d, p in orders:
        log_trade({'date': today, 'ticker': t, 'side': 'BUY' if d > 0 else 'SELL', 'shares': abs(d),
                   'price': round(p, 4), 'value': round(abs(d) * p, 2), 'regime': tag})
    log_nav({'date': today, 'nav': round(nav_open, 2), 'day_return': round(day_ret, 6),
             'total_return': round(total_ret, 6), 'regime': tag,
             **{f'tw_{k}': tw.get(k, 0) for k in (*TRADED, CASH_PROXY, 'CASH')},
             **{f'pos_{t}': new_pos.get(t, 0) for t in HELD}, 'cash': new_cash})
    st.update({'cash': new_cash, 'positions': new_pos, 'last_run': today, 'prev_nav': round(nav_open, 2),
               'fade_active': bool(fade), 'fade_revert_target': fade_base if fade else None})
    save_state(st); send_sms(body)
    print(f"\n[*] Logged + SMS sent.{' PANIC FADE active — revert run will fire ~11am.' if fade else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--revert', action='store_true', help='~11am phase: unwind an active panic-fade to the ensemble target')
    ap.add_argument('--init', type=float, default=None, help='reset starting capital (e.g. 6000)')
    run_cycle(ap.parse_args())


if __name__ == '__main__':
    main()
