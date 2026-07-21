#!/usr/bin/env python3
"""
alpaca_vol.py — Alpaca-powered data layer for the cheap-gamma scanner (replaces Yahoo option chains).

Why: Yahoo rate-limits hard on bulk option-chain pulls (162/173 skipped on the 7/6 run). Alpaca free
tier gives us, reliably and without that throttling:
  - option QUOTES (bid/ask) via the indicative snapshot/quotes feed   -> straddle, moveR, spreads
  - OPEN INTEREST via the trading /v2/options/contracts endpoint       -> GEX (real dealer gamma)
Alpaca free does NOT provide IV/greeks (OPRA is paid), so we BACK IT OUT of the option price via
Black-Scholes inversion. Underlying daily history + earnings still come from yfinance (light, 1 call,
not the bottleneck).

analyze(name) returns the SAME dict shape as vol_target_scanner.analyze(), so rank() works unchanged.
"""
import numpy as np, pandas as pd, requests, warnings, logging
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import alphahconfig as cfg

warnings.filterwarnings('ignore'); logging.getLogger('yfinance').setLevel(logging.CRITICAL)

DATA = 'https://data.alpaca.markets'
TRADE = 'https://paper-api.alpaca.markets'
HDR = {'APCA-API-KEY-ID': cfg.ALPACA_KEY_ID, 'APCA-API-SECRET-KEY': cfg.ALPACA_SECRET_KEY}
R = 0.04
WIN_LO, WIN_HI = 4, 9
BAND = 0.10
MAX_SPREAD = 0.08
MIN_OI = 200


def _get(url, params):
    r = requests.get(url, headers=HDR, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_occ(sym):
    """AAPL260710C00195000 -> (root, 'YYYY-MM-DD', 'C'/'P', strike)."""
    strike = int(sym[-8:]) / 1000
    typ = sym[-9]
    ymd = sym[-15:-9]
    return sym[:-15], f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", typ, strike


def bs_price(S, K, T, iv, typ):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if typ == 'C' else (K - S))
    d1 = (np.log(S / K) + (R + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if typ == 'C':
        return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, typ):
    intrinsic = max(0.0, (S - K) if typ == 'C' else (K - S))
    if price <= intrinsic + 1e-4 or T <= 0:
        return np.nan
    try:
        return brentq(lambda v: bs_price(S, K, T, v, typ) - price, 1e-3, 5.0, maxiter=60)
    except Exception:
        return np.nan


def gamma_bs(S, K, iv, T):
    d1 = (np.log(S / K) + (R + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return norm.pdf(d1) / (S * iv * np.sqrt(T))


def contracts_in_window(name, today):
    """OI-bearing contracts expiring in [today+WIN_LO, today+WIN_HI]."""
    lo = (today + pd.Timedelta(days=WIN_LO)).date().isoformat()
    hi = (today + pd.Timedelta(days=WIN_HI)).date().isoformat()
    out, tok = [], None
    while True:
        p = {'underlying_symbols': name, 'expiration_date_gte': lo, 'expiration_date_lte': hi,
             'limit': 10000}
        if tok:
            p['page_token'] = tok
        j = _get(f"{TRADE}/v2/options/contracts", p)
        out += j.get('option_contracts', [])
        tok = j.get('next_page_token')
        if not tok:
            break
    return out


def latest_quotes(symbols):
    """Batch latest option quotes -> {symbol: (bid, ask)}."""
    q = {}
    for i in range(0, len(symbols), 100):                       # batch to keep URLs sane
        chunk = symbols[i:i + 100]
        j = _get(f"{DATA}/v1beta1/options/quotes/latest",
                 {'symbols': ','.join(chunk), 'feed': 'indicative'})
        for s, v in j.get('quotes', {}).items():
            q[s] = (float(v.get('bp', 0) or 0), float(v.get('ap', 0) or 0))
    return q


def analyze(name):
    # --- underlying (yfinance: accurate, light) ---
    hist = yf.Ticker(name).history(period='1y', auto_adjust=True)['Close'].dropna()
    if len(hist) < 150:
        return None
    spot = float(hist.iloc[-1])
    lr = np.log(hist / hist.shift(1)).dropna()
    rv20 = float(lr.tail(20).std() * np.sqrt(252))
    roll_rv = (lr.rolling(20).std() * np.sqrt(252)).dropna()
    today = pd.Timestamp.now().normalize()

    # --- option contracts (Alpaca: OI) in the target window ---
    try:
        cons = contracts_in_window(name, today)
    except Exception:
        return None
    if not cons:
        return None
    # choose the single expiry with the most contracts (the liquid weekly)
    byexp = {}
    for c in cons:
        byexp.setdefault(c['expiration_date'], []).append(c)
    exp = max(byexp, key=lambda e: len(byexp[e]))
    cons = byexp[exp]
    cal_days = (pd.Timestamp(exp) - today).days
    T = max(cal_days, 1) / 365
    n_bd = max(int(round(cal_days * 5 / 7)), 1)

    # --- quotes for those contracts (Alpaca) ---
    syms = [c['symbol'] for c in cons]
    try:
        quotes = latest_quotes(syms)
    except Exception:
        return None
    # build a per-strike table: strike -> {'C':(bid,ask,oi), 'P':(...)}
    tbl = {}
    for c in cons:
        b, a = quotes.get(c['symbol'], (0, 0))
        if a <= 0:
            continue
        _, _, typ, strike = parse_occ(c['symbol'])
        oi = c.get('open_interest')
        oi = float(oi) if oi not in (None, '', '0') else 0.0
        tbl.setdefault(strike, {})[typ] = (b, a, oi)
    strikes = sorted(k for k, v in tbl.items() if 'C' in v and 'P' in v)
    if not strikes:
        return None

    # --- ATM straddle ---
    K = min(strikes, key=lambda k: abs(k - spot))
    cb, ca, coi = tbl[K]['C']
    pb, pa, poi = tbl[K]['P']
    call_mid, put_mid = (cb + ca) / 2, (pb + pa) / 2
    straddle = call_mid + put_mid
    if straddle <= 0:
        return None

    def spr(b, a):
        m = (b + a) / 2
        return (a - b) / m if m > 0 else 9.99
    liquid = (spr(cb, ca) <= MAX_SPREAD) and (spr(pb, pa) <= MAX_SPREAD) and (coi + poi >= MIN_OI)

    # --- IV (invert BS on the ATM call & put, average) ---
    ivc = implied_vol(call_mid, spot, K, T, 'C')
    ivp = implied_vol(put_mid, spot, K, T, 'P')
    atm_iv = np.nanmean([ivc, ivp])
    if not (atm_iv and atm_iv > 0.01):
        return None

    impl_move = straddle / spot
    hist_move = float(np.log(hist / hist.shift(n_bd)).dropna().std())
    move_ratio = impl_move / hist_move if hist_move > 0 else np.nan
    iv_rv = atm_iv - rv20
    iv_pctile = float((roll_rv < atm_iv).mean())

    # --- GEX: per-strike IV (inverted) x OI x gamma, calls +, puts - ---
    gex_total, npts = 0.0, 0
    for k in strikes:
        if not (spot * (1 - BAND) < k < spot * (1 + BAND)):
            continue
        for typ, sign in (('C', +1), ('P', -1)):
            b, a, oi = tbl[k][typ]
            if oi <= 0 or a <= 0:
                continue
            iv = implied_vol((b + a) / 2, spot, k, T, typ)
            if not (iv and iv > 0.01):
                continue
            gex_total += sign * gamma_bs(spot, k, iv, T) * oi * 100 * spot ** 2 * 0.01
            npts += 1
    gex = gex_total if npts >= 6 else np.nan

    # --- earnings (yfinance) ---
    ed = None
    try:
        cal = yf.Ticker(name).calendar
        e = cal.get('Earnings Date') if isinstance(cal, dict) else None
        if e:
            dd = e[0] if isinstance(e, (list, tuple)) else e
            ed = (pd.Timestamp(dd).normalize() - today).days
    except Exception:
        pass
    earn_in_win = ed is not None and 0 <= ed <= cal_days

    return dict(name=name, spot=spot, exp=exp, dte=cal_days,
                strike=K, call_mid=call_mid, put_mid=put_mid, straddle=straddle,
                atm_iv=atm_iv, rv20=rv20, iv_rv=iv_rv, iv_pctile=iv_pctile,
                impl_move=impl_move, hist_move=hist_move, move_ratio=move_ratio,
                gex=gex, atm_spr=max(spr(cb, ca), spr(pb, pa)), oi=coi + poi, liquid=liquid,
                earn_days=ed, earn_in_win=bool(earn_in_win))


if __name__ == '__main__':
    import sys
    for n in (sys.argv[1:] or ['AAPL', 'NVDA', 'AMD', 'TLT']):
        r = analyze(n)
        if not r:
            print(f"{n:<6} (skip)")
            continue
        print(f"{n:<6} spot {r['spot']:.2f} exp {r['exp']} K {r['strike']:.1f}  "
              f"straddle ${r['straddle']:.2f}  IV {r['atm_iv']*100:.0f}%/RV {r['rv20']*100:.0f}%  "
              f"moveR {r['move_ratio']:.2f}  GEX {('%.2fB'%(r['gex']/1e9)) if r['gex']==r['gex'] else 'na'}  "
              f"OI {r['oi']:.0f}  liq {r['liquid']}")
