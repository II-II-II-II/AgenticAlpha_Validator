#!/usr/bin/env python3
"""
vol_target_scanner.py — BUY-CHEAP-GAMMA target scanner (a few days -> 1 week horizon).

The thesis (see the brainstorm): a short-dated LONG option is a VOLATILITY bet, not a direction bet.
Edge = buy when implied vol is cheap relative to what the stock actually realizes, in names where
dealer positioning (GEX) is set to AMPLIFY the move. Direction only picks the structure later.

For each name it computes, on the nearest expiry ~4-9 calendar days out:
  - ATM implied vol (from the chain)
  - realized vol (20d, annualized)                          -> IV-RV spread (cheap if IV <= RV)
  - a proxy IV-percentile: current ATM IV vs trailing 1yr of rolling-20d realized vol
       (TRUE IV-Rank needs a paid historical-IV feed; this is an honest free proxy, labeled as such)
  - implied move to expiry (ATM straddle / spot) vs the stock's TYPICAL move over the same horizon
       -> move_ratio < 1 means options UNDERPRICE motion = cheap
  - GEX regime (negative gamma / below flip = PRIMED to move) via Black-Scholes gamma on the chain
  - LIQUIDITY GATE (hard): ATM bid/ask as % of premium + open interest

Composite score (higher = better BUY-vol target), z-scored across the scanned set:
  score = z(-IV_RV) + z(1 - move_ratio) + z(-iv_pctile) + z(-GEX_total)   [only among liquidity-PASS names]

Usage:
  python vol_target_scanner.py                 # curated ~200 liquid list
  python vol_target_scanner.py --limit 50      # first N of the list (fast demo)
  python vol_target_scanner.py AAPL NVDA MU     # explicit tickers
"""
import sys, argparse, warnings, logging, time
import numpy as np, pandas as pd
from scipy.stats import norm
import yfinance as yf
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)   # ETFs 404 on earnings calendar — harmless

# ---- data layer: Alpaca (reliable, real OI, IV-from-price) with Yahoo as fallback ----
try:
    import alpaca_vol
    _ALPACA_OK = True
except Exception:
    _ALPACA_OK = False
USE_ALPACA = _ALPACA_OK

THROTTLE = 0.15     # seconds between names (Alpaca doesn't burst-throttle like Yahoo option chains)


def _R(fn, tries=3, wait=1.5):
    """Retry a flaky network call with linear backoff (Yahoo throttles bursts)."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))

R = 0.04
WIN_LO, WIN_HI = 4, 9        # target expiry: calendar days to expiration
BAND = 0.10                  # strikes within +/-10% of spot for GEX
MAX_SPREAD = 0.08            # liquidity gate: ATM bid/ask <= 8% of mid
MIN_OI = 200                 # liquidity gate: ATM open interest

# ~200 most-liquid US optionable names (mega/large cap + high-volume single names + key ETFs)
UNIVERSE = """
AAPL MSFT NVDA AMZN GOOGL GOOG META TSLA AVGO AMD NFLX ADBE CRM INTC MU QCOM
TXN AMAT ORCL CSCO IBM NOW PANW SNOW PLTR SMCI ARM MRVL ON NXPI ASML TSM
JPM BAC WFC GS MS C SCHW AXP V MA PYPL COF BLK BX KKR SPGI
UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD CVS ISRG VRTX REGN MRNA
XOM CVX COP SLB EOG OXY MPC PSX VLO HAL DVN
HD LOW NKE SBUX MCD TGT WMT COST TJX LULU CMG BKNG MAR ABNB DIS
PG KO PEP PM MO CL KMB MDLZ
BA CAT DE GE HON LMT RTX UPS FDX MMM EMR ETN
T VZ TMUS CMCSA CHTR
BABA JD PDD NIO SHOP UBER LYFT DASH COIN HOOD SQ SOFI RIVN LCID F GM
CAT WDAY DDOG NET CRWD ZS MDB TEAM OKTA FTNT ANET DELL HPQ
SPY QQQ IWM DIA TLT HYG XLF XLE XLK XLV XLU XLI XLP XLY XLC XLB XLRE
SMH SOXX ARKK GLD SLV USO XBI KRE EEM FXI EWZ
""".split()


def gamma_bs(S, K, iv, T):
    d1 = (np.log(S / K) + (R + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return norm.pdf(d1) / (S * iv * np.sqrt(T))


def earn_days(tk, today):
    """Calendar days to next earnings (None if unknown). Free/defensive via yfinance calendar."""
    try:
        cal = tk.calendar
        ed = cal.get('Earnings Date') if isinstance(cal, dict) else None
        if ed:
            d = ed[0] if isinstance(ed, (list, tuple)) else ed
            return (pd.Timestamp(d).normalize() - today).days
    except Exception:
        pass
    return None


def analyze(name):
    tk = yf.Ticker(name)
    hist = _R(lambda: tk.history(period='1y', auto_adjust=True))['Close'].dropna()
    if len(hist) < 150:
        return None
    spot = float(hist.iloc[-1])
    lr = np.log(hist / hist.shift(1)).dropna()
    rv20 = float(lr.tail(20).std() * np.sqrt(252))                  # current realized vol
    roll_rv = (lr.rolling(20).std() * np.sqrt(252)).dropna()        # 1yr distribution of RV

    today = pd.Timestamp.now().normalize()
    all_exps = _R(lambda: tk.options)
    exps = [e for e in all_exps if WIN_LO <= (pd.Timestamp(e) - today).days <= WIN_HI]
    if not exps:
        # fall back to the nearest expiry beyond 2 days if nothing in the sweet spot
        exps = [e for e in all_exps if (pd.Timestamp(e) - today).days >= 2][:1]
    if not exps:
        return None
    exp = exps[0]
    cal_days = (pd.Timestamp(exp) - today).days
    T = max(cal_days, 1) / 365
    n_bd = max(int(round(cal_days * 5 / 7)), 1)                      # horizon in trading days
    ed = earn_days(tk, today)
    earn_in_win = ed is not None and 0 <= ed <= cal_days            # earnings before this expiry

    try:
        ch = _R(lambda: tk.option_chain(exp))
    except Exception:
        return None
    calls, puts = ch.calls, ch.puts
    if calls.empty or puts.empty:
        return None

    # --- ATM contracts (SAME strike for a clean straddle) ---
    c_atm = calls.iloc[(calls.strike - spot).abs().argmin()]
    K = float(c_atm.strike)
    p_match = puts[puts.strike == K]
    p_atm = p_match.iloc[0] if not p_match.empty else puts.iloc[(puts.strike - spot).abs().argmin()]
    atm_iv = np.nanmean([c_atm.impliedVolatility, p_atm.impliedVolatility])
    if not (atm_iv and atm_iv > 0.01):
        return None

    # --- liquidity gate (ATM) ---
    def spr(row):
        mid = (row.bid + row.ask) / 2
        return (row.ask - row.bid) / mid if mid > 0 else 9.99
    c_spr, p_spr = spr(c_atm), spr(p_atm)
    oi = float(c_atm.openInterest or 0) + float(p_atm.openInterest or 0)
    liquid = (c_spr <= MAX_SPREAD) and (p_spr <= MAX_SPREAD) and (oi >= MIN_OI)

    # --- implied move (ATM straddle) vs typical move over same horizon ---
    call_mid = float((c_atm.bid + c_atm.ask) / 2)
    put_mid = float((p_atm.bid + p_atm.ask) / 2)
    straddle = call_mid + put_mid
    impl_move = straddle / spot                                     # to-expiry expected move
    hist_move = float(np.log(hist / hist.shift(n_bd)).dropna().std())  # typical n-day move
    move_ratio = impl_move / hist_move if hist_move > 0 else np.nan  # <1 = options underprice motion

    iv_rv = atm_iv - rv20                                           # cheap if <= 0
    iv_pctile = float((roll_rv < atm_iv).mean())                    # proxy IV-rank (0..1)

    # --- GEX regime ---
    opts = []
    for df, sign in [(calls, +1), (puts, -1)]:
        d = df[['strike', 'openInterest', 'impliedVolatility']].dropna()
        d = d[(d.strike > spot * (1 - BAND)) & (d.strike < spot * (1 + BAND)) &
              (d.openInterest > 0) & (d.impliedVolatility > 0.01)]
        for _, x in d.iterrows():
            opts.append((float(x.strike), float(x.impliedVolatility), T, float(x.openInterest), sign))
    if len(opts) >= 12:
        gex_total = sum(s * gamma_bs(spot, K, iv, T) * oi_ * 100 * spot ** 2 * 0.01
                        for K, iv, T, oi_, s in opts)
    else:
        gex_total = np.nan

    return dict(name=name, spot=spot, exp=exp, dte=cal_days,
                strike=K, call_mid=call_mid, put_mid=put_mid, straddle=straddle,
                atm_iv=atm_iv, rv20=rv20, iv_rv=iv_rv, iv_pctile=iv_pctile,
                impl_move=impl_move, hist_move=hist_move, move_ratio=move_ratio,
                gex=gex_total, atm_spr=max(c_spr, p_spr), oi=oi, liquid=liquid,
                earn_days=ed, earn_in_win=bool(earn_in_win))


def zscore(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def rank(names, progress=False):
    """Scan `names`, return (ranked_liquid_df, all_scanned_df). Shared by the paper-logger."""
    src = alpaca_vol.analyze if USE_ALPACA else analyze
    rows = []
    for i, n in enumerate(names, 1):
        try:
            r = src(n)
        except Exception:
            r = None
        if progress:
            print(f"  [{i:>3}/{len(names)}] {n:<6} {'ok' if r else 'skip'}", end='\r')
        if r:
            rows.append(r)
        time.sleep(THROTTLE)                       # avoid Yahoo burst rate-limiting
    if progress:
        print(' ' * 60, end='\r')
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    liq = df[df.liquid].copy()
    if liq.empty:
        return liq, df
    liq['s_ivrv']  = zscore(-liq['iv_rv'])
    liq['s_move']  = zscore(1 - liq['move_ratio'])
    liq['s_ivpct'] = zscore(-liq['iv_pctile'])
    liq['s_gex']   = zscore(-liq['gex'].fillna(liq['gex'].median()))
    liq['score']   = liq[['s_ivrv', 's_move', 's_ivpct', 's_gex']].sum(axis=1)

    # ---- #1 TRIGGER LAYER: convert "primed" from a soft vote into a hard gate ----
    liq['cheap']  = liq['move_ratio'] < 1.0                       # options underprice the move
    liq['primed'] = liq['gex'].fillna(0) < 0                      # negative dealer gamma = amplifying
    liq['trigger'] = np.where(liq['primed'] & liq['earn_in_win'], 'GEX+EARN',
                     np.where(liq['primed'], 'GEX',
                     np.where(liq['earn_in_win'], 'EARN', '')))
    # qualifies for the TRIGGER signal: liquid + cheap + (primed OR catalyst in window)
    liq['passes_trigger'] = liq['cheap'] & (liq['primed'] | liq['earn_in_win'])
    return liq.sort_values('score', ascending=False), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tickers', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    names = [t.upper() for t in args.tickers] or UNIVERSE
    if args.limit:
        names = names[:args.limit]

    print(f"scanning {len(names)} names for CHEAP-GAMMA buy targets ({WIN_LO}-{WIN_HI}d expiry)...\n")
    liq, df = rank(names, progress=True)
    if df.empty:
        print("no candidates."); return
    if liq.empty:
        print("no names passed the liquidity gate."); return

    show = liq[['name', 'spot', 'dte', 'atm_iv', 'rv20', 'iv_rv', 'iv_pctile',
                'move_ratio', 'gex', 'atm_spr', 'score']].copy()
    show['atm_iv'] = (show['atm_iv'] * 100).round(0)
    show['rv20'] = (show['rv20'] * 100).round(0)
    show['iv_rv'] = (show['iv_rv'] * 100).round(0)
    show['iv_pctile'] = (show['iv_pctile'] * 100).round(0)
    show['move_ratio'] = show['move_ratio'].round(2)
    show['gex'] = (show['gex'] / 1e9).round(2)
    show['atm_spr'] = (show['atm_spr'] * 100).round(1)
    show['score'] = show['score'].round(2)
    show.columns = ['name', 'spot', 'dte', 'IV%', 'RV%', 'IV-RV', 'IVpct',
                    'moveR', 'GEX$B', 'spr%', 'SCORE']

    print(f"=== CHEAP-GAMMA targets ({len(liq)} liquid of {len(df)} scanned) ===")
    print("  IV-RV<0 & moveR<1 & IVpct low & GEX<0  => cheap vol, primed to move\n")
    print(show.head(20).to_string(index=False))
    print(f"\n  worst (richest / most-pinned, AVOID buying):")
    print(show.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
