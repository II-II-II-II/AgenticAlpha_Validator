#!/usr/bin/env python3
"""
confluence.py — indicator library for the Reddit 5-indicator "confluence" strategy test.

Reimplements (from published formulas) the five indicators named in the Reddit post:
  1. MACD                    (ChrisMoody's is a cosmetic reskin of standard MACD)
  2. Squeeze Momentum        (LazyBear — open source, formula published)
  3. ADX                     (BOSWaves' "Volatility Waves" wraps a standard ADX)
  4. SuperTrend              (LuxAlgo's "AI" layer k-means-selects the factor; we test
                              the base indicator, and sweep the factor instead — the
                              adaptive version is in-sample fitting by construction)
  5. Price-action structure  (LuxAlgo "Pure Price Action" is closed source; approximated
                              by break-of-structure over the prior swing high)

EVERY indicator is computed from CLOSED bars only and shifted before use by the harness,
so nothing can repaint or peek. All functions take/return numpy arrays.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- resampling
RTH_START, RTH_END = "09:30", "15:55"


def load_5m(path, min_bars_per_day=60):
    """1-min OHLCV csv -> regular-trading-hours 5-minute bars. Drops thin days."""
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    df = df.between_time(RTH_START, "15:59")
    if df.empty:
        return None
    good = df.groupby(df.index.date).size()
    good = set(d for d, n in good.items() if n >= min_bars_per_day)
    if not good:
        return None
    df = df[[d in good for d in df.index.date]]
    out = df.resample("5min").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum"}).dropna()
    out = out.between_time(RTH_START, RTH_END)
    return out if len(out) > 300 else None


# ---------------------------------------------------------------- primitives
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rma(x, n):
    """Wilder's smoothing."""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def stdev(x, n):
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy()


def rolling_linreg_end(y, n):
    """Value of an OLS fit over the trailing n bars, evaluated at the last bar.
    (This is Pine's linreg(src, n, 0).)"""
    y = np.asarray(y, dtype=float)
    if len(y) < n:
        return np.full(len(y), np.nan)
    x = np.arange(n, dtype=float)
    sx, sxx = x.sum(), (x * x).sum()
    sy = pd.Series(y).rolling(n).sum().to_numpy()
    sxy = np.full(len(y), np.nan)
    sxy[n - 1:] = np.correlate(y, x, mode="valid")
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a + b * (n - 1)


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


# ---------------------------------------------------------------- indicators
def macd_bull(c, fast=12, slow=26, sig=9):
    line = ema(c, fast) - ema(c, slow)
    signal = ema(line, sig)
    return (line - signal) > 0


def squeeze_momentum(h, l, c, n=20, bb_mult=2.0, kc_mult=1.5):
    """LazyBear. Returns (momentum_positive, squeeze_on)."""
    basis, dev = sma(c, n), bb_mult * stdev(c, n)
    ub, lb = basis + dev, basis - dev
    ma = sma(c, n)
    rng = rma(true_range(h, l, c), n)
    ukc, lkc = ma + kc_mult * rng, ma - kc_mult * rng
    squeeze_on = (lb > lkc) & (ub < ukc)
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    ref = (((hh + ll) / 2.0) + sma(c, n)) / 2.0
    mom = rolling_linreg_end(c - ref, n)
    return mom > 0, squeeze_on


def adx(h, l, c, n=14):
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(plus, n) / tr
        mdi = 100 * rma(minus, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(np.nan_to_num(dx), n)


def supertrend_bull(h, l, c, period=10, factor=3.0):
    """Standard SuperTrend; True where trend is up. Iterative (band memory)."""
    atr = rma(true_range(h, l, c), period)
    hl2 = (h + l) / 2.0
    upper, lower = hl2 + factor * atr, hl2 - factor * atr
    n = len(c)
    fu, fl = np.copy(upper), np.copy(lower)
    dirn = np.ones(n, dtype=bool)
    for i in range(1, n):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            dirn[i] = True
        elif c[i] < fl[i - 1]:
            dirn[i] = False
        else:
            dirn[i] = dirn[i - 1]
    return dirn


def structure_bull(h, l, c, lookback=20):
    """Break-of-structure proxy for LuxAlgo 'Pure Price Action': bullish once price
    closes above the prior swing high, bearish once it closes below the prior swing low.
    Uses only CLOSED prior bars (shifted) so it cannot repaint."""
    ph = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    pl = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    state = np.zeros(len(c), dtype=bool)
    cur = False
    for i in range(len(c)):
        if not np.isnan(ph[i]) and c[i] > ph[i]:
            cur = True
        elif not np.isnan(pl[i]) and c[i] < pl[i]:
            cur = False
        state[i] = cur
    return state


# ---------------------------------------------------------------- signal stack
def signals(df, p):
    """Return dict of boolean 'bullish' arrays, one per indicator."""
    h, l, c = (df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
    mom, _sq = squeeze_momentum(h, l, c, p["sqz_n"], 2.0, 1.5)
    return {
        "macd": macd_bull(c, p["macd_f"], p["macd_s"], p["macd_sig"]),
        "sqz": mom,
        "st": supertrend_bull(h, l, c, p["st_p"], p["st_f"]),
        "adx": adx(h, l, c, p["adx_n"]) > p["adx_thr"],
        "pa": structure_bull(h, l, c, p["pa_n"]),
    }


DEFAULTS = dict(macd_f=12, macd_s=26, macd_sig=9, sqz_n=20,
                st_p=10, st_f=3.0, adx_n=14, adx_thr=20.0, pa_n=20)

# Ablation ladder: 1 -> 5 indicators, added in the order a trader would stack them.
LADDER = [
    ("L1_macd",            ["macd"]),
    ("L2_+squeeze",        ["macd", "sqz"]),
    ("L3_+supertrend",     ["macd", "sqz", "st"]),
    ("L4_+adx",            ["macd", "sqz", "st", "adx"]),
    ("L5_+priceaction",    ["macd", "sqz", "st", "adx", "pa"]),
]
