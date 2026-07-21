# qc_live — a validation-first quant research & live-trading system

A personal research engine for finding (and, more often, *disproving*) trading edges — paired with a
live paper-trading strategy, a multi-agent LLM market analyst, and a real-time dashboard. Built local-first
on free data (Alpaca, Yahoo) and local LLMs (ollama).

**The whole point of this repo is the methodology, not a get-rich claim.** Almost everything I tested
*failed*, and I kept the receipts. What survived did so only after passing controls designed to stop me
from fooling myself. That discipline — not any single strategy — is what this project demonstrates.

---

## The philosophy: validation-first, kill your darlings

Every idea here was held to the same bar before it earned a line of production code:

- **Beat the *right* benchmark, not a rigged one.** A momentum strategy on today's large-caps beats the
  S&P easily — *from survivorship bias alone*. So the benchmark isn't the S&P; it's an **equal-weight hold
  of the same universe**, which carries the identical bias. Only the *excess* over that is a real edge.
- **No look-ahead.** Signals use only data available at the decision moment; returns are measured forward.
- **Across regimes, walk-forward.** A strategy that shines in one market and dies in the next is noise
  wearing a costume. Several ideas "passed" a single-regime backtest and then failed the full history.
- **Costs and skew are not optional.** A 90%-win-rate strategy that hides one catastrophic loss is a
  *negative-skew trap*, not an edge.

The `research/graveyard/` directory is a first-class citizen: **a dozen plausible strategies, tested and
killed, with the reason documented.** Reading it is the fastest way to understand how the system thinks.

---

## What's inside

### 1. QC4 — the validated live strategy (`letf_strategy.py`, `etf_live_trader.py`)
A regime-switching leveraged-ETF ensemble: four transparent sub-models (trend, dip-buy, defensive short)
gated by a slow moving-average + RSI regime signal, holding leveraged Nasdaq exposure (TQQQ) in bull
regimes and rotating to cash/bonds/inverse when the trend breaks. Runs live in paper.

| 2018–2026 | CAGR | Max DD | Sharpe |
|---|---|---|---|
| **QC4** | **+24%** | −38% | **0.92** |
| QQQ buy-hold | +20% | −37% | 0.90 |
| SPY | +14% | −32% | 0.83 |

*Honest read:* it doubles the market's return and made money in 2018 while the market fell — but it's
**leveraged, with a −38% drawdown**, and its risk-adjusted edge over simply holding QQQ is thin. The
value is *intelligent leverage* (a regime gate that lets you use 3× without the Sharpe collapsing), not
a free lunch. *The regime gate is deliberately simple and slow — clever, fast regime detectors overfit.*

### 2. Cross-sectional momentum — the validated research edge (`research/momentum.py`)
Buy the 12-month winners (skip the last month), hold top-N equal-weight, rebalance monthly.

| 2018–2026, top-10 | CAGR | Sharpe | vs survivorship-controlled benchmark |
|---|---|---|---|
| **Momentum** | **+21%** | **1.06** | **+6%/yr excess** |

The **only** strategy that beat the survivorship-controlled benchmark, and the edge is *monotonic in N*
(a factor signature, not overfitting). *Caveats documented in the module:* one broad regime, no
momentum-crash in-sample, tax drag from turnover, and — tested — **do not regime-gate it** (momentum
already self-adapts; the gate whipsaws it).

### 3. The Oracle — a multi-agent LLM market analyst (`oracle_debate.py`, `oracle_agents/`)
A LangGraph debate: four analyst agents (bull / bear / macro / skeptic), each a separate local LLM with a
tunable markdown persona, argue over a daily direction call → a judge renders a verdict. Every forecast is
logged and **scored against the actual outcome** the next day. Explicitly framed as a *track-record
experiment*: the honest prior is that daily direction is ~a coin flip, and the logging exists to *prove or
disprove* the agents against that base rate — not to trade on them.

### 4. Live dashboard (`dashboard.py`)
A single-view Flask dashboard: QC4 allocation, QQQ/VIX/TQQQ moving averages with live prices (Yahoo,
matches broker across pre/regular/post-market), news, and backtest performance toggles. LAN-accessible.

---

## Repo structure

```
qc_live/
├── letf_strategy.py        # QC4 core (regime-gated LETF ensemble) + backtest CLI
├── etf_live_trader.py      # QC4 live paper-trading engine (scheduled)
├── dashboard.py            # Flask dashboard (localhost:8787, scheduled)
├── oracle_debate.py        # multi-agent LangGraph Oracle (scheduled 9:25 ET)
├── oracle_forecast.py      # Oracle signal/news/scoring infrastructure
├── oracle_agents/          # agent personas as tunable markdown (bull/bear/macro/skeptic/judge)
├── research/
│   ├── momentum.py         # the validated cross-sectional momentum backtest
│   └── graveyard/          # every strategy tested and KILLED, with the reason
├── alphahconfig.example.py # config template (copy to alphahconfig.py, gitignored)
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
cp alphahconfig.example.py alphahconfig.py     # then fill in your Alpaca keys (+ optional others)

python research/momentum.py                    # the validated momentum backtest + current picks
python letf_strategy.py                        # QC4 backtest vs honest benchmarks
python dashboard.py                            # dashboard at http://localhost:8787
```
The Oracle's analyst LLMs run locally via [ollama](https://ollama.com); the strategy/data code needs only
free Alpaca + Yahoo access.

## What the graveyard killed (a partial list)
Intraday dip-buy · sell-high/buy-low band swings · ORB breakouts (census + gap + volume filters + a full
stop-size sweep) · TQQQ/SQQQ leverage flipping · a cheap-gamma options straddle scanner · dual-momentum
ETF rotation · regime-gated momentum · 15-minute LLM direction prediction. Each has a documented reason —
mostly *negative skew*, *no edge after costs*, or *worked in one regime only*.

---

## Disclaimer
This is a **personal research and paper-trading project**, not investment advice, not a solicitation, and
not a live fund. Backtested results are hypothetical and include documented biases and caveats. Leveraged
ETFs carry severe drawdown and decay risk. Nothing here should be traded with real money without your own
independent validation.
