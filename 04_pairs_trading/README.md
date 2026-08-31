# Strategy 4: Statistical Arbitrage / Pairs Trading

**Notebook:** [`pairs_trading.ipynb`](./pairs_trading.ipynb)

Tests a pair for cointegration before trading a mean-reversion strategy
on the spread. Reuses the companion quant-portfolio project's findings —
BTC/ETH fails cointegration; a systematic 21-pair scan found XRP/DOGE
genuinely cointegrates.

## Results

**BTC/ETH:** cointegration p = 0.409 — fails, correctly not traded.

**XRP/DOGE:** cointegration p = 0.0013, ADF p = 0.0002, R² = 0.939, n =
2,275 — a strong, well-supported relationship.

**Half-life-calibrated backtest (window = 120 days, motivated by a
measured ~49-day mean-reversion half-life):** Sharpe 0.33, ann. return
36.5%, max drawdown −91.8%. 90% bootstrap CI **[-0.35, 1.82]** — spans zero.

**Regime split (on the spread itself):** best in Trending-HighVol
(Sharpe ~3.4) — plausibly because a "trending" spread, for a
mean-reverting instrument, often means an active reversion in progress
rather than the relationship breaking down.

**Window sensitivity — the key new finding versus the earlier version of
this analysis:** only window=120 is positive among {60, 90, 120, 150,
180}; the other four are all negative, several sharply so. This is a
narrow ridge, not a robust plateau.

## Interpretation

Diagnosing *why* a naive backtest failed (measuring the half-life) is a
different, weaker claim than proving the resulting fix generalizes. The
window-sensitivity check clears that bar honestly: it doesn't. The half-life
calibration was a real, principled step — it just wasn't sufficient on
its own to establish robustness.

## Risks

Parameter fragility is the headline risk here, more than for any other
strategy in this repo. Severe drawdown (~-90%) persists even in the best
configuration. Relationship breakdown risk is real and undetectable in
real time except through worsening P&L after the fact.
