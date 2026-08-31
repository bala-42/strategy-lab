# Strategy 1: Trend-Following (Time-Series Momentum)

**Notebook:** [`trend_following.ipynb`](./trend_following.ipynb)

Long if an asset's trailing return over a lookback window is positive,
short if negative — a standalone bet on an asset's own direction, tested
across equity, crypto, and FX.

## Results

| Asset | Sharpe | Ann. return | Max drawdown | Trades |
|---|---|---|---|---|
| AAPL (equity) | 0.24 | 6.0% | −35.5% | 63 |
| **BTC (crypto)** | **1.31** | **115.5%** | −82.5% | 182 |
| EUR/USD (FX) | −0.29 | −2.3% | −36.5% | 844 |

**BTC Sharpe confidence interval:** 90% CI **[0.54, 2.45]** — does not
span zero.

**Regime split (BTC):** Trending-LowVol dominates (Sharpe ~4.4);
Trending-HighVol is nearly flat (Sharpe ~0.02) despite also being
"trending" — volatility matters as much as trend classification.

**Parameter sensitivity:** positive across every lookback tested (20–252
days) — a broad plateau, not a lucky single value.

**Walk-forward:** 31 folds, 61.3% positive — a real majority, but far
from every period, with the strongest folds concentrated in BTC's most
extreme historical bull runs.

## Interpretation

A genuine, statistically-supported edge on BTC, structurally absent on
FX. The regime split is the key nuance: "trending" alone isn't enough —
it needs to be paired with contained volatility, or the whipsaw inside a
noisy trend erodes the edge.

## Risks

Whipsaw in high-vol trending regimes; performance concentrated in a
handful of strong episodes rather than spread evenly; high capacity/crowding
risk (one of the most widely traded systematic styles); transaction costs
matter more on choppier assets like FX.
