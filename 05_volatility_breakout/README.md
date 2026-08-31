# Strategy 5: Volatility Breakout

**Notebook:** [`volatility_breakout.ipynb`](./volatility_breakout.ipynb)

Trades breakouts from a volatility "squeeze" — when trailing realized
volatility is in its lowest quartile and price makes a new local
high/low, take a position in the breakout direction for a short holding
period.

## Results

| Asset | Sharpe | Trades | % time in market |
|---|---|---|---|
| AAPL | −0.35 | 98 | 29% |
| XOM | −0.44 | 101 | 30% |
| BTC | −0.11 | 321 | 30% |
| EUR/USD | −0.80 | 688 | 78% |

**Negative on every asset tested — no exceptions.**

**BTC bootstrap CI:** [-0.46, 0.33] — comfortably spans zero, even on the
least-bad asset.

**Regime split (BTC):** no bucket clears Sharpe 0.5; one (Ranging-LowVol)
is clearly negative.

**Parameter sensitivity (48 combinations on BTC):** min −0.40, median
0.06, max 0.51, only 56.2% positive — scattered around zero, not a real
plateau.

**Walk-forward:** 31 folds, **35.5% positive — worse than a coin flip.**

## Interpretation

Every check in this notebook points the same direction: no convincing
edge for this signal, in this simple form, on any asset tested. This is
reported plainly rather than reframed, because a strategy-comparison
project that only shows winners would be missing the most common
real-world outcome of testing a plausible trading idea.

**Note from the synthesis notebook:** averaging across all regimes here
hides a real nuance — this strategy actually looks respectable
specifically in High-Vol regimes on both BTC and KO (see
[`07_synthesis`](../07_synthesis)), even though its aggregate verdict is
negative.

## What a real next step would look like

Higher-frequency (intraday) data, since breakout logic is more commonly
argued to work closer to the volatility event itself; a trend/momentum
filter on breakout direction rather than trading every squeeze-breakout
independently; volume/order-flow confirmation, which this price-only
dataset can't test.
