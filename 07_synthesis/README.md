# Synthesis: Which Strategy, In Which Situation?

**Notebook:** [`strategy_selection_guide.ipynb`](./strategy_selection_guide.ipynb)

The capstone. Pulls the evidence from all six strategy notebooks together
into an evidence-backed answer to "which strategy, for which asset, in
which regime" — with an explicit confidence level attached to each
recommendation, not just a strategy name.

## The core finding

The clearest, best-evidenced result in the whole repo: **trend-following
and mean-reversion are near-mirror-images of each other across the two
assets both were deeply tested on.** A regime × strategy Sharpe matrix,
computed identically on BTC and KO:

**BTC (trending, volatile):**

| Regime | Trend-following | Mean-reversion | Vol-breakout |
|---|---|---|---|
| Ranging-HighVol | 0.70 | −1.47 | 0.48 |
| Ranging-LowVol | 0.42 | −0.52 | −0.73 |
| Trending-HighVol | 0.02 | −1.10 | 0.10 |
| Trending-LowVol | **4.41** | −0.81 | −0.07 |

**KO (stable, mature):**

| Regime | Trend-following | Mean-reversion | Vol-breakout |
|---|---|---|---|
| Ranging-HighVol | −0.54 | 0.01 | **1.35** |
| Ranging-LowVol | −0.85 | **0.91** | −1.24 |
| Trending-HighVol | −0.19 | 0.51 | 0.76 |
| Trending-LowVol | −0.67 | **1.41** | −1.30 |

Trend-following is positive almost everywhere on BTC and negative almost
everywhere on KO. Mean-reversion is close to the exact opposite. **Which
asset you're trading predicts the right strategy at least as well as
which regime it's currently in.**

## The nuance regime-conditioning surfaces

Volatility breakout showed no aggregate edge in its own notebook — but
the matrix above shows it's actually respectable specifically in
High-Vol regimes on both assets (0.48–1.35). An aggregate "this doesn't
work" verdict can hide a real, narrower pattern; the regime breakdown is
what surfaces it.

## The practical decision guide (full version in the notebook)

| Situation | Lean | Confidence |
|---|---|---|
| Trending, volatile asset, low-vol phase | Trend-following | High |
| Same asset, high-vol/choppy | Neither — explicitly flagged as unknown | None |
| Stable, mature asset, low-vol phase | Mean-reversion | Moderate-high |
| Stable, mature asset, high-vol phase | Volatility breakout (needs its own validation) | Low-moderate |
| Cross-sectional equity universe | Cross-sectional momentum, cautiously | Moderate |
| Two cointegrated assets | Pairs trading, only after checking window sensitivity on that specific pair | Low |
| Calendar-based signal | Investigate, don't deploy | Low |

## Risk and robustness comparison

Two strategies (trend-following, mean-reversion) showed genuinely broad
parameter plateaus. Three (cross-sectional momentum, pairs trading,
volatility breakout) showed real fragility once actually checked — a
distinction invisible from a single well-chosen backtest, and the reason
`stratlib.robustness` exists as shared infrastructure rather than being
skipped.

## What this project demonstrates

Not "here are six strategies that make money" — two don't show a real
edge in their tested form. What's demonstrated instead: the same rigor,
applied consistently across all six, sorts real findings from fragile
ones; strategy selection is conditional, not absolute; and a negative or
fragile result, reported with the same care as a positive one, is more
useful than silence.
