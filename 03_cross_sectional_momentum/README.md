# Strategy 3: Cross-Sectional Momentum

**Notebook:** [`cross_sectional_momentum.ipynb`](./cross_sectional_momentum.ipynb)

Rank ~500 NYSE stocks by trailing 12-1 month return each month; long the
top decile, short the bottom decile. Unlike Strategy 1, this is a bet on
relative performance within a universe, not an asset's own direction.

## Results

71 monthly rebalances, 2011-01-31 to 2016-11-30, on 467 tickers with
complete history.

**Long-short Sharpe:** 0.275. **90% bootstrap CI: [-0.41, 1.06]** — spans
zero, despite a confirmed monotonic decile spread (the signal carries
real information; the specific long-short Sharpe just isn't estimated
precisely enough from 71 observations to call proven).

**Market-regime split:** noisier than the single-asset regime splits
elsewhere in this repo (each bucket has only 9–31 monthly observations),
but suggestively better in Trending market regimes than Ranging ones.

**Robustness — the most concerning finding in this notebook:** Sharpe
swings from ~0.04 (9-month lookback) to ~0.30 (15-month lookback) across
a lookback sweep — a real range, not a flat plateau, and non-monotonic.
Decile-count sensitivity (tercile vs. decile cuts) is more stable.

## Interpretation

The underlying signal is real (confirmed by the monotonic pattern), but
the specific "12-1 month" convention borrowed from the academic
literature (Jegadeesh & Titman, 1993) shows real sensitivity in this
sample — worth treating as a genuine design choice to validate, not an
assumed-optimal setting.

## Risks

Momentum-crash risk on the short leg during reversals (a real, documented
phenomenon — Daniel & Moskowitz, 2016); survivorship bias in this
backtest (complete-history tickers only); high capacity/crowding risk as
one of the most widely traded equity factors; turnover/transaction costs
not modeled in the gross figures above.
