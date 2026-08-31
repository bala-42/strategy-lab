# Strategy 2: Short-Term Mean-Reversion (Contrarian)

**Notebook:** [`mean_reversion.ipynb`](./mean_reversion.ipynb)

Short when price is far above its rolling mean, long when far below —
the direct opposite bet from trend-following.

## Results

| Asset | Sharpe | Ann. return | Max drawdown | Trades |
|---|---|---|---|---|
| AAPL (equity, growth) | −0.28 | −6.0% | −49.4% | 217 |
| **KO (equity, staple)** | **0.73** | **8.9%** | −13.7% | 280 |
| BTC (crypto) | −0.96 | −77.5% | **−100.0%** | 605 |
| EUR/USD (FX) | −0.20 | −1.4% | −27.7% | 265 |

**KO Sharpe confidence interval:** 90% CI **[0.15, 1.34]** — solidly positive.

**Regime split (KO):** dominated by volatility level more than trend
classification — both Low-Vol regimes outperform both High-Vol regimes;
Trending-LowVol (Sharpe 1.41) even beats Ranging-LowVol (0.91).
Ranging-HighVol, the regime that sounds like this strategy's home turf,
is nearly flat.

**Parameter sensitivity:** entire grid (window × entry threshold)
positive, Sharpe 0.38–1.11 — a real plateau.

**Walk-forward:** 13 folds, 53.8% positive.

## Interpretation

A near-mirror-image of Strategy 1: mean-reversion loses badly on the
asset that trends hardest (BTC — a near-total-loss max drawdown) and
works on a stable, mature stock. The regime finding reframes the
strategy's own name: it's really a bet on *contained* volatility more
than on "ranging" price action specifically.

## Risks

Structurally exposed to a trend that simply doesn't revert (as BTC
shows, severely); more capacity-constrained than trend-following; no
protection against a large move being genuine new information rather
than noise; the walk-forward win rate (54%) is more modest than the
aggregate Sharpe alone suggests.
