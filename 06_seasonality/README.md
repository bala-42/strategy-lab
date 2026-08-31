# Strategy 6: Calendar / Seasonality Effects

**Notebook:** [`seasonality.ipynb`](./seasonality.ipynb)

Uses no price history at all — only the calendar date. Two variants
tested: turn-of-month (long the last day of the month through the first
3 of the next) and day-of-week (long Monday/Tuesday only), across three
assets — 6 combinations total.

## Results

| Effect | AAPL | KO | BTC |
|---|---|---|---|
| Turn-of-month | Sharpe 0.08 | Sharpe −0.26 | Sharpe 0.61 |
| Day-of-week | Sharpe 0.18 | Sharpe −0.11 | **Sharpe 0.79** |

**BTC day-of-week — the strongest result in the six-way grid:** 90%
bootstrap CI **[0.29, 1.40]** — excludes zero. Walk-forward: 31 folds,
**71.0% positive** — the most temporally consistent result anywhere in
this repo.

## Interpretation — treat with real skepticism despite the strength

Six combinations were tested before finding this result; with that many
comparisons, the chance of at least one producing a CI that excludes
zero purely by chance is elevated well above what a single pre-registered
test would imply. There's also a plausibility problem: BTC trades
continuously with no exchange open/close cycle, so the institutional-flow
mechanisms usually cited for equity day-of-week effects don't obviously
transfer.

The walk-forward consistency (71% across 31 independent windows spanning
2011–2026) is the fact that keeps this from being dismissed outright — a
pure multiple-comparisons artifact wouldn't typically be this stable
across such different market regimes. But "worth investigating further"
and "proven" remain different claims, and only the former is supported
here.

## Risks

Multiple-comparisons/data-mining risk is the central concern for this
entire strategy category, more than for any other strategy in this repo.
No clear economic mechanism identified for the strongest result. Uses no
risk management responsive to price action — a defining feature and a
real limitation if the pattern ever decays.

## A real next step

Test the BTC day-of-week effect against a completely fresh period not
used in any analysis above, and look for a crypto-specific mechanism
(e.g. exchange/fund weekly settlement cycles) before treating the
pattern as tradeable.
