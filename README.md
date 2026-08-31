# Strategy Lab: Which Trading Strategy, In Which Situation?

Six systematic trading strategies, each coded, backtested on real market
data across equities/crypto/FX, and put through the same four checks:
cross-asset performance, a bootstrap confidence interval, regime-conditional
performance, and parameter/walk-forward robustness — then synthesized into
an evidence-backed answer to the actual question a trader cares about:
**not "does this strategy work," but "which strategy, for which asset, in
which market regime, and how much should I trust the answer."**

**Start here → [`07_synthesis/strategy_selection_guide.ipynb`](./07_synthesis)**
— the capstone that cross-tabs all six strategies against regime and asset
class into a practical decision guide.

## The six strategies

| # | Strategy | Verdict | Strongest evidence |
|---|---|---|---|
| 1 | [Trend-following](./01_trend_following) | Real edge on BTC specifically | 90% CI [0.53, 2.48]; broad parameter plateau |
| 2 | [Mean-reversion](./02_mean_reversion) | Real edge on KO specifically | 90% CI [0.15, 1.34]; near-total loss on BTC (mirror image of #1) |
| 3 | [Cross-sectional momentum](./03_cross_sectional_momentum) | Real signal, fragile calibration | Monotonic decile spread confirmed; lookback sensitivity is a genuine concern |
| 4 | [Pairs trading](./04_pairs_trading) (XRP/DOGE) | Works, but on a knife-edge | Only one of five tested windows is positive, despite genuinely strong cointegration |
| 5 | [Volatility breakout](./05_volatility_breakout) | No aggregate edge found | CI spans zero; 35.5% walk-forward win rate — reported honestly as a negative result |
| 6 | [Seasonality](./06_seasonality) (BTC day-of-week) | Interesting but unproven | CI excludes zero and 71% walk-forward consistency — but drawn from 6 tested combinations |

## The core finding

The single clearest result in this repo: **trend-following and
mean-reversion are near-mirror-images of each other across the two assets
both were deeply tested on.** Trend-following is profitable in almost
every regime on BTC (a young, historically trending asset) and loses money
in every regime on KO (a stable, mature stock). Mean-reversion shows
almost the exact opposite pattern. Which asset you're trading predicts the
right strategy at least as well as which regime it's currently in.

## Why this repo also reports what *doesn't* work

Two of the six strategies here don't hold up under scrutiny in their
simplest form — volatility breakout shows no real edge in aggregate, and
pairs trading's "fix" (calibrating the window to the spread's measured
half-life) turns out to sit on a narrow parameter ridge, not a robust
plateau. Both are reported with the same depth of testing as the
strategies that worked, because a strategy-comparison project that only
shows winners isn't actually demonstrating a selection process — it's
just showing four backtests that happened to work.

## Architecture

A shared, unit-tested library (`stratlib/`) — not six copies of similar
backtest code — is what makes the final cross-strategy comparison a fair
one: every strategy is measured through the exact same performance
statistics, regime classifier, and robustness tools.

```
strategy-lab/
├── stratlib/
│   ├── stats.py         # perf_stats, bootstrap_ci, sharpe_ci
│   ├── strategies.py     # signal generators for all 6 strategies
│   ├── backtest.py       # single-asset, cross-sectional, and pairs backtest engines
│   ├── regime.py         # trend-strength + volatility regime classification
│   └── robustness.py     # parameter sensitivity + walk-forward validation
├── tests/                # 33 unit tests covering stratlib
├── requirements.txt      # pinned dependency versions
├── download_data.py      # fetches all source datasets into data/
├── 01_trend_following/
├── 02_mean_reversion/
├── 03_cross_sectional_momentum/
├── 04_pairs_trading/
├── 05_volatility_breakout/
├── 06_seasonality/
└── 07_synthesis/          # the decision-matrix capstone
```

One real bug was caught by the test suite during development: an early
version of `regime_conditional_returns` hardcoded a daily (252) return
frequency regardless of the data actually passed in, which silently
produced nonsensical annualized figures when fed monthly cross-sectional
momentum returns. A regression test (`test_regime_conditional_returns_respects_freq_argument`)
now checks this directly.

## Reproducing locally

```bash
git clone <this-repo>
cd strategy-lab
pip install -r requirements.txt
python download_data.py
python -m pytest tests/ -v     # confirm the library itself is correct
jupyter notebook                # open any notebook and run all cells
```

## Data sources (all public)

- **NYSE equities (2010–2016, ~500 tickers):** [kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
- **BTC/USD, ETH/USD:** [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization), [blockchain-unica/ethereum-ponzi](https://github.com/blockchain-unica/ethereum-ponzi)
- **XRP, DOGE (for pairs trading):** [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets) (Cryptocurrency folder)
- **FX daily rates (U.S. Federal Reserve H.10 release):** [datasets/exchange-rates](https://github.com/datasets/exchange-rates)

## Stack

Python, pandas, NumPy, SciPy, statsmodels, matplotlib, Jupyter, pytest.

## Related work

This repo is a companion to [quant-portfolio](https://github.com/bala-42/quant-portfolio),
which covers momentum, statistical arbitrage, derivatives pricing/risk, and
portfolio optimization in depth. This repo reuses several of the same real
data sources but is a standalone project focused specifically on
comparative strategy evaluation across regimes and asset classes.
