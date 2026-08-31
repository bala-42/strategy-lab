"""
stratlib: shared research infrastructure for the strategy-lab project.

Six strategies (trend-following, mean-reversion, cross-sectional momentum,
pairs trading, volatility breakout, seasonality) all import from this
package rather than reimplementing backtest mechanics, statistics, or
regime classification independently. That's what makes the final
cross-strategy comparison in 07_synthesis a fair one: every strategy is
evaluated through the exact same performance-measurement and
regime-classification code.

Modules
-------
stats       : performance statistics and bootstrap confidence intervals
strategies  : signal generators for all six strategies
backtest    : generic backtest engines (single-asset, cross-sectional, pairs)
regime      : trend-strength and volatility-regime classification
robustness  : parameter sensitivity grids and walk-forward validation
"""
from . import stats, strategies, backtest, regime, robustness

__all__ = ["stats", "strategies", "backtest", "regime", "robustness"]
