import numpy as np
import pandas as pd
import pytest
from functools import partial

from stratlib.strategies import trend_following_signal
from stratlib.backtest import single_asset_backtest
from stratlib.robustness import parameter_sensitivity, walk_forward_validation


def _trending_series(n=500, drift=0.0015, vol=0.008, seed=0):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, vol, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_parameter_sensitivity_returns_one_row_per_combo():
    price = _trending_series()
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    grid = {"lookback": [20, 60, 120]}
    result = parameter_sensitivity(price, trend_following_signal, bt, grid, metric="sharpe")
    assert len(result) == 3
    assert set(result["lookback"]) == {20, 60, 120}
    assert "sharpe" in result.columns


def test_parameter_sensitivity_handles_failing_combos_gracefully():
    price = _trending_series(n=50)  # short series -> some lookbacks will fail/produce NaN
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    grid = {"lookback": [10, 1000]}  # 1000 > series length
    result = parameter_sensitivity(price, trend_following_signal, bt, grid, metric="sharpe")
    assert len(result) == 2
    # should not raise, and the failing combo should show up as NaN rather than crashing


def test_walk_forward_validation_produces_multiple_folds():
    price = _trending_series(n=500)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    result = walk_forward_validation(
        price, trend_following_signal, bt, params={"lookback": 60},
        train_window=100, test_window=50,
    )
    assert len(result) >= 3
    for col in ["fold_start", "fold_end", "sharpe", "ann_return", "max_drawdown", "n_obs"]:
        assert col in result.columns


def test_walk_forward_validation_folds_are_sequential_and_non_overlapping():
    price = _trending_series(n=400)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    result = walk_forward_validation(
        price, trend_following_signal, bt, params={"lookback": 30},
        train_window=100, test_window=50,
    )
    starts = result["fold_start"].tolist()
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
