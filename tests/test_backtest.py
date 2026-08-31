import numpy as np
import pandas as pd
import pytest

from stratlib.backtest import (
    single_asset_backtest, decile_backtest, cointegration_test, pairs_backtest,
)


def test_single_asset_backtest_perfect_foresight_beats_naive():
    # Construct a price series with a clean alternating trend, and a
    # signal that (by construction) always has the right sign one step
    # ahead. A "perfect foresight" signal should dramatically outperform
    # always being long.
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rets = np.tile([0.02, 0.02, -0.02, -0.02], 25)
    price = pd.Series(100 * np.cumprod(1 + rets), index=idx)

    perfect_signal = pd.Series(np.sign(rets), index=idx).shift(-1).fillna(0)  # knows next-period return's sign
    always_long = pd.Series(1.0, index=idx)

    r1 = single_asset_backtest(price, perfect_signal, cost_bps=0.0, freq=252)
    r2 = single_asset_backtest(price, always_long, cost_bps=0.0, freq=252)
    assert r1["ann_return"] > r2["ann_return"]


def test_single_asset_backtest_costs_reduce_return_with_high_turnover():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx)
    flip_flop = pd.Series(np.tile([1.0, -1.0], 100), index=idx)  # trades every single period

    zero_cost = single_asset_backtest(price, flip_flop, cost_bps=0.0)
    high_cost = single_asset_backtest(price, flip_flop, cost_bps=50.0)
    assert high_cost["ann_return"] < zero_cost["ann_return"]
    assert zero_cost["n_position_changes"] > 100


def test_single_asset_backtest_returns_expected_keys():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    price = pd.Series(np.linspace(100, 110, 50), index=idx)
    signal = pd.Series(1.0, index=idx)
    result = single_asset_backtest(price, signal)
    for key in ["returns", "signal", "ann_return", "ann_vol", "sharpe",
                "max_drawdown", "hit_rate", "n_position_changes", "pct_time_in_market"]:
        assert key in result


def test_decile_backtest_top_beats_bottom():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    assets = [f"A{i}" for i in range(30)]
    quality = rng.normal(0, 1, 30)
    signal = pd.DataFrame(quality[None, :] + rng.normal(0, 0.05, (6, 30)), index=dates, columns=assets)
    fwd_returns = pd.DataFrame(0.01 * quality[None, :] + rng.normal(0, 0.001, (6, 30)), index=dates, columns=assets)
    result = decile_backtest(signal, fwd_returns, n_deciles=10, min_names=10)
    assert (result["long_short"] > 0).all()


def test_cointegration_test_detects_cointegrated_series():
    rng = np.random.default_rng(2)
    n = 400
    b = np.cumsum(rng.normal(0, 1, n)) + 50
    a = 1.5 * b + rng.normal(0, 0.5, n)
    idx = pd.date_range("2020-01-01", periods=n)
    result = cointegration_test(pd.Series(a, index=idx), pd.Series(b, index=idx))
    assert result["coint_pvalue"] < 0.05
    assert result["hedge_ratio"] == pytest.approx(1.5, abs=0.1)


def test_pairs_backtest_runs_and_returns_expected_keys():
    rng = np.random.default_rng(3)
    n = 300
    b = np.cumsum(rng.normal(0, 0.01, n)) + 5
    a = b + rng.normal(0, 0.02, n)
    idx = pd.date_range("2020-01-01", periods=n)
    result = pairs_backtest(pd.Series(a, index=idx), pd.Series(b, index=idx), hedge_ratio=1.0, window=20)
    for key in ["returns", "zscore", "position", "sharpe", "n_position_changes", "pct_time_in_market"]:
        assert key in result
