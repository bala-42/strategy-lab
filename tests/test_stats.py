import numpy as np
import pandas as pd
import pytest

from stratlib.stats import perf_stats, max_drawdown, bootstrap_ci, sharpe_ci


def test_perf_stats_constant_return():
    # A constant monthly return of 1% should annualize to (1.01)^12 - 1
    r = pd.Series([0.01] * 24)
    stats = perf_stats(r, freq=12)
    assert stats["ann_return"] == pytest.approx(1.01 ** 12 - 1, rel=1e-9)
    assert stats["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert stats["hit_rate"] == 1.0


def test_perf_stats_with_name():
    r = pd.Series([0.01, -0.02, 0.03])
    stats = perf_stats(r, freq=12, name="my_strategy")
    assert stats["name"] == "my_strategy"


def test_perf_stats_empty_raises():
    with pytest.raises(ValueError):
        perf_stats(pd.Series(dtype=float), freq=12)


def test_max_drawdown_known_path():
    # Growth path: 1 -> 1.10 -> 0.99 -> 1.05  => returns are 10%, -10%, +6.06...%
    prices = pd.Series([1.0, 1.10, 0.99, 1.05])
    returns = prices.pct_change().dropna()
    dd = max_drawdown(returns)
    # trough is 0.99 vs peak 1.10 -> drawdown = 0.99/1.10 - 1
    assert dd == pytest.approx(0.99 / 1.10 - 1, rel=1e-9)


def test_bootstrap_ci_contains_point_estimate_direction():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.05, 200))
    result = bootstrap_ci(r, lambda s: s.mean(), n_boot=500, seed=1)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert result["ci_level"] == 0.90
    assert result["n_boot"] == 500


def test_bootstrap_ci_too_short_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(pd.Series([0.01, 0.02]), lambda s: s.mean())


def test_sharpe_ci_wraps_perf_stats():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.02, 0.04, 100))
    result = sharpe_ci(r, freq=12, n_boot=300, seed=2)
    point_sharpe = perf_stats(r, freq=12)["sharpe"]
    assert result["point_estimate"] == pytest.approx(point_sharpe, rel=1e-9)
