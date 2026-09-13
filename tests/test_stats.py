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


def test_perf_stats_records_which_sharpe_convention_it_used():
    r = pd.Series(np.random.default_rng(3).normal(0.01, 0.04, 120))
    assert perf_stats(r, freq=12)["sharpe_convention"] == "arithmetic"
    assert perf_stats(r, freq=12, sharpe_convention="geometric")["sharpe_convention"] == "geometric"


def test_perf_stats_rejects_an_unknown_convention():
    r = pd.Series([0.01] * 20)
    with pytest.raises(ValueError, match="sharpe_convention"):
        perf_stats(r, freq=12, sharpe_convention="harmonic")


def test_geometric_sharpe_explodes_on_a_short_window_with_a_huge_move():
    """Why arithmetic is the default.

    The geometric convention raises the window's cumulative return to the
    power freq/len(r). A 180-day fold that returned ~44x -- which early BTC
    genuinely did -- annualizes to a "Sharpe" in the thousands. The number is
    arithmetically correct and completely unusable, and it sat in a committed
    walk-forward table reading 116.9 before the annualization frequency was
    also fixed.
    """
    rng = np.random.default_rng(7)
    # ~2.1% a day for 180 days compounds to roughly 44x.
    r = pd.Series(0.021 + rng.normal(0, 0.05, 180))

    geometric = perf_stats(r, freq=365, sharpe_convention="geometric")
    arithmetic = perf_stats(r, freq=365, sharpe_convention="arithmetic")

    # The claim is the relationship, not a magic threshold: the geometric
    # convention lands an order of magnitude away, in a range no reader can
    # use, while the arithmetic one stays interpretable.
    assert geometric["sharpe"] > 10 * arithmetic["sharpe"]
    assert geometric["sharpe"] > 50, "the fixture no longer reproduces the blow-up"
    assert 0 < arithmetic["sharpe"] < 20
    # Same underlying data, same annualized return -- only the ratio differs.
    assert geometric["ann_return"] == arithmetic["ann_return"]


def test_the_two_conventions_agree_closely_on_an_ordinary_sample():
    """Arithmetic is not simply a different number everywhere.

    On a long sample with modest returns the two are close, which is why the
    geometric one survived unnoticed for so long.
    """
    r = pd.Series(np.random.default_rng(11).normal(0.0004, 0.01, 2000))
    geometric = perf_stats(r, freq=252, sharpe_convention="geometric")["sharpe"]
    arithmetic = perf_stats(r, freq=252, sharpe_convention="arithmetic")["sharpe"]
    assert abs(geometric - arithmetic) / abs(arithmetic) < 0.10


def test_sharpe_ci_uses_the_same_convention_as_its_point_estimate():
    r = pd.Series(np.random.default_rng(5).normal(0.01, 0.05, 200))
    for convention in ("arithmetic", "geometric"):
        point = perf_stats(r, freq=252, sharpe_convention=convention)["sharpe"]
        ci = sharpe_ci(r, freq=252, sharpe_convention=convention, n_boot=200, seed=1)
        assert ci["ci_low"] <= point <= ci["ci_high"]
