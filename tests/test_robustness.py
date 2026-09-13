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


def test_parameter_sensitivity_keeps_the_reason_a_combo_failed():
    """A NaN with no explanation is indistinguishable from a real result.

    This used to assert only that nothing crashed. The failing combination
    came back as a bare NaN, which on a heatmap looks exactly like a
    parameterization that lost money -- there was no way from the output to
    tell "this lost money" apart from "this never ran".
    """
    price = _trending_series(n=50)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    grid = {"lookback": [10, 1000]}  # 1000 > series length
    with pytest.warns(RuntimeWarning, match="no usable sharpe"):
        result = parameter_sensitivity(price, trend_following_signal, bt, grid, metric="sharpe")

    assert len(result) == 2
    ok = result[result["lookback"] == 10].iloc[0]
    bad = result[result["lookback"] == 1000].iloc[0]
    assert pd.isna(ok["error"]), "a combination that ran should carry no error"
    assert np.isnan(bad["sharpe"])
    assert "not finite" in bad["error"]


def test_parameter_sensitivity_raises_when_every_combo_fails():
    """An all-NaN grid renders as a blank heatmap that reads as "no edge"."""
    price = _trending_series(n=50)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    with pytest.raises(ValueError, match="blank heatmap"):
        with pytest.warns(RuntimeWarning):
            parameter_sensitivity(
                price, trend_following_signal, bt, {"lookback": [1000, 2000]}
            )


def test_parameter_sensitivity_strict_surfaces_the_first_failure():
    price = _trending_series(n=50)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    with pytest.raises(ValueError):
        parameter_sensitivity(
            price, trend_following_signal, bt, {"lookback": [1000]}, strict=True
        )


def test_walk_forward_validation_annualizes_at_the_frequency_the_backtest_used():
    """Regression test for a hardcoded freq=252.

    Three of the four callers in this repo pass BTC at freq=365. The fold
    statistics were annualized at 252 regardless, understating every fold's
    Sharpe by about sqrt(252/365) -- roughly 17%. This is the same class of
    bug the README already credits the suite with catching in
    `regime_conditional_returns`; it was still live one module over.
    """
    price = _trending_series(n=500)
    kwargs = dict(params={"lookback": 60}, train_window=100, test_window=50)

    daily = walk_forward_validation(
        price, trend_following_signal,
        partial(single_asset_backtest, cost_bps=5.0, freq=252), **kwargs)
    crypto = walk_forward_validation(
        price, trend_following_signal,
        partial(single_asset_backtest, cost_bps=5.0, freq=365), **kwargs)

    assert len(daily) == len(crypto)
    assert not np.allclose(daily["sharpe"], crypto["sharpe"])
    # A Sharpe scales as sqrt(freq) for small periodic returns.
    ratio = (crypto["sharpe"] / daily["sharpe"]).mean()
    assert 1.1 < ratio < 1.3

    # The sign is invariant to freq, which is why win-rate figures computed
    # as (sharpe > 0).mean() survived the bug unchanged.
    assert ((daily["sharpe"] > 0) == (crypto["sharpe"] > 0)).all()


def test_walk_forward_validation_refuses_to_guess_a_frequency():
    """Better to stop than to quietly annualize crypto at 252."""
    price = _trending_series(n=500)

    def unbound_bt(p, s):
        return single_asset_backtest(p, s, cost_bps=5.0, freq=365)

    with pytest.raises(ValueError, match="needs a `freq`"):
        walk_forward_validation(
            price, trend_following_signal, unbound_bt,
            params={"lookback": 60}, train_window=100, test_window=50,
        )


def test_walk_forward_validation_explicit_freq_wins_over_the_bound_one():
    price = _trending_series(n=500)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    kwargs = dict(params={"lookback": 60}, train_window=100, test_window=50)
    bound = walk_forward_validation(price, trend_following_signal, bt, **kwargs)
    explicit = walk_forward_validation(
        price, trend_following_signal, bt, freq=365, **kwargs)
    assert not np.allclose(bound["sharpe"], explicit["sharpe"])


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


def test_walk_forward_reports_the_fold_return_not_only_its_extrapolation():
    """Annualizing a short fold with a large move is not a measurement.

    A 180-day fold over early-2011 BTC returned ~44x; extrapolated to a year
    that is 212,796%, which describes a year that did not happen. The fold's
    own cumulative return is a fact, so it is reported alongside.
    """
    price = _trending_series(n=500)
    bt = partial(single_asset_backtest, cost_bps=5.0, freq=252)
    wf = walk_forward_validation(
        price, trend_following_signal, bt,
        params={"lookback": 60}, train_window=100, test_window=50,
    )
    assert "fold_return" in wf.columns
    # Over a 50-day fold at freq=252, annualizing magnifies: the extrapolated
    # figure should be the larger of the two in absolute terms for a winner.
    winners = wf[wf["fold_return"] > 0]
    assert len(winners), "the fixture produced no profitable fold"
    assert (winners["ann_return"].abs() >= winners["fold_return"].abs()).all()
    assert np.isfinite(wf["fold_return"]).all()
