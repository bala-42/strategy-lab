import numpy as np
import pandas as pd
import pytest

from stratlib.strategies import (
    trend_following_signal, mean_reversion_signal,
    volatility_breakout_signal, seasonality_signal,
)


def _trending_series(n=300, drift=0.002, vol=0.005, seed=0):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, vol, n)
    price = 100 * np.cumprod(1 + rets)
    return pd.Series(price, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _ranging_series(n=300, level=100, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    price = level + np.cumsum(rng.normal(0, vol * level, n))
    price = level + (price - price.mean())  # keep it mean-reverting around `level`
    return pd.Series(np.abs(price), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_trend_following_signal_is_positive_in_strong_uptrend():
    price = _trending_series(drift=0.003)
    sig = trend_following_signal(price, lookback=60)
    # after the initial warm-up, the signal should be long almost always
    # in a strong, low-noise uptrend
    assert (sig.iloc[100:] == 1).mean() > 0.8


def test_trend_following_signal_is_negative_in_strong_downtrend():
    price = _trending_series(drift=-0.003)
    sig = trend_following_signal(price, lookback=60)
    assert (sig.iloc[100:] == -1).mean() > 0.8


def test_trend_following_signal_bounded():
    price = _trending_series()
    sig = trend_following_signal(price, lookback=30)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_mean_reversion_signal_shorts_after_spike_above_mean():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    price = pd.Series([100.0] * 40 + [130.0] * 20, index=idx)  # flat, then a sustained spike
    sig = mean_reversion_signal(price, window=20, entry_z=1.0, exit_z=0.25)
    # once the spike persists, the z-score should climb above entry_z and
    # the strategy should go short
    assert (sig.iloc[45:] == -1).any()


def test_mean_reversion_signal_bounded():
    price = _ranging_series()
    sig = mean_reversion_signal(price)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_volatility_breakout_signal_bounded_and_aligned():
    price = _trending_series(n=400)
    sig = volatility_breakout_signal(price)
    assert sig.index.equals(price.index)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_seasonality_turn_of_month_only_fires_near_boundaries():
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="B")
    price = pd.Series(100.0, index=idx)
    sig = seasonality_signal(price, effect="turn_of_month")
    on_days = sig[sig > 0].index
    # every "on" day should be within the first 3 or the last 1 day of its month
    for d in on_days:
        assert d.day <= 3 or d.day >= d.days_in_month


def test_seasonality_day_of_week_only_monday_tuesday():
    idx = pd.date_range("2021-01-04", periods=20, freq="B")  # starts on a Monday
    price = pd.Series(100.0, index=idx)
    sig = seasonality_signal(price, effect="day_of_week")
    on_days = sig[sig > 0].index
    assert set(on_days.dayofweek.unique()).issubset({0, 1})


def test_seasonality_unknown_effect_raises():
    price = _trending_series(n=10)
    with pytest.raises(ValueError):
        seasonality_signal(price, effect="not_a_real_effect")
