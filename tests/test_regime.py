import numpy as np
import pandas as pd
import pytest

from stratlib.regime import trend_strength, volatility_level, classify_regime, regime_conditional_returns


def _strong_trend(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rets = 0.003 + rng.normal(0, 0.001, n)  # strong drift, tiny noise
    return pd.Series(100 * np.cumprod(1 + rets), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _pure_noise(n=200, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    price = 100 + (price - pd.Series(price).expanding().mean().values)  # keep it range-bound around 100
    return pd.Series(np.abs(price), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_trend_strength_high_for_clean_trend():
    price = _strong_trend()
    ts = trend_strength(price, window=60)
    assert ts.iloc[-1] > 0.9


def test_trend_strength_bounded_zero_to_one():
    price = _strong_trend()
    ts = trend_strength(price, window=60).dropna()
    assert (ts >= 0).all() and (ts <= 1).all()


def test_trend_strength_low_for_flat_price():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    price = pd.Series(100.0, index=idx)  # perfectly flat -> zero variance -> r2 defined as 0
    ts = trend_strength(price, window=60)
    assert ts.iloc[-1] == 0.0


def test_volatility_level_higher_for_noisier_series():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.001, 100)), index=idx)
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.05, 100)), index=idx)
    assert volatility_level(wild).iloc[-1] > volatility_level(calm).iloc[-1]


def test_classify_regime_produces_all_expected_columns_and_labels():
    price = _strong_trend(n=400)
    result = classify_regime(price, trend_window=60, vol_window=20)
    for col in ["trend_strength", "is_trending", "vol", "vol_percentile", "is_high_vol", "regime"]:
        assert col in result.columns
    valid_labels = {"Trending-HighVol", "Trending-LowVol", "Ranging-HighVol", "Ranging-LowVol"}
    observed = set(result["regime"].dropna().unique())
    assert observed.issubset(valid_labels)


def test_regime_conditional_returns_splits_correctly():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    returns = pd.Series(np.concatenate([np.full(50, 0.01), np.full(50, -0.01)]), index=idx)
    regime = pd.Series(["A"] * 50 + ["B"] * 50, index=idx)
    result = regime_conditional_returns(returns, regime)
    assert "A" in result.index and "B" in result.index
    assert result.loc["A", "ann_return"] > 0
    assert result.loc["B", "ann_return"] < 0


def test_regime_conditional_returns_respects_freq_argument():
    # Same constant return, but interpreted as monthly (freq=12) vs. daily
    # (freq=252) must annualize to different, correctly-computed figures --
    # this is a regression test for a real bug where freq was hardcoded to
    # 252 regardless of the actual data periodicity, producing wildly
    # wrong annualized numbers when fed non-daily returns (e.g. monthly
    # cross-sectional momentum returns).
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    returns = pd.Series([0.01] * 10, index=idx)
    regime = pd.Series(["X"] * 10, index=idx)

    monthly = regime_conditional_returns(returns, regime, freq=12)
    daily = regime_conditional_returns(returns, regime, freq=252)

    expected_monthly = 1.01 ** 12 - 1
    expected_daily = 1.01 ** 252 - 1
    assert monthly.loc["X", "ann_return"] == pytest.approx(expected_monthly, rel=1e-6)
    assert daily.loc["X", "ann_return"] == pytest.approx(expected_daily, rel=1e-6)
    # the daily interpretation should be dramatically larger -- confirming the
    # two calls actually used different exponents, not the same hardcoded one
    assert daily.loc["X", "ann_return"] > monthly.loc["X", "ann_return"] * 50
