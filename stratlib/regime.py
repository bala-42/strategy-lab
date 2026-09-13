"""
Regime classification.

Every strategy's performance is later cross-tabbed against the regime
that was in effect at the time, using the two classifications below. Both
are computed purely from price -- no external "someone decided this was a
bull market" labeling, which would be circular (and impossible to apply
consistently across equities/crypto/FX anyway).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_strength(price: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling R^2 of a linear regression of log-price on time. High R^2
    means price has been moving cleanly in one direction (trending, up or
    down); low R^2 means price has been moving sideways/choppily
    (ranging). This is a standard, simple, interpretable trend-strength
    proxy -- deliberately not the strategy's own signal, so evaluating a
    strategy "conditional on trend regime" isn't circular.
    """
    log_price = np.log(price)

    def r_squared(y):
        if np.any(np.isnan(y)):
            return np.nan
        x = np.arange(len(y))
        x_mean, y_mean = x.mean(), y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((y - y_mean) ** 2)
        if ss_xx == 0 or ss_yy == 0:
            return 0.0
        slope = ss_xy / ss_xx
        r2 = (slope * ss_xy) / ss_yy if ss_yy > 0 else 0.0
        return max(0.0, min(1.0, r2))

    return log_price.rolling(window).apply(r_squared, raw=True)


def volatility_level(price: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility of returns (not annualized -- used only
    for relative ranking within a series, so the scale doesn't matter)."""
    return price.pct_change().rolling(window).std()


def classify_regime(price: pd.Series, trend_window: int = 60, vol_window: int = 20,
                     trend_threshold: float = 0.3, vol_lookback: int = 252) -> pd.DataFrame:
    """
    Classify each date into one of four regimes based on trend strength
    and volatility level, both measured relative to their own trailing
    history (so "high vol" means high relative to how volatile this
    specific asset normally is, not an absolute cutoff that would mean
    something different for BTC than for a utility stock).

    Both inputs need a warm-up before they mean anything: `trend_strength`
    needs `trend_window` observations and `vol_percentile` needs 60 of
    `vol`, which itself needs `vol_window`. Until both exist the regime is
    **NaN**, not a label.

    That distinction is load-bearing. `NaN >= threshold` is `False` in
    pandas, not NaN -- so an unguarded comparison silently reports "not
    trending" and "not high vol" throughout the warm-up, and every one of
    those dates comes out labelled "Ranging-LowVol". The label is not a
    measurement; it is the absence of one. Left in, it packs the opening
    stretch of every series into a single bucket, and
    `regime_conditional_returns` then reports that bucket's performance as
    if the regime had been observed.

    Returns
    -------
    DataFrame indexed like `price` with columns:
        trend_strength, is_trending, vol, vol_percentile, is_high_vol,
        regime_known, regime
    where regime is NaN during the warm-up and otherwise one of:
        "Trending-HighVol", "Trending-LowVol", "Ranging-HighVol", "Ranging-LowVol"

    `is_trending` and `is_high_vol` are nullable booleans, pd.NA wherever
    the underlying measurement does not exist yet, so `.isna()` separates
    "measured, and not trending" from "not measurable yet". Note that
    pandas still treats pd.NA as False when such a Series is used directly
    as a boolean mask -- it does not raise -- so filter on `regime_known`
    rather than relying on the mask to object.
    """
    ts = trend_strength(price, window=trend_window)

    vol = volatility_level(price, window=vol_window)
    vol_percentile = vol.rolling(vol_lookback, min_periods=60).apply(
        lambda x: (x[-1] <= x).mean(), raw=True
    )

    # Both halves must be measurable before the pair can be labelled.
    regime_known = ts.notna() & vol_percentile.notna()

    is_trending = (ts >= trend_threshold).astype("boolean").where(ts.notna())
    is_high_vol = (vol_percentile >= 0.5).astype("boolean").where(
        vol_percentile.notna()
    )

    trending = is_trending.fillna(False).astype(bool)
    high_vol = is_high_vol.fillna(False).astype(bool)

    regime = pd.Series(np.nan, index=price.index, dtype=object)
    regime[trending & high_vol] = "Trending-HighVol"
    regime[trending & ~high_vol] = "Trending-LowVol"
    regime[~trending & high_vol] = "Ranging-HighVol"
    regime[~trending & ~high_vol] = "Ranging-LowVol"
    regime[~regime_known] = np.nan

    return pd.DataFrame({
        "trend_strength": ts, "is_trending": is_trending,
        "vol": vol, "vol_percentile": vol_percentile, "is_high_vol": is_high_vol,
        "regime_known": regime_known,
        "regime": regime,
    })


def regime_conditional_returns(returns: pd.Series, regime: pd.Series, freq: int = 252) -> pd.DataFrame:
    """
    Split a strategy's return series by the regime active at each date and
    summarize performance within each regime bucket. `regime` should be
    the "regime" column from `classify_regime`, reindexed/aligned to
    `returns` -- typically the *previous* period's regime, since that's
    what was knowable when the position was taken.

    `freq` must match the actual periodicity of `returns` (252 for daily,
    12 for monthly, 365 for crypto calendar-day data, etc.) -- passing the
    wrong frequency silently produces nonsensical annualized figures for
    thin regime buckets, since perf_stats extrapolates a short bucket's
    average return to a full year using this exponent.
    """
    from .stats import perf_stats

    aligned = pd.concat([returns.rename("ret"), regime.rename("regime")], axis=1).dropna()
    rows = []
    for label, group in aligned.groupby("regime"):
        if len(group) < 5:
            continue
        try:
            stats = perf_stats(group["ret"], freq=freq)
        except ValueError:
            continue
        stats["regime"] = label
        stats["n_obs"] = len(group)
        rows.append(stats)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("regime")
