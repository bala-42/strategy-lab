"""
Signal generators for six strategies.

Every single-asset strategy here follows the same contract: it takes a
price series and returns a position series in [-1, 1] (or a subset of that
range), where the position at time t is the desired exposure to hold going
INTO period t+1 (i.e., it has not been shifted yet -- shifting is the
backtest engine's job, so a strategy's own logic is never accidentally
looking ahead). This uniform contract is what lets a single backtest
engine (`stratlib.backtest.single_asset_backtest`) run all four
single-asset strategies without four different sets of glue code.

Cross-sectional momentum and pairs trading don't fit the single-asset
contract (they need multiple assets simultaneously) and are handled by
their own generator + backtest engine pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Trend-following (time-series momentum)
# ---------------------------------------------------------------------------
def trend_following_signal(price: pd.Series, lookback: int = 90) -> pd.Series:
    """
    Absolute (time-series) momentum: long if the asset's trailing return
    over `lookback` periods is positive, short if negative. This is the
    "managed futures" style signal -- it says nothing about how the asset
    compares to others, only about its own recent direction.
    """
    trailing_return = price.pct_change(lookback)
    signal = np.sign(trailing_return)
    return signal.fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Short-term mean-reversion (contrarian)
# ---------------------------------------------------------------------------
def mean_reversion_signal(price: pd.Series, window: int = 20, entry_z: float = 1.0,
                            exit_z: float = 0.25) -> pd.Series:
    """
    Z-score contrarian signal: short when price is far above its rolling
    mean (bet on reversion down), long when far below (bet on reversion
    up), flat near the mean. Unlike trend-following, this explicitly bets
    against the recent direction.
    """
    roll_mean = price.rolling(window).mean()
    roll_std = price.rolling(window).std()
    z = (price - roll_mean) / roll_std

    position = np.zeros(len(z))
    pos = 0
    z_vals = z.values
    for i in range(len(z_vals)):
        zi = z_vals[i]
        if np.isnan(zi):
            position[i] = pos
            continue
        if pos == 0:
            if zi > entry_z:
                pos = -1
            elif zi < -entry_z:
                pos = 1
        elif pos == 1 and zi >= -exit_z:
            pos = 0
        elif pos == -1 and zi <= exit_z:
            pos = 0
        position[i] = pos
    return pd.Series(position, index=price.index)


# ---------------------------------------------------------------------------
# 3. Volatility breakout (regime-transition strategy)
# ---------------------------------------------------------------------------
def volatility_breakout_signal(price: pd.Series, squeeze_window: int = 20,
                                 breakout_window: int = 5, squeeze_pctile: float = 0.25) -> pd.Series:
    """
    Trades breakouts that follow a volatility "squeeze": when trailing
    realized volatility (over `squeeze_window`) is in its lowest quartile
    relative to its own trailing history, and price then makes a new
    `breakout_window`-period high or low, take a position in the breakout
    direction. Flat otherwise. This is deliberately a regime-*transition*
    strategy -- it should do best right as a ranging market starts to
    trend, unlike trend-following (which needs an established trend) or
    mean-reversion (which needs an established range).
    """
    ret = price.pct_change()
    realized_vol = ret.rolling(squeeze_window).std()
    vol_rank = realized_vol.rolling(252, min_periods=60).apply(
        lambda x: (x[-1] <= x).mean(), raw=True
    )
    is_squeeze = vol_rank <= squeeze_pctile

    rolling_high = price.rolling(breakout_window).max()
    rolling_low = price.rolling(breakout_window).min()
    breaks_up = price >= rolling_high
    breaks_down = price <= rolling_low

    signal = pd.Series(0.0, index=price.index)
    signal[is_squeeze & breaks_up] = 1.0
    signal[is_squeeze & breaks_down] = -1.0
    # Hold the position for `breakout_window` periods after triggering,
    # then flatten (a breakout signal is a short-lived event, not a
    # standing position like trend-following).
    signal = signal.replace(0.0, np.nan).ffill(limit=breakout_window).fillna(0.0)
    return signal


# ---------------------------------------------------------------------------
# 4. Calendar / seasonality
# ---------------------------------------------------------------------------
def seasonality_signal(price: pd.Series, effect: str = "turn_of_month") -> pd.Series:
    """
    Calendar-based signals, long-only, structurally different from every
    other strategy here: it uses no information about the asset's own
    price history at all, only the calendar date.

    effect="turn_of_month": long for the last trading day of the month
        through the first 3 trading days of the next month (the
        well-documented turn-of-month effect in equities).
    effect="day_of_week": long on Mondays and Tuesdays only (proxy for
        documented day-of-week seasonality patterns).
    """
    idx = price.index
    if effect == "turn_of_month":
        day = idx.day
        days_in_month = idx.days_in_month
        is_month_end = day >= (days_in_month - 0)
        is_early_month = day <= 3
        signal = pd.Series(0.0, index=idx)
        signal[is_month_end | is_early_month] = 1.0
    elif effect == "day_of_week":
        dow = idx.dayofweek  # Monday=0
        signal = pd.Series(0.0, index=idx)
        signal[dow.isin([0, 1])] = 1.0
    else:
        raise ValueError(f"Unknown seasonality effect: {effect}")
    return signal
