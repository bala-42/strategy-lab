"""
Backtest engines.

single_asset_backtest : the one engine that runs trend-following,
                         mean-reversion, volatility-breakout, and
                         seasonality -- every strategy that follows the
                         "price in, position out" contract in
                         `stratlib.strategies`.
decile_backtest        : cross-sectional signal backtest (for
                         cross-sectional momentum).
cointegration_test / pairs_backtest : for the pairs-trading strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

from .stats import perf_stats


def single_asset_backtest(price: pd.Series, signal: pd.Series, cost_bps: float = 5.0,
                            freq: int = 252) -> dict:
    """
    Apply a position signal to a price series and compute net-of-cost
    performance. The signal is shifted by one period internally (the
    position held during period t was decided using information available
    at the end of t-1), so strategy functions in `stratlib.strategies`
    never need to shift their own output.

    Parameters
    ----------
    price    : price series
    signal   : position series in [-1, 1], same index as price, NOT
               pre-shifted
    cost_bps : round-trip transaction cost in basis points, charged on
               every change in position
    freq     : periods per year for annualization (252 trading days,
               365 crypto calendar days, 12 monthly, etc.)

    Returns
    -------
    dict with "returns" (net strategy return series), "signal", plus the
    standard perf_stats keys and "n_position_changes".
    """
    ret = price.pct_change()
    aligned_signal = signal.reindex(price.index).fillna(0.0)
    strat_ret = aligned_signal.shift(1) * ret

    trades = aligned_signal.diff().abs().fillna(0)
    cost = trades * (cost_bps / 1e4)
    net_ret = (strat_ret - cost).dropna()

    stats = perf_stats(net_ret, freq=freq)
    stats["n_position_changes"] = int((trades > 0).sum())
    stats["pct_time_in_market"] = float((aligned_signal != 0).mean())
    return {"returns": net_ret, "signal": aligned_signal, **stats}


def decile_backtest(signal: pd.DataFrame, fwd_returns: pd.DataFrame, n_deciles: int = 10,
                     min_names: int = 20) -> pd.DataFrame:
    """
    Generic cross-sectional decile backtest -- ranks the cross-section at
    each date into `n_deciles` groups by `signal`, computes the
    equal-weighted forward return of each group from `fwd_returns`, plus
    the top-minus-bottom long-short spread and the equal-weight universe
    average.
    """
    records = []
    for date in signal.index:
        sig_t = signal.loc[date].dropna()
        if len(sig_t) < min_names:
            continue
        if date not in fwd_returns.index:
            continue
        fwd_t = fwd_returns.loc[date].dropna()

        ranks = pd.qcut(sig_t, n_deciles, labels=False, duplicates="drop")
        decile_rets = {}
        for d in range(n_deciles):
            names = ranks[ranks == d].index.intersection(fwd_t.index)
            if len(names) == 0:
                continue
            decile_rets[d] = fwd_t[names].mean()

        if (n_deciles - 1) in decile_rets and 0 in decile_rets:
            long_short = decile_rets[n_deciles - 1] - decile_rets[0]
        else:
            long_short = np.nan

        row = {"date": date, "long_short": long_short}
        row.update({f"decile_{d}": v for d, v in decile_rets.items()})
        row["universe_avg"] = fwd_t[ranks.index.intersection(fwd_t.index)].mean()
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["long_short", "universe_avg"]).rename_axis("date")
    out = pd.DataFrame(records).set_index("date").sort_index()
    return out.dropna(subset=["long_short"])


def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Engle-Granger cointegration test + OLS hedge ratio + ADF on the spread."""
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    aligned.columns = ["a", "b"]

    score, pvalue, _ = coint(aligned["a"], aligned["b"])

    X = sm.add_constant(aligned["b"])
    model = sm.OLS(aligned["a"], X).fit()
    alpha, beta = model.params["const"], model.params["b"]

    spread = aligned["a"] - beta * aligned["b"]
    adf_result = adfuller(spread, result_object=True)

    return {
        "coint_pvalue": float(pvalue),
        "adf_pvalue": float(adf_result.pvalue),
        "hedge_ratio": float(beta),
        "alpha": float(alpha),
        "r_squared": float(model.rsquared),
        "n_obs": len(aligned),
    }


def pairs_backtest(log_price_a: pd.Series, log_price_b: pd.Series, hedge_ratio: float,
                    window: int = 30, entry_z: float = 2.0, exit_z: float = 0.5,
                    stop_z: float = 3.5, cost_bps: float = 10.0, ann_factor: int = 365) -> dict:
    """Threshold mean-reversion backtest on a cointegrated spread."""
    aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
    aligned.columns = ["a", "b"]

    spread = aligned["a"] - hedge_ratio * aligned["b"]
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    position = np.zeros(len(zscore))
    pos = 0
    z_vals = zscore.values
    for i in range(1, len(z_vals)):
        z = z_vals[i]
        if np.isnan(z):
            position[i] = pos
            continue
        if pos == 0:
            if z > entry_z:
                pos = -1
            elif z < -entry_z:
                pos = 1
        elif pos == 1 and (z >= -exit_z or z < -stop_z):
            pos = 0
        elif pos == -1 and (z <= exit_z or z > stop_z):
            pos = 0
        position[i] = pos
    position = pd.Series(position, index=zscore.index)

    ret_a, ret_b = aligned["a"].diff(), aligned["b"].diff()
    spread_ret = ret_a - hedge_ratio * ret_b
    strat_ret = position.shift(1) * spread_ret

    trades = position.diff().abs().fillna(0)
    cost = trades * (cost_bps / 1e4) * 2
    net_ret = (strat_ret - cost.reindex(strat_ret.index).fillna(0)).dropna()

    stats = perf_stats(net_ret, freq=ann_factor)
    stats["n_position_changes"] = int((trades > 0).sum())
    stats["pct_time_in_market"] = float((position != 0).mean())
    return {"returns": net_ret, "zscore": zscore, "position": position, **stats}
