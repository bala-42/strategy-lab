"""
Performance statistics and bootstrap confidence intervals.

Every strategy in this repo needs the same handful of numbers (annualized
return, annualized vol, Sharpe, max drawdown, hit rate) computed from a
return
series. Previously each notebook had its own copy of this function; now
there's exactly one, and it's tested.

`perf_stats` reports which Sharpe convention it used. It used to apply a
geometric annualized return over an arithmetic annualized volatility without
saying so, which is defensible on a long sample and produces absurdities on a
short one -- a 180-day fold that returned 43.8x came out with a "Sharpe" of
1046. See the docstring for the comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative growth curve."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    return float(drawdown.min())


SHARPE_CONVENTIONS = ("arithmetic", "geometric")


def perf_stats(returns: pd.Series, freq: int, rf: float = 0.0, name: str | None = None,
                 sharpe_convention: str = "arithmetic") -> dict:
    """
    Compute standard performance statistics from a periodic return series.

    Two Sharpe conventions are available and the result records which one it
    used, because they are not interchangeable:

    ``arithmetic`` (default)
        ``(mean(r) - rf/freq) * freq / (std(r) * sqrt(freq))``. The textbook
        definition, and the one a reader comparing against a published Sharpe
        assumes.

    ``geometric``
        ``(compound annualized return - rf) / annualized vol``. A real
        statistic answering a different question -- compound growth per unit
        of volatility -- and the convention this function used to apply
        without saying so.

    The difference is not cosmetic on short windows with large moves, because
    the compound annualization raises the fold's cumulative return to the
    power ``freq / len(r)``. A 180-day walk-forward fold over early BTC that
    returned 43.8x annualizes to 212,796% and a "Sharpe" of 1046 under the
    geometric convention, against 4.75 under the arithmetic one. The first
    number is arithmetically correct and useless; extrapolating half a year
    of a 43x move out to a full year is not a measurement anyone can act on.

    The two also disagree on *sign* for some folds, so a win rate computed as
    ``(sharpe > 0).mean()`` depends on the convention -- which is exactly why
    it is now recorded rather than assumed.

    Parameters
    ----------
    returns           : pd.Series of periodic (not cumulative) returns
    freq              : periods per year used for annualization (12=monthly,
                        252=daily trading, 365=daily calendar/crypto)
    rf                : annualized risk-free rate, used in the Sharpe ratio
    name              : optional label, included in the result as "name"
    sharpe_convention : "arithmetic" (default) or "geometric"

    Returns
    -------
    dict with ann_return, ann_vol, sharpe, sharpe_convention, max_drawdown,
    hit_rate. `ann_return` remains the compound annualized return under both
    conventions -- it is what an investor actually experiences, and only the
    Sharpe definition changes.
    """
    if sharpe_convention not in SHARPE_CONVENTIONS:
        raise ValueError(
            f"sharpe_convention must be one of {SHARPE_CONVENTIONS}, "
            f"got {sharpe_convention!r}"
        )

    r = returns.dropna()
    if len(r) == 0:
        raise ValueError("perf_stats received an empty return series")

    ann_return = (1 + r).prod() ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)

    if ann_vol > 0:
        if sharpe_convention == "arithmetic":
            excess = r - rf / freq
            sharpe = excess.mean() * freq / (excess.std() * np.sqrt(freq))
        else:
            sharpe = (ann_return - rf) / ann_vol
    else:
        sharpe = np.nan

    result = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "sharpe_convention": sharpe_convention,
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
    }
    if name is not None:
        result = {"name": name, **result}
    return result


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([
        (np.arange(start, start + block_size) % n) for start in starts
    ])
    return idx[:n]


def bootstrap_ci(
    returns: pd.Series,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.90,
    block_size: int | None = None,
    seed: int | None = 42,
) -> dict:
    """
    Moving-block bootstrap confidence interval for an arbitrary statistic.

    A plain i.i.d. bootstrap understates uncertainty for financial returns
    because they're autocorrelated (volatility clustering, momentum in the
    signal itself). The moving-block bootstrap resamples contiguous chunks
    of the series instead of individual points, which preserves short-run
    dependence structure.

    Parameters
    ----------
    returns    : pd.Series of periodic returns
    stat_fn    : callable(pd.Series) -> float, e.g. lambda r: perf_stats(r, 12)["sharpe"]
    n_boot     : number of bootstrap resamples
    ci         : confidence level (0.90 -> 5th/95th percentile interval)
    block_size : block length; defaults to round(n**(1/3)), a standard
                 rule-of-thumb for block bootstrap on n observations
    seed       : RNG seed for reproducibility

    Returns
    -------
    dict with point_estimate, ci_low, ci_high, ci_level, n_boot
    """
    r = returns.dropna().values
    n = len(r)
    if n < 8:
        raise ValueError("bootstrap_ci needs at least 8 observations")
    if block_size is None:
        block_size = max(2, round(n ** (1 / 3)))

    rng = np.random.default_rng(seed)
    point_estimate = stat_fn(pd.Series(r))

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block_size, rng)
        boot_stats[i] = stat_fn(pd.Series(r[idx]))

    alpha = 1 - ci
    lo, hi = np.nanpercentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "block_size": block_size,
    }


def sharpe_ci(returns: pd.Series, freq: int, rf: float = 0.0,
                sharpe_convention: str = "arithmetic", **kwargs) -> dict:
    """Convenience wrapper: bootstrap CI specifically for the Sharpe ratio.

    Takes the same `sharpe_convention` as `perf_stats`, so an interval and the
    point estimate it surrounds cannot end up computed two different ways.
    """
    def _sharpe(r):
        return perf_stats(r, freq=freq, rf=rf,
                          sharpe_convention=sharpe_convention)["sharpe"]
    return bootstrap_ci(returns, _sharpe, **kwargs)
