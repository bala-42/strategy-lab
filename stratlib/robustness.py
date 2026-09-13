"""
Robustness checks: parameter sensitivity and walk-forward validation.

A single backtest with hand-picked parameters proves almost nothing about
whether a strategy has a real edge -- it could just be the one parameter
combination that happened to work on this exact sample. These two tools
address that directly: sweep parameters to see if performance is a broad
plateau or a knife-edge, and validate out-of-sample on rolling windows
instead of a single train/test split.
"""
from __future__ import annotations

import itertools
import warnings
import numpy as np
import pandas as pd


def parameter_sensitivity(price: pd.Series, strategy_fn, backtest_fn, param_grid: dict,
                            metric: str = "sharpe", strict: bool = False) -> pd.DataFrame:
    """
    Sweep every combination of parameters in `param_grid`, run
    `strategy_fn(price, **params)` to get a signal, backtest it with
    `backtest_fn(price, signal)`, and collect `metric` from the result.

    A combination that raises is recorded as NaN *with the reason kept* in
    an `error` column, and warned about. It is not discarded: a sweep that
    quietly turns every failure into NaN renders as a blank heatmap that
    looks exactly like a strategy with no edge, and there is no way from
    the output to tell "this parameterization lost money" apart from "this
    parameterization crashed". If every combination fails the sweep raises,
    because an all-NaN grid is a bug in the call, not a result.

    Parameters
    ----------
    price       : price series
    strategy_fn : callable(price, **params) -> signal series
    backtest_fn : callable(price, signal) -> dict of stats (e.g. a
                  partial of stratlib.backtest.single_asset_backtest with
                  cost_bps/freq already bound)
    param_grid  : dict of param_name -> list of values to try
    metric      : which key to extract from the backtest result
    strict      : re-raise the first failure instead of recording it

    Returns
    -------
    Long-format DataFrame: one row per parameter combination, columns for
    each parameter, the metric value, and `error` (None where the
    combination ran). Suitable for a heatmap when param_grid has exactly
    two parameters.
    """
    names = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows = []
    for combo in combos:
        params = dict(zip(names, combo))
        error = None
        try:
            signal = strategy_fn(price, **params)
            result = backtest_fn(price, signal)
            value = result.get(metric, np.nan)
            if metric not in result:
                error = f"backtest result has no {metric!r} key"
            elif not np.isfinite(value):
                # Ran without raising and produced nothing usable -- e.g. a
                # lookback longer than the series, which yields an all-NaN
                # signal and a NaN Sharpe. On a heatmap this is
                # indistinguishable from a real result, so it is named.
                error = f"{metric} was not finite ({value!r})"
        except Exception as exc:
            if strict:
                raise
            value = np.nan
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            if strict:
                raise ValueError(f"parameter_sensitivity: {params} -- {error}")
            warnings.warn(
                f"parameter_sensitivity: {params} produced no usable {metric} -- {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        rows.append({**params, metric: value, "error": error})

    frame = pd.DataFrame(rows)
    if len(frame) and frame[metric].isna().all():
        raise ValueError(
            f"every one of the {len(frame)} parameter combinations produced no "
            f"usable {metric}; the first reason was {frame['error'].iloc[0]}. An "
            "all-NaN sweep renders as a blank heatmap that looks exactly like a "
            "strategy with no edge, so this raises rather than returning one"
        )
    return frame


def _bound_freq(backtest_fn) -> int | None:
    """Recover the annualisation the caller already bound to `backtest_fn`.

    Every call site in this repo passes a
    `partial(single_asset_backtest, cost_bps=..., freq=...)`, so the correct
    frequency is already sitting on the callable. Reading it is not a
    convenience: the fold statistics must annualize at the same rate as the
    backtest those folds came from, and any other value is wrong by
    construction.
    """
    keywords = getattr(backtest_fn, "keywords", None)
    if isinstance(keywords, dict):
        freq = keywords.get("freq")
        if isinstance(freq, int) and not isinstance(freq, bool) and freq > 0:
            return freq
    return None


def walk_forward_validation(price: pd.Series, strategy_fn, backtest_fn, params: dict,
                              train_window: int, test_window: int, step: int | None = None,
                              freq: int | None = None) -> pd.DataFrame:
    """
    Rolling walk-forward validation: for each fold, take `test_window`
    periods immediately following a `train_window`-period lookback, apply
    the strategy (with fixed `params` -- this checks temporal stability of
    a given parameterization, not re-fitting), and record out-of-sample
    performance for that fold only. Rolls forward by `step` (defaults to
    `test_window`, i.e. non-overlapping folds).

    This is a materially different check than a single 50/50 train/test
    split: it asks "does this strategy behave consistently across many
    different out-of-sample periods," not just "did it work on the one
    period I held out."

    `freq` is the annualisation used for each fold's statistics. Left
    unset it is recovered from `backtest_fn` when that is a partial with
    `freq` bound -- which every call site in this repo is -- so the folds
    annualize at the same rate as the backtest they came from. If it can be
    neither given nor recovered this raises rather than assuming 252.

    That default used to be hardcoded, and it was wrong wherever the data
    was not daily-trading: three of the four callers here pass BTC at
    freq=365, and every fold Sharpe they reported was understated by about
    17% (sqrt(252/365)). The *sign* of a Sharpe does not depend on freq, so
    win-rate figures computed as `(sharpe > 0).mean()` were unaffected --
    but the per-fold Sharpe and ann_return tables were.

    Returns
    -------
    DataFrame with one row per fold: fold_start, fold_end, sharpe,
    ann_return, max_drawdown, n_obs.
    """
    if step is None:
        step = test_window

    if freq is None:
        freq = _bound_freq(backtest_fn)
    if freq is None:
        raise ValueError(
            "walk_forward_validation needs a `freq` for annualizing each fold. "
            "Pass one explicitly, or pass a backtest_fn with freq bound (e.g. "
            "partial(single_asset_backtest, cost_bps=5.0, freq=365)). It is not "
            "defaulted, because silently annualizing daily crypto at 252 "
            "understates every fold's Sharpe by about 17%"
        )

    from .stats import perf_stats

    signal_full = strategy_fn(price, **params)
    rows = []
    start = train_window
    while start + test_window <= len(price):
        fold_idx = price.index[start:start + test_window]
        fold_signal = signal_full.reindex(price.index).loc[:fold_idx[-1]]
        # Only the test-window slice of returns counts toward this fold's
        # stats, but the position signal itself is allowed to use its
        # normal lookback (which may reach back into the training window --
        # that's fine, it's not re-fit on test data, just continuing to
        # apply the same fixed rule).
        fold_price = price.loc[:fold_idx[-1]]
        bt = backtest_fn(fold_price, fold_signal)
        fold_ret = bt["returns"].reindex(fold_idx).dropna()
        if len(fold_ret) < 5:
            start += step
            continue
        try:
            stats = perf_stats(fold_ret, freq=freq)
        except ValueError:
            start += step
            continue
        rows.append({
            "fold_start": fold_idx[0], "fold_end": fold_idx[-1],
            "sharpe": stats["sharpe"], "ann_return": stats["ann_return"],
            "max_drawdown": stats["max_drawdown"], "n_obs": len(fold_ret),
        })
        start += step
    return pd.DataFrame(rows)
