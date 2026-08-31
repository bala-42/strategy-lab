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
import numpy as np
import pandas as pd


def parameter_sensitivity(price: pd.Series, strategy_fn, backtest_fn, param_grid: dict,
                            metric: str = "sharpe") -> pd.DataFrame:
    """
    Sweep every combination of parameters in `param_grid`, run
    `strategy_fn(price, **params)` to get a signal, backtest it with
    `backtest_fn(price, signal)`, and collect `metric` from the result.

    Parameters
    ----------
    price       : price series
    strategy_fn : callable(price, **params) -> signal series
    backtest_fn : callable(price, signal) -> dict of stats (e.g. a
                  partial of stratlib.backtest.single_asset_backtest with
                  cost_bps/freq already bound)
    param_grid  : dict of param_name -> list of values to try
    metric      : which key to extract from the backtest result

    Returns
    -------
    Long-format DataFrame: one row per parameter combination, columns for
    each parameter plus the metric value. Suitable for a heatmap when
    param_grid has exactly two parameters.
    """
    names = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows = []
    for combo in combos:
        params = dict(zip(names, combo))
        try:
            signal = strategy_fn(price, **params)
            result = backtest_fn(price, signal)
            value = result.get(metric, np.nan)
        except Exception:
            value = np.nan
        rows.append({**params, metric: value})
    return pd.DataFrame(rows)


def walk_forward_validation(price: pd.Series, strategy_fn, backtest_fn, params: dict,
                              train_window: int, test_window: int, step: int | None = None) -> pd.DataFrame:
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

    Returns
    -------
    DataFrame with one row per fold: fold_start, fold_end, sharpe,
    ann_return, max_drawdown, n_obs.
    """
    if step is None:
        step = test_window

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
            stats = perf_stats(fold_ret, freq=252)
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
