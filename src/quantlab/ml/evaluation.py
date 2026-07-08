"""Evaluation of ML alpha signals.

Two lenses, and both matter:

- **Regression quality** (:func:`regression_metrics`): how close are
  predictions to realized returns? Expect tiny R² — single-digit
  percent is a strong alpha model; markets are mostly noise.
- **Cross-sectional rank IC** (:func:`prediction_rank_ic`): does the
  model *order* assets correctly each date? This is what a long/short
  portfolio actually monetizes, and it is the same metric the research
  layer uses for hand-built factors, so ML and classical signals are
  directly comparable.

:func:`permutation_importance` measures how much each feature actually
contributes: shuffle one column, watch the score degrade. Unlike a
model's built-in importances it works for any wrapper and is measured on
data you choose (use held-out data to detect features the model merely
memorized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import get_settings
from quantlab.exceptions import ModelError
from quantlab.ml.models import ModelWrapper
from quantlab.research.evaluation import forward_returns, ic_series
from quantlab.types import WideFrame

__all__ = ["permutation_importance", "prediction_rank_ic", "regression_metrics"]


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """MSE, RMSE, MAE, R², Pearson correlation, and sample count."""
    aligned = pd.DataFrame({"t": y_true, "p": y_pred}).dropna()
    if len(aligned) < 2:
        raise ModelError(f"Need at least 2 aligned observations, got {len(aligned)}")
    err = aligned["t"] - aligned["p"]
    mse = float((err**2).mean())
    var = float(((aligned["t"] - aligned["t"].mean()) ** 2).mean())
    r2 = float("nan") if var == 0 else 1.0 - mse / var
    corr = float(aligned["t"].corr(aligned["p"]))
    return pd.Series(
        {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(err.abs().mean()),
            "r2": r2,
            "correlation": corr,
            "n": float(len(aligned)),
        },
        name="regression_metrics",
    )


def prediction_rank_ic(
    predictions: WideFrame,
    prices: WideFrame,
    horizon: int,
    method: str = "spearman",
) -> pd.Series:
    """Per-date IC between OOS predictions and realized forward returns.

    ``horizon`` must match the panel the predictions were trained on;
    the alignment (prediction at ``t`` vs return over ``(t, t+h]``) is
    the framework's standard one, so this is an honest out-of-sample
    IC when the predictions came from a walk-forward run.
    """
    fwd = forward_returns(prices, horizon)
    return ic_series(predictions, fwd, method=method)


def permutation_importance(
    model: ModelWrapper,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> pd.Series:
    """Mean MSE increase when each feature is shuffled (higher = matters).

    A feature the model ignores scores ~0; a feature it relies on scores
    the loss it would cost to destroy. Scores can be slightly negative
    for pure-noise features (shuffling noise can accidentally help).
    """
    if n_repeats < 1:
        raise ModelError(f"n_repeats must be >= 1, got {n_repeats}")
    aligned = X.join(y.rename("__y__"), how="inner").dropna()
    if len(aligned) < 3:
        raise ModelError(f"Need at least 3 complete rows, got {len(aligned)}")
    features = aligned[X.columns.tolist()]
    target = aligned["__y__"].to_numpy(dtype=float)

    baseline = float(np.mean((target - model.predict(features)) ** 2))
    seed = get_settings().random_seed if random_state is None else random_state
    rng = np.random.default_rng(seed)

    scores: dict[str, float] = {}
    for column in features.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            mse = float(np.mean((target - model.predict(shuffled)) ** 2))
            increases.append(mse - baseline)
        scores[str(column)] = float(np.mean(increases))
    return pd.Series(scores, name="permutation_importance")
