"""Return attribution.

Two complementary decompositions:

- :func:`asset_contributions` splits a portfolio's per-bar return across
  its holdings: how much did each symbol add or subtract.
- :func:`factor_attribution` splits a return *series* across systematic
  factors via OLS: how much is explained by known risk factors (alpha
  plus factor betas) versus idiosyncratic residual.

Both use the standard linear-attribution approximation (no compounding
cross-terms, no transaction costs): contributions are computed
per-period and summed, which is exact for a single period and a close
approximation over a full sample when returns are small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["FactorAttribution", "asset_contributions", "factor_attribution"]


def asset_contributions(weights: WideFrame, asset_returns: WideFrame) -> WideFrame:
    """Per-bar, per-asset contribution to portfolio return.

    Uses the *prior* bar's weight — the position actually held during
    the bar — times that bar's asset return:
    ``contribution_t = weight_{t-1} * return_t``. Summed across assets,
    a bar's contributions equal that bar's portfolio return under a
    constant-weight assumption; with rebalancing this is the standard
    linear-attribution approximation, not an exact decomposition of
    compounded, cost-inclusive returns.
    """
    idx = weights.index.intersection(asset_returns.index)
    cols = weights.columns.intersection(asset_returns.columns)
    if len(idx) < 2 or len(cols) == 0:
        raise DataError("weights and asset_returns share too little overlap")
    idx = idx.sort_values()
    prior_weight = weights.loc[idx, cols].shift(1)
    realized_return = asset_returns.loc[idx, cols]
    return (prior_weight * realized_return).rename_axis(columns="symbol")


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    """OLS decomposition of a return series into alpha and factor betas.

    Attributes
    ----------
    alpha:
        Per-period intercept (return unexplained by the factors).
    betas:
        Loadings on each factor, indexed by factor name.
    r_squared:
        Fraction of return variance explained by the regression.
    contributions:
        Per-period frame with columns ``alpha``, one per factor, and
        ``residual``; every row sums exactly to the input return.
    """

    alpha: float
    betas: pd.Series
    r_squared: float
    contributions: pd.DataFrame


def factor_attribution(returns: pd.Series, factor_returns: WideFrame) -> FactorAttribution:
    """Regress ``returns`` on ``factor_returns`` and attribute the result.

    Observations are aligned on common dates; rows with any NaN are
    dropped. Collinear factors make betas unidentified and raise
    :class:`~quantlab.exceptions.DataError` rather than returning an
    arbitrary least-squares solution.
    """
    n_factors = factor_returns.shape[1]
    if n_factors == 0:
        raise DataError("factor_returns has no columns")
    frame = pd.concat(
        [returns.rename("__r__"), factor_returns], axis=1, sort=False
    ).dropna()
    if len(frame) < n_factors + 2:
        raise DataError(
            f"Need at least {n_factors + 2} overlapping observations "
            f"for {n_factors} factors, got {len(frame)}"
        )
    factor_names = [str(c) for c in factor_returns.columns]
    y = frame["__r__"].to_numpy()
    factor_values = frame[factor_names].to_numpy()
    x = np.column_stack([np.ones(len(frame)), factor_values])
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise DataError("Factor returns are collinear; attribution is not identified")

    alpha = float(coef[0])
    betas = pd.Series(coef[1:], index=factor_names, name="beta")
    fitted = x @ coef
    residual = y - fitted
    ss_res = float((residual**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    contributions = pd.DataFrame(
        {name: betas[name] * frame[name] for name in factor_names}, index=frame.index
    )
    contributions.insert(0, "alpha", alpha)
    contributions["residual"] = residual

    return FactorAttribution(
        alpha=alpha, betas=betas, r_squared=r_squared, contributions=contributions
    )
