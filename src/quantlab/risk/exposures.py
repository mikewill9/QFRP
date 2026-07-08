"""Exposure tracking.

Exposures answer "where is the risk pointed?" independently of how large
it is: how much leverage the book runs (gross), which direction it leans
(net), which sectors it concentrates in, and which systematic factors its
returns load on.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["factor_exposures", "gross_net_exposure", "sector_exposures"]


def gross_net_exposure(weights: WideFrame) -> pd.DataFrame:
    """Per-bar gross (``sum |w|``) and net (``sum w``) exposure."""
    if weights.empty:
        raise DataError("Weight frame is empty")
    filled = weights.fillna(0.0)
    return pd.DataFrame(
        {"gross": filled.abs().sum(axis=1), "net": filled.sum(axis=1)}
    )


def sector_exposures(
    weights: WideFrame, sectors: Mapping[str, str] | pd.Series
) -> WideFrame:
    """Net weight per sector per bar (columns = sectors, sorted).

    Every symbol carrying a nonzero weight must appear in ``sectors``;
    unmapped symbols raise rather than being silently dropped, because a
    missing mapping usually means a data problem upstream.
    """
    if weights.empty:
        raise DataError("Weight frame is empty")
    mapping = dict(sectors.items()) if isinstance(sectors, pd.Series) else dict(sectors)
    filled = weights.fillna(0.0)
    held = [c for c in filled.columns if (filled[c] != 0).any()]
    unmapped = [c for c in held if c not in mapping]
    if unmapped:
        raise DataError(f"Symbols without a sector mapping: {unmapped[:5]}")
    groups = pd.Series({c: mapping[c] for c in held})
    out = filled[held].T.groupby(groups).sum().T
    return out.reindex(sorted(out.columns), axis=1)


def factor_exposures(returns: pd.Series, factor_returns: WideFrame) -> pd.Series:
    """OLS loadings of ``returns`` on factor return series.

    Returns a Series with an ``alpha`` entry (per-period intercept)
    followed by one beta per factor column. Observations are aligned on
    common dates; rows with any NaN are dropped.
    """
    frame = pd.concat(
        [returns.rename("__r__"), factor_returns], axis=1, sort=False
    ).dropna()
    n_factors = factor_returns.shape[1]
    if n_factors == 0:
        raise DataError("factor_returns has no columns")
    if len(frame) < n_factors + 2:
        raise DataError(
            f"Need at least {n_factors + 2} overlapping observations "
            f"for {n_factors} factors, got {len(frame)}"
        )
    y = frame["__r__"].to_numpy()
    x = np.column_stack(
        [np.ones(len(frame)), frame.drop(columns="__r__").to_numpy()]
    )
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise DataError(
            "Factor returns are collinear; exposures are not identified"
        )
    index = ["alpha", *[str(c) for c in factor_returns.columns]]
    return pd.Series(coef, index=index, name="exposure")
