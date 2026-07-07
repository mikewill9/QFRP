"""Performance statistics.

All functions take a *simple* per-period return series (the portfolio
accounting convention) with a DatetimeIndex, infer annualization from the
index spacing, and are NaN-tolerant. ``performance_summary`` bundles the
standard table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = [
    "annualized_vol",
    "cagr",
    "calmar_ratio",
    "hit_rate",
    "kurtosis",
    "max_drawdown",
    "performance_summary",
    "periods_per_year",
    "rolling_sharpe",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "total_return",
    "turnover",
]


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualization factor from median index spacing."""
    if len(index) < 2:
        raise DataError("Need at least 2 observations to infer frequency")
    median_days = float(index.to_series().diff().dt.days.median())
    if median_days <= 1.6:
        return 252.0
    if median_days <= 10.0:
        return 52.0
    if median_days <= 45.0:
        return 12.0
    return 365.25 / median_days


def _clean(returns: pd.Series) -> pd.Series:
    out = returns.dropna()
    if out.empty:
        raise DataError("Return series is empty after dropping NaN")
    return out


def total_return(returns: pd.Series) -> float:
    """Cumulative simple return over the full sample."""
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    ppy = periods_per_year(r.index)
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (ppy / len(r)) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.std() * np.sqrt(periods_per_year(r.index)))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio; ``rf`` is an annual risk-free rate."""
    r = _clean(returns)
    ppy = periods_per_year(r.index)
    excess = r - rf / ppy
    sd = excess.std()
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Sharpe variant penalizing only downside deviation."""
    r = _clean(returns)
    ppy = periods_per_year(r.index)
    excess = r - rf / ppy
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dd = np.sqrt(float((downside**2).mean()))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(ppy))


def max_drawdown(returns: pd.Series) -> float:
    """Deepest peak-to-trough loss of the compounded curve (negative)."""
    r = _clean(returns)
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR over absolute max drawdown."""
    dd = max_drawdown(returns)
    if dd == 0:
        return float("inf")
    return cagr(returns) / abs(dd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = _clean(returns)
    return float((r > 0).mean())


def skewness(returns: pd.Series) -> float:
    return float(_clean(returns).skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal = 0)."""
    return float(_clean(returns).kurtosis())


def turnover(weights: WideFrame) -> pd.Series:
    """One-way turnover per bar: ``0.5 * sum |w_t - w_{t-1}|``.

    A full liquidation-and-rebuild of a fully invested book is 1.0.
    """
    return (weights.fillna(0.0).diff().abs().sum(axis=1) * 0.5).rename("turnover")


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """Annualized Sharpe over a trailing window."""
    r = returns.dropna()
    ppy = periods_per_year(r.index)
    mean = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return (mean / sd * np.sqrt(ppy)).rename(f"sharpe_{window}")


def performance_summary(
    returns: pd.Series,
    weights: WideFrame | None = None,
    rf: float = 0.0,
) -> pd.Series:
    """Standard headline table as a Series.

    Includes annualized turnover when a weight frame is supplied.
    """
    r = _clean(returns)
    ppy = periods_per_year(r.index)
    summary = {
        "total_return": total_return(r),
        "cagr": cagr(r),
        "annualized_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r, rf),
        "sortino": sortino_ratio(r, rf),
        "calmar": calmar_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "skew": skewness(r),
        "excess_kurtosis": kurtosis(r),
        "n_periods": float(len(r)),
    }
    if weights is not None:
        summary["annualized_turnover"] = float(turnover(weights).mean() * ppy)
    return pd.Series(summary, name="performance")
