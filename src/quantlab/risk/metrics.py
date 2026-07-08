"""Tail-risk and volatility metrics.

All functions take a *simple* per-period return series (the portfolio
accounting convention). VaR and Expected Shortfall are reported as
**positive loss fractions**: ``value_at_risk(r, 0.95) == 0.02`` means "on
the worst 5% of periods you lose at least 2%".

Three estimation methods, in increasing order of distributional
assumption:

- ``historical``: the empirical quantile of realized returns. No
  assumptions, but needs enough observations to populate the tail.
- ``normal``: assumes returns are Gaussian. Smooth and stable, but
  understates tails when returns are skewed or fat-tailed (they usually
  are).
- ``cornish_fisher``: adjusts the Gaussian quantile using the sample's
  skewness and excess kurtosis — a middle ground that keeps the
  parametric smoothness while acknowledging non-normal tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.analytics.performance import periods_per_year
from quantlab.exceptions import DataError

__all__ = [
    "annualized_vol",
    "beta",
    "expected_shortfall",
    "rolling_vol",
    "value_at_risk",
]

_METHODS = ("historical", "normal", "cornish_fisher")


def _clean(returns: pd.Series, min_obs: int = 2) -> pd.Series:
    out = returns.dropna()
    if len(out) < min_obs:
        raise DataError(f"Need at least {min_obs} returns, got {len(out)}")
    if not np.isfinite(out.to_numpy()).all():
        raise DataError("Returns contain non-finite values (inf)")
    return out


def _check_level(level: float) -> float:
    if not 0.5 <= level < 1.0:
        raise DataError(f"level must be in [0.5, 1), got {level}")
    return level


def annualized_vol(returns: pd.Series, periods: float | None = None) -> float:
    """Annualized standard deviation; frequency inferred when ``periods`` is None."""
    r = _clean(returns)
    ppy = periods_per_year(r.index) if periods is None else float(periods)
    return float(r.std() * np.sqrt(ppy))


def rolling_vol(
    returns: pd.Series, window: int = 63, periods: float | None = None
) -> pd.Series:
    """Annualized volatility over a trailing window, as a series."""
    if window < 2:
        raise DataError(f"window must be >= 2, got {window}")
    r = _clean(returns, min_obs=window)
    ppy = periods_per_year(r.index) if periods is None else float(periods)
    return (r.rolling(window).std() * np.sqrt(ppy)).rename(f"vol_{window}")


def _cornish_fisher_z(z: np.ndarray, skew: float, ex_kurt: float) -> np.ndarray:
    """Cornish-Fisher expansion of the standard-normal quantile ``z``."""
    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * ex_kurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def value_at_risk(
    returns: pd.Series, level: float = 0.95, method: str = "historical"
) -> float:
    """Value at Risk at ``level``, as a positive loss fraction.

    The one-period loss that is exceeded with probability ``1 - level``.
    """
    _check_level(level)
    r = _clean(returns, min_obs=3)
    tail = 1.0 - level
    if method == "historical":
        return float(-r.quantile(tail))
    if method == "normal":
        z = stats.norm.ppf(tail)
        return float(-(r.mean() + z * r.std()))
    if method == "cornish_fisher":
        z = np.array([stats.norm.ppf(tail)])
        z_cf = _cornish_fisher_z(z, float(r.skew()), float(r.kurtosis()))
        return float(-(r.mean() + z_cf[0] * r.std()))
    raise DataError(f"Unknown VaR method '{method}'; expected one of {_METHODS}")


def expected_shortfall(
    returns: pd.Series, level: float = 0.95, method: str = "historical"
) -> float:
    """Expected Shortfall (CVaR): the average loss *given* a VaR breach.

    Answers "when the bad tail happens, how bad is it on average?" — a
    strictly more informative number than VaR, which only marks the
    tail's edge. Positive loss fraction, always >= the matching VaR.
    """
    _check_level(level)
    r = _clean(returns, min_obs=3)
    tail = 1.0 - level
    if method == "historical":
        cutoff = r.quantile(tail)
        losses = r[r <= cutoff]
        return float(-losses.mean())
    if method == "normal":
        z = stats.norm.ppf(tail)
        return float(-(r.mean() - r.std() * stats.norm.pdf(z) / tail))
    if method == "cornish_fisher":
        # Average the CF-adjusted quantile over the tail (midpoint rule):
        # ES = -E[q_u | u < tail]. With zero skew/kurtosis this converges
        # to the normal ES.
        u = (np.arange(200) + 0.5) / 200.0 * tail
        z_cf = _cornish_fisher_z(stats.norm.ppf(u), float(r.skew()), float(r.kurtosis()))
        return float(-(r.mean() + z_cf.mean() * r.std()))
    raise DataError(f"Unknown ES method '{method}'; expected one of {_METHODS}")


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """OLS beta of ``returns`` on ``benchmark`` over their common dates."""
    r_aligned, b_aligned = returns.align(benchmark, join="inner")
    aligned = pd.DataFrame({"r": r_aligned, "b": b_aligned}).dropna()
    if len(aligned) < 3:
        raise DataError(
            f"Need at least 3 overlapping observations, got {len(aligned)}"
        )
    bench = aligned["b"].to_numpy()
    var = bench.var(ddof=1)
    if var <= 0:
        raise DataError("Benchmark has zero variance; beta is undefined")
    cov = np.cov(aligned["r"].to_numpy(), bench, ddof=1)[0, 1]
    return float(cov / var)
