"""Covariance estimation.

All estimators take a wide frame of per-period returns (research code
passes log returns) and produce a **per-period** covariance matrix as a
symbol-by-symbol DataFrame. Nothing here annualizes; multiply by periods
per year at the point of use so one estimate can serve any horizon.

Why more than one estimator exists
----------------------------------
The sample covariance is unbiased but noisy: with ``N`` assets it has
``N * (N + 1) / 2`` free parameters, and when the observation count is not
a large multiple of that, its extreme entries are estimation error, which
optimizers then amplify. ``ewma_cov`` reacts faster to regime changes by
weighting recent data more; ``ledoit_wolf_cc`` trades a little bias for a
large variance reduction by shrinking toward a structured target.

Rows containing any NaN are dropped before estimation (an asset with a
missing return contributes no cross-sectional information that period);
estimators raise :class:`~quantlab.exceptions.DataError` when too few
complete rows remain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["ewma_cov", "ledoit_wolf_cc", "sample_cov"]


def _complete_rows(returns: WideFrame, min_obs: int) -> WideFrame:
    """Validate a returns frame and drop rows with any missing value."""
    if not isinstance(returns, pd.DataFrame):
        raise DataError("Covariance estimators expect a wide DataFrame of returns")
    if returns.shape[1] == 0:
        raise DataError("Returns frame has no columns")
    clean = returns.dropna(how="any")
    if len(clean) < min_obs:
        raise DataError(
            f"Need at least {min_obs} complete return rows, got {len(clean)} "
            f"(of {len(returns)} total after dropping rows with NaN)"
        )
    if not np.isfinite(clean.to_numpy()).all():
        raise DataError("Returns contain non-finite values (inf)")
    return clean


def sample_cov(returns: WideFrame) -> pd.DataFrame:
    """Per-period sample covariance (``ddof=1``) over complete rows."""
    clean = _complete_rows(returns, min_obs=2)
    x = clean.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    cov = x.T @ x / (len(x) - 1)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def ewma_cov(returns: WideFrame, halflife: float = 63.0) -> pd.DataFrame:
    """Exponentially weighted per-period covariance.

    Observation at lag ``k`` (0 = most recent) gets weight proportional to
    ``0.5 ** (k / halflife)``; weights are normalized to sum to one and the
    mean is the same weighted mean (so the estimate is a proper weighted
    central moment). As ``halflife -> inf`` this converges to the
    equal-weight (``ddof=0``) sample covariance.
    """
    if halflife <= 0:
        raise DataError(f"halflife must be positive, got {halflife}")
    clean = _complete_rows(returns, min_obs=2)
    x = clean.to_numpy(dtype=float)
    lags = np.arange(len(x) - 1, -1, -1, dtype=float)  # oldest row = largest lag
    w = np.power(0.5, lags / halflife)
    w /= w.sum()
    mean = w @ x
    xc = x - mean
    cov = (xc * w[:, None]).T @ xc
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def ledoit_wolf_cc(returns: WideFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward the constant-correlation target.

    Returns ``(covariance, intensity)`` where ``intensity`` in ``[0, 1]``
    is the estimated optimal weight on the target. The target keeps every
    asset's sample variance but replaces all pairwise correlations with
    their cross-sectional average; the intensity that minimizes expected
    Frobenius loss is estimated from the data (Ledoit & Wolf, 2004,
    "Honey, I Shrunk the Sample Covariance Matrix").

    With one asset (or when the target equals the sample matrix, e.g. two
    assets) there is nothing to shrink and the intensity is 0.
    """
    clean = _complete_rows(returns, min_obs=3)
    x = clean.to_numpy(dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    # The paper's estimator uses the 1/T (maximum-likelihood) normalization.
    s = x.T @ x / t
    var = np.diag(s).copy()
    sd = np.sqrt(var)
    if np.any(sd <= 0):
        zero = [clean.columns[i] for i in np.flatnonzero(sd <= 0)]
        raise DataError(f"Zero-variance assets make correlation undefined: {zero[:5]}")

    corr = s / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi: asymptotic variance of the sample covariance entries.
    y = x**2
    pi_mat = y.T @ y / t - s**2
    pi_hat = pi_mat.sum()

    # rho: covariance between the sample estimates and the target's entries.
    # theta[i, j] = asymptotic cov of (s_ii, s_ij); the paper's rho sums the
    # diagonal of pi plus a correlation-weighted combination of the thetas.
    theta_ii = (x**3).T @ x / t - var[:, None] * s
    rho_hat = float(np.trace(pi_mat))
    scale = np.outer(1.0 / sd, sd)  # scale[i, j] = sd_j / sd_i
    off = ~np.eye(n, dtype=bool)
    rho_hat += (r_bar / 2.0) * float(
        (scale * theta_ii + scale.T * theta_ii.T)[off].sum()
    )

    # gamma: squared distance between sample and target.
    gamma_hat = float(((target - s) ** 2).sum())

    if gamma_hat <= 1e-16:
        intensity = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        intensity = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = intensity * target + (1.0 - intensity) * s
    frame = pd.DataFrame(shrunk, index=clean.columns, columns=clean.columns)
    return frame, intensity
