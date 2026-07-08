"""Portfolio optimizers.

Every optimizer implements one method::

    weights(mu, cov, prev_weights=None) -> pd.Series

``cov`` is a per-period covariance DataFrame (see ``covariance.py``);
``mu`` is a Series of per-period expected returns aligned to ``cov`` —
producing ``mu`` is the caller's responsibility, and the risk-based
optimizers (inverse-vol, ERC, HRP) ignore it entirely. ``prev_weights``
is part of the stable interface for turnover-aware optimizers; the
bundled implementations accept and ignore it (turnover control lives in
``rebalance.py``).

Two solver strategies (see ``constraints.py``):

- Heuristic optimizers compute closed-form/iterative weights and then
  apply ``Constraints.project`` (approximate).
- ``MeanVariance`` and ``MaxSharpe`` hand bounds and exposure constraints
  to scipy's SLSQP and satisfy them exactly.

Optimizers that invert the covariance raise
:class:`~quantlab.exceptions.OptimizationError` when the matrix is
singular or ill-conditioned instead of silently regularizing; use
``ledoit_wolf_cc`` (or more data) to get an invertible estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.optimize as sco
from scipy.spatial.distance import squareform

from quantlab.exceptions import OptimizationError
from quantlab.portfolio.constraints import Constraints

__all__ = [
    "EqualRiskContribution",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaxSharpe",
    "MeanVariance",
    "Optimizer",
    "VolatilityTargeting",
]

#: Condition numbers above this mean the inverse is numerically meaningless.
_COND_LIMIT = 1e12


def _validate_cov(cov: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate a covariance frame; return it (non-None) with its ndarray."""
    if not isinstance(cov, pd.DataFrame) or cov.shape[0] != cov.shape[1]:
        raise OptimizationError("cov must be a square DataFrame")
    if cov.shape[0] == 0:
        raise OptimizationError("cov is empty")
    if not cov.index.equals(cov.columns):
        raise OptimizationError("cov index and columns must match")
    values = cov.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise OptimizationError("cov contains non-finite values")
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-12):
        raise OptimizationError("cov must be symmetric")
    if (np.diag(values) <= 0).any():
        bad = [str(cov.index[i]) for i in np.flatnonzero(np.diag(values) <= 0)]
        raise OptimizationError(f"cov has non-positive variances: {bad[:5]}")
    return cov, values


def _align_mu(mu: pd.Series | None, cov: pd.DataFrame, who: str) -> np.ndarray:
    if mu is None:
        raise OptimizationError(f"{who} requires expected returns (mu)")
    aligned = mu.reindex(cov.index)
    if aligned.isna().any():
        missing = list(aligned.index[aligned.isna()])
        raise OptimizationError(f"mu is missing entries for: {missing[:5]}")
    return aligned.to_numpy(dtype=float)


def _require_invertible(values: np.ndarray) -> None:
    cond = np.linalg.cond(values)
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise OptimizationError(
            f"Covariance matrix is singular or ill-conditioned (cond={cond:.2e}); "
            "use shrinkage (ledoit_wolf_cc) or a longer sample instead"
        )


class Optimizer(ABC):
    """Base class: map ``(mu, cov)`` to portfolio weights."""

    #: Identifier for logs and experiment configs.
    name: str = "optimizer"

    @abstractmethod
    def weights(
        self,
        mu: pd.Series | None,
        cov: pd.DataFrame,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Portfolio weights indexed like ``cov``."""


class InverseVolatility(Optimizer):
    """Weight each asset by ``1 / sigma``, normalized to sum to one.

    The simplest risk-based scheme: ignores correlations entirely, so it
    is robust to estimation error but blind to concentration in
    correlated clusters (that is what ERC and HRP address).
    """

    name = "inverse_volatility"

    def __init__(self, constraints: Constraints | None = None) -> None:
        self.constraints = constraints

    def weights(
        self,
        mu: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov, values = _validate_cov(cov)
        inv_vol = 1.0 / np.sqrt(np.diag(values))
        w = pd.Series(inv_vol / inv_vol.sum(), index=cov.index, name=self.name)
        return self.constraints.project(w) if self.constraints else w


class EqualRiskContribution(Optimizer):
    """Long-only portfolio in which every asset contributes equal risk.

    Asset ``i``'s risk contribution is ``w_i * (cov @ w)_i / variance``;
    ERC makes all contributions ``1/n``. Unlike inverse-vol it accounts
    for correlations: two highly correlated assets share one risk budget.

    Solved by cyclical coordinate descent on the equivalent problem
    ``min sqrt(w' cov w) - sum(log w_i) / n``: each coordinate update is
    the positive root of a quadratic, and the fixed point (rescaled to
    sum to one) is the unique ERC portfolio.
    """

    name = "equal_risk_contribution"

    def __init__(
        self,
        constraints: Constraints | None = None,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ) -> None:
        if tol <= 0:
            raise OptimizationError("tol must be positive")
        self.constraints = constraints
        self.tol = tol
        self.max_iter = max_iter

    def weights(
        self,
        mu: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov, values = _validate_cov(cov)
        n = values.shape[0]
        diag = np.diag(values)
        budget = 1.0 / n
        w = (1.0 / np.sqrt(diag))
        w /= w.sum()

        for _ in range(self.max_iter):
            w_old = w.copy()
            for i in range(n):
                sigma_w = values @ w
                variance = float(w @ sigma_w)
                if variance <= 0:
                    raise OptimizationError(
                        "Portfolio variance went non-positive; cov is not PSD"
                    )
                vol = np.sqrt(variance)
                c = sigma_w[i] - diag[i] * w[i]
                w[i] = (-c + np.sqrt(c * c + 4.0 * diag[i] * budget * vol)) / (
                    2.0 * diag[i]
                )
            if np.max(np.abs(w - w_old)) < self.tol * max(np.max(np.abs(w_old)), 1e-30):
                break
        else:
            raise OptimizationError(
                f"ERC coordinate descent did not converge in {self.max_iter} sweeps"
            )

        result = pd.Series(w / w.sum(), index=cov.index, name=self.name)
        return self.constraints.project(result) if self.constraints else result


class MeanVariance(Optimizer):
    """Maximize ``mu'w - (risk_aversion / 2) w' cov w``.

    Without constraints the solution is the closed form
    ``cov^-1 mu / risk_aversion`` (an easy correctness check); with
    constraints the same objective is solved by SLSQP. There is no
    full-investment constraint: cash is implicit, and total exposure is
    governed by ``risk_aversion`` and any gross/net caps.
    """

    name = "mean_variance"

    def __init__(
        self, risk_aversion: float = 1.0, constraints: Constraints | None = None
    ) -> None:
        if risk_aversion <= 0:
            raise OptimizationError(f"risk_aversion must be positive, got {risk_aversion}")
        self.risk_aversion = risk_aversion
        self.constraints = constraints

    def weights(
        self,
        mu: pd.Series | None,
        cov: pd.DataFrame,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov, values = _validate_cov(cov)
        mu_vec = _align_mu(mu, cov, "MeanVariance")
        _require_invertible(values)
        analytic = np.linalg.solve(values, mu_vec) / self.risk_aversion
        if self.constraints is None:
            return pd.Series(analytic, index=cov.index, name=self.name)

        gamma = self.risk_aversion

        def objective(w: np.ndarray) -> float:
            return float(0.5 * gamma * w @ values @ w - mu_vec @ w)

        def gradient(w: np.ndarray) -> np.ndarray:
            return gamma * (values @ w) - mu_vec

        x0 = self.constraints.project(
            pd.Series(analytic, index=cov.index)
        ).to_numpy()
        result = sco.minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=self.constraints.scipy_bounds(len(x0)),
            constraints=self.constraints.scipy_constraints(),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise OptimizationError(f"MeanVariance SLSQP failed: {result.message}")
        w = pd.Series(result.x, index=cov.index, name=self.name)
        if not self.constraints.is_satisfied(w, tol=1e-6):
            w = self.constraints.project(w)
        return w


class MaxSharpe(Optimizer):
    """Maximize the ex-ante Sharpe ratio ``mu'w / sqrt(w' cov w)``.

    ``mu`` is treated as *excess* expected returns. The Sharpe ratio is
    scale-invariant, so a scale must be pinned: the returned portfolio has
    gross exposure ``max_gross`` when the constraints cap it, else 1.
    Without constraints this is the tangency portfolio
    ``cov^-1 mu`` rescaled to gross 1.
    """

    name = "max_sharpe"

    def __init__(self, constraints: Constraints | None = None) -> None:
        self.constraints = constraints

    def weights(
        self,
        mu: pd.Series | None,
        cov: pd.DataFrame,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov, values = _validate_cov(cov)
        mu_vec = _align_mu(mu, cov, "MaxSharpe")
        _require_invertible(values)
        tangency = np.linalg.solve(values, mu_vec)
        gross_target = 1.0
        if self.constraints is not None and self.constraints.max_gross is not None:
            gross_target = self.constraints.max_gross

        if self.constraints is None:
            w = tangency / np.abs(tangency).sum() * gross_target
            return pd.Series(w, index=cov.index, name=self.name)

        def neg_sharpe(w: np.ndarray) -> float:
            variance = float(w @ values @ w)
            if variance < 1e-18:
                return 1e6  # off the zero portfolio
            return -float(mu_vec @ w) / np.sqrt(variance)

        cons = self.constraints.scipy_constraints()
        cons.append({
            "type": "eq",
            "fun": lambda w, g=gross_target: np.abs(w).sum() - g,
        })
        x0 = self.constraints.project(
            pd.Series(tangency / np.abs(tangency).sum() * gross_target, index=cov.index)
        ).to_numpy()
        if np.abs(x0).sum() < 1e-12:  # projection collapsed the start point
            x0 = np.full(len(mu_vec), gross_target / len(mu_vec))
        result = sco.minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=self.constraints.scipy_bounds(len(x0)),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise OptimizationError(f"MaxSharpe SLSQP failed: {result.message}")
        w = pd.Series(result.x, index=cov.index, name=self.name)
        if not self.constraints.is_satisfied(w, tol=1e-6):
            w = self.constraints.project(w)
        return w


class HierarchicalRiskParity(Optimizer):
    """Lopez de Prado's HRP: cluster assets, then split risk down the tree.

    Three steps: (1) hierarchical clustering on the correlation-distance
    matrix ``sqrt((1 - rho) / 2)``; (2) reorder assets so correlated ones
    sit together (quasi-diagonalization); (3) recursive bisection, giving
    each half of every split capital inversely proportional to its
    inverse-variance-portfolio risk. Never inverts the covariance, which
    is exactly why it tolerates ill-conditioned estimates that break
    mean-variance.
    """

    name = "hierarchical_risk_parity"

    def __init__(
        self,
        linkage_method: str = "single",
        constraints: Constraints | None = None,
    ) -> None:
        self.linkage_method = linkage_method
        self.constraints = constraints

    def weights(
        self,
        mu: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov, values = _validate_cov(cov)
        n = values.shape[0]
        if n == 1:
            w = pd.Series([1.0], index=cov.index, name=self.name)
            return self.constraints.project(w) if self.constraints else w

        sd = np.sqrt(np.diag(values))
        corr = np.clip(values / np.outer(sd, sd), -1.0, 1.0)
        dist = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
        np.fill_diagonal(dist, 0.0)
        link = sch.linkage(squareform(dist, checks=False), method=self.linkage_method)
        order = list(sch.leaves_list(link))

        w = np.ones(n)
        clusters: list[list[int]] = [order]
        while clusters:
            cluster = clusters.pop()
            if len(cluster) < 2:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            alloc_left = 1.0 - self._cluster_var(values, left) / (
                self._cluster_var(values, left) + self._cluster_var(values, right)
            )
            w[left] *= alloc_left
            w[right] *= 1.0 - alloc_left
            clusters.extend([left, right])

        result = pd.Series(w / w.sum(), index=cov.index, name=self.name)
        return self.constraints.project(result) if self.constraints else result

    @staticmethod
    def _cluster_var(values: np.ndarray, members: list[int]) -> float:
        """Variance of the inverse-variance portfolio over ``members``."""
        sub = values[np.ix_(members, members)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)


class VolatilityTargeting(Optimizer):
    """Scale another optimizer's weights to a target annualized vol.

    Ex-ante vol is ``sqrt(w' cov w * periods_per_year)`` (per-period
    covariance annualized at the point of use). Scaling changes gross
    exposure, so an optional ``max_leverage`` caps the post-scaling gross;
    the inner optimizer's own constraints apply *before* scaling.
    """

    name = "volatility_targeting"

    def __init__(
        self,
        optimizer: Optimizer,
        target: float,
        periods_per_year: int = 252,
        max_leverage: float | None = None,
    ) -> None:
        if target <= 0:
            raise OptimizationError(f"target vol must be positive, got {target}")
        if periods_per_year <= 0:
            raise OptimizationError("periods_per_year must be positive")
        if max_leverage is not None and max_leverage <= 0:
            raise OptimizationError("max_leverage must be positive")
        self.optimizer = optimizer
        self.target = target
        self.periods_per_year = periods_per_year
        self.max_leverage = max_leverage

    def weights(
        self,
        mu: pd.Series | None,
        cov: pd.DataFrame,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        base = self.optimizer.weights(mu, cov, prev_weights)
        cov, values = _validate_cov(cov)
        aligned = base.reindex(cov.index).fillna(0.0).to_numpy()
        variance = float(aligned @ values @ aligned)
        if variance <= 0:
            raise OptimizationError(
                "Inner optimizer produced a zero-variance portfolio; cannot scale"
            )
        ann_vol = np.sqrt(variance * self.periods_per_year)
        scaled = base * (self.target / ann_vol)
        if self.max_leverage is not None:
            gross = float(scaled.abs().sum())
            if gross > self.max_leverage:
                scaled = scaled * (self.max_leverage / gross)
        return scaled.rename(self.name)
