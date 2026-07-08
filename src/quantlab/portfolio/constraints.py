"""Portfolio constraints.

:class:`Constraints` is a declarative bundle of position and exposure
limits used two ways:

1. ``project(weights)`` maps any weight vector to a nearby feasible one.
   This is a fast *approximate* projection (clip, then rescale), suitable
   for cleaning up the output of heuristic optimizers (inverse-vol, ERC,
   HRP). It is not the exact Euclidean projection onto the feasible set.
2. ``scipy_bounds()`` / ``scipy_constraints()`` express the same limits in
   the form :func:`scipy.optimize.minimize` (SLSQP) consumes, so solver-
   based optimizers (mean-variance, max-Sharpe) satisfy them exactly.

Conventions: weights are fractions of equity; gross exposure is
``sum |w|``; net exposure is ``sum w``; ``max_net`` caps ``|net|``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.exceptions import OptimizationError

__all__ = ["Constraints"]


@dataclass(frozen=True, slots=True)
class Constraints:
    """Position and exposure limits.

    Parameters
    ----------
    long_only:
        Forbid negative weights.
    max_weight / min_weight:
        Per-asset bounds. ``long_only`` implies an effective lower bound
        of 0 (a negative ``min_weight`` is then rejected).
    max_gross:
        Cap on gross exposure ``sum |w|`` (leverage).
    max_net:
        Cap on absolute net exposure ``|sum w|``.
    """

    long_only: bool = False
    max_weight: float | None = None
    min_weight: float | None = None
    max_gross: float | None = None
    max_net: float | None = None

    def __post_init__(self) -> None:
        if self.max_weight is not None and self.max_weight <= 0:
            raise OptimizationError(f"max_weight must be positive, got {self.max_weight}")
        if (
            self.min_weight is not None
            and self.max_weight is not None
            and self.min_weight > self.max_weight
        ):
            raise OptimizationError(
                f"min_weight {self.min_weight} exceeds max_weight {self.max_weight}"
            )
        if self.long_only and self.min_weight is not None and self.min_weight < 0:
            raise OptimizationError("long_only conflicts with a negative min_weight")
        if self.max_gross is not None and self.max_gross <= 0:
            raise OptimizationError(f"max_gross must be positive, got {self.max_gross}")
        if self.max_net is not None and self.max_net < 0:
            raise OptimizationError(f"max_net must be >= 0, got {self.max_net}")

    # ------------------------------------------------------------ projection

    def project(self, weights: pd.Series) -> pd.Series:
        """Return a feasible weight vector near ``weights``.

        Order of operations: clip to per-asset bounds (and to zero from
        below if long-only), then scale everything down if gross exposure
        exceeds ``max_gross``, then scale again if ``|net|`` exceeds
        ``max_net``. Scaling preserves relative sizes but shrinks total
        exposure, so the result can be more conservative than the closest
        feasible point; that is the accepted tradeoff for a closed-form
        projection.
        """
        w = weights.astype(float).fillna(0.0)
        lo = 0.0 if self.long_only else -np.inf
        if self.min_weight is not None:
            lo = max(lo, self.min_weight)
        hi = np.inf if self.max_weight is None else self.max_weight
        w = w.clip(lower=lo if np.isfinite(lo) else None, upper=hi if np.isfinite(hi) else None)

        gross = float(w.abs().sum())
        if self.max_gross is not None and gross > self.max_gross > 0:
            w = w * (self.max_gross / gross)
        net = float(w.sum())
        if self.max_net is not None and abs(net) > self.max_net:
            if self.max_net == 0.0:
                # Cannot reach net 0 by scaling; demean instead (may leave
                # per-asset bounds slightly conservative after re-clip).
                w = w - net / len(w)
                w = w.clip(lower=lo if np.isfinite(lo) else None,
                           upper=hi if np.isfinite(hi) else None)
            else:
                w = w * (self.max_net / abs(net))
        return w

    # ------------------------------------------------------------ solver hooks

    def scipy_bounds(self, n: int) -> list[tuple[float | None, float | None]]:
        """Per-asset ``(lo, hi)`` bounds for :func:`scipy.optimize.minimize`."""
        lo: float | None = 0.0 if self.long_only else None
        if self.min_weight is not None:
            lo = self.min_weight if lo is None else max(lo, self.min_weight)
        hi = self.max_weight
        return [(lo, hi)] * n

    def scipy_constraints(self) -> list[dict[str, Any]]:
        """Inequality constraints (``fun >= 0`` form) for SLSQP.

        The gross constraint uses ``sum |w|``, which is not differentiable
        at zero weights; SLSQP copes in practice because the optimum
        rarely sits exactly on a kink, and the solver-based optimizers
        verify feasibility of the returned point anyway.
        """
        cons: list[dict[str, Any]] = []
        if self.max_gross is not None:
            gross_cap = self.max_gross
            cons.append({
                "type": "ineq",
                "fun": lambda w, cap=gross_cap: cap - np.abs(w).sum(),
            })
        if self.max_net is not None:
            net_cap = self.max_net
            cons.append({
                "type": "ineq",
                "fun": lambda w, cap=net_cap: cap - abs(w.sum()),
            })
        return cons

    def is_satisfied(self, weights: pd.Series, tol: float = 1e-6) -> bool:
        """True if ``weights`` violates no limit by more than ``tol``."""
        w = weights.astype(float).fillna(0.0)
        if self.long_only and (w < -tol).any():
            return False
        if self.max_weight is not None and (w > self.max_weight + tol).any():
            return False
        if self.min_weight is not None and (w < self.min_weight - tol).any():
            return False
        if self.max_gross is not None and w.abs().sum() > self.max_gross + tol:
            return False
        return not (
            self.max_net is not None and abs(w.sum()) > self.max_net + tol
        )
