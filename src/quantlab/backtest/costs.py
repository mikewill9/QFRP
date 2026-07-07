"""Transaction cost models.

Two orthogonal components, composed by the execution simulator:

- :class:`CommissionModel`: explicit fees in dollars per fill.
- :class:`SlippageModel`: adverse price movement expressed as a
  *fraction of the reference price*, always applied against the trader
  (buys fill higher, sells lower).

The workhorse slippage model is :class:`VolumeImpactSlippage`, half spread
plus a square-root market impact term
``impact_coeff * sigma_daily * sqrt(|q| / ADV)``, the standard functional
form for institutional cost models: costs grow with participation, but
sublinearly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quantlab.exceptions import BacktestError

__all__ = [
    "BpsCommission",
    "CommissionModel",
    "FixedBpsSlippage",
    "PerShareCommission",
    "SlippageModel",
    "VolumeImpactSlippage",
    "ZeroCommission",
    "ZeroSlippage",
]


# ---------------------------------------------------------------- commission
class CommissionModel(ABC):
    """Explicit fee charged per fill."""

    @abstractmethod
    def commission(self, quantity: float, price: float) -> float:
        """Dollar commission for a fill of ``quantity`` at ``price`` (>= 0)."""


class ZeroCommission(CommissionModel):
    """Frictionless fees (baseline runs, cost-attribution deltas)."""

    def commission(self, quantity: float, price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """US-broker style: cents per share with a per-order minimum."""

    def __init__(self, rate: float = 0.005, minimum: float = 1.0) -> None:
        if rate < 0 or minimum < 0:
            raise BacktestError("rate and minimum must be >= 0")
        self.rate = rate
        self.minimum = minimum

    def commission(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0
        return max(abs(quantity) * self.rate, self.minimum)


class BpsCommission(CommissionModel):
    """Fee proportional to executed notional (basis points)."""

    def __init__(self, bps: float = 1.0) -> None:
        if bps < 0:
            raise BacktestError("bps must be >= 0")
        self.bps = bps

    def commission(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.bps / 1e4


# ------------------------------------------------------------------ slippage
class SlippageModel(ABC):
    """Adverse price movement as a fraction of the reference price."""

    @abstractmethod
    def slippage_fraction(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        sigma: float | None,
    ) -> float:
        """Non-negative fractional price penalty for this fill.

        Parameters
        ----------
        quantity:
            Signed fill quantity.
        price:
            Unperturbed reference price.
        volume:
            Bar volume in shares (None/NaN when unavailable).
        sigma:
            Trailing daily return volatility estimate known *before* the
            fill (None/NaN when unavailable).
        """


class ZeroSlippage(SlippageModel):
    """No slippage (idealized fills)."""

    def slippage_fraction(self, quantity, price, volume, sigma) -> float:
        return 0.0


class FixedBpsSlippage(SlippageModel):
    """Constant slippage in basis points regardless of size."""

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise BacktestError("bps must be >= 0")
        self.bps = bps

    def slippage_fraction(self, quantity, price, volume, sigma) -> float:
        return self.bps / 1e4


class VolumeImpactSlippage(SlippageModel):
    """Half spread plus square-root market impact.

    ``fraction = spread_bps / 2e4
               + impact_coeff * sigma * sqrt(|quantity| / volume)``

    Parameters
    ----------
    spread_bps:
        Full quoted spread in bps; half is paid on each fill.
    impact_coeff:
        Impact scale; ~0.1 is a common starting calibration for the
        square-root law on daily participation.
    fallback_sigma:
        Daily vol used when no estimate is available yet (warmup bars).

    When bar volume is unavailable the impact term is dropped and only the
    half spread applies; the execution layer separately caps participation.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_coeff: float = 0.1,
        fallback_sigma: float = 0.02,
    ) -> None:
        if spread_bps < 0 or impact_coeff < 0 or fallback_sigma <= 0:
            raise BacktestError(
                "spread_bps/impact_coeff must be >= 0, fallback_sigma > 0"
            )
        self.spread_bps = spread_bps
        self.impact_coeff = impact_coeff
        self.fallback_sigma = fallback_sigma

    def slippage_fraction(self, quantity, price, volume, sigma) -> float:
        frac = self.spread_bps / 2e4
        if volume is not None and np.isfinite(volume) and volume > 0:
            s = sigma if (sigma is not None and np.isfinite(sigma) and sigma > 0) \
                else self.fallback_sigma
            frac += self.impact_coeff * s * np.sqrt(abs(quantity) / volume)
        return float(frac)
