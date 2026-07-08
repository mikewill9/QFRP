"""Declarative risk limits.

A :class:`RiskLimit` inspects a :class:`PortfolioState` snapshot and
reports zero or more :class:`LimitBreach` records; :class:`RiskMonitor`
runs a list of limits and (optionally) raises when anything breaches.
Limits are declarative — they describe *what must hold*, not what to do
about it — so the same limit set can drive a hard stop in one context
and a warning report in another.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from quantlab.exceptions import DataError, RiskLimitError
from quantlab.risk.drawdown import drawdown_series
from quantlab.risk.metrics import annualized_vol

__all__ = [
    "DrawdownLimit",
    "GrossLeverageLimit",
    "LimitBreach",
    "PortfolioState",
    "PositionLimit",
    "RiskLimit",
    "RiskMonitor",
    "VolLimit",
]


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Snapshot handed to limits.

    ``weights`` is the current weight vector. ``returns`` is the realized
    per-period return history up to now (required by history-based limits
    such as drawdown and volatility). ``timestamp`` labels breaches.
    """

    weights: pd.Series
    returns: pd.Series | None = None
    timestamp: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class LimitBreach:
    """One violated limit: what was measured against what threshold."""

    limit: str
    message: str
    value: float
    threshold: float
    timestamp: pd.Timestamp | None = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        when = f" at {self.timestamp}" if self.timestamp is not None else ""
        return f"[{self.limit}]{when} {self.message}"


class RiskLimit(ABC):
    """Base class: check one invariant against a portfolio state."""

    #: Identifier used in breach records.
    name: str = "risk_limit"

    @abstractmethod
    def check(self, state: PortfolioState) -> list[LimitBreach]:
        """Breaches found in ``state`` (empty list = compliant)."""

    def _require_returns(self, state: PortfolioState) -> pd.Series:
        if state.returns is None:
            raise DataError(f"{self.name} requires PortfolioState.returns")
        return state.returns

    def _breach(
        self, state: PortfolioState, message: str, value: float, threshold: float
    ) -> LimitBreach:
        return LimitBreach(
            limit=self.name,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=state.timestamp,
        )


class GrossLeverageLimit(RiskLimit):
    """Gross exposure ``sum |w|`` must not exceed ``max_gross``."""

    name = "gross_leverage"

    def __init__(self, max_gross: float) -> None:
        if max_gross <= 0:
            raise DataError(f"max_gross must be positive, got {max_gross}")
        self.max_gross = max_gross

    def check(self, state: PortfolioState) -> list[LimitBreach]:
        gross = float(state.weights.fillna(0.0).abs().sum())
        if gross <= self.max_gross:
            return []
        return [
            self._breach(
                state,
                f"gross exposure {gross:.3f} exceeds {self.max_gross:.3f}",
                gross,
                self.max_gross,
            )
        ]


class PositionLimit(RiskLimit):
    """No single position's ``|w|`` may exceed ``max_weight``."""

    name = "position"

    def __init__(self, max_weight: float) -> None:
        if max_weight <= 0:
            raise DataError(f"max_weight must be positive, got {max_weight}")
        self.max_weight = max_weight

    def check(self, state: PortfolioState) -> list[LimitBreach]:
        w = state.weights.fillna(0.0)
        offenders = w[w.abs() > self.max_weight]
        return [
            self._breach(
                state,
                f"|{symbol}| weight {abs(value):.3f} exceeds {self.max_weight:.3f}",
                float(abs(value)),
                self.max_weight,
            )
            for symbol, value in offenders.items()
        ]


class DrawdownLimit(RiskLimit):
    """Current drawdown must not exceed ``max_drawdown`` (a positive fraction)."""

    name = "drawdown"

    def __init__(self, max_drawdown: float) -> None:
        if not 0 < max_drawdown < 1:
            raise DataError(f"max_drawdown must be in (0, 1), got {max_drawdown}")
        self.max_drawdown = max_drawdown

    def check(self, state: PortfolioState) -> list[LimitBreach]:
        returns = self._require_returns(state)
        current = float(-drawdown_series(returns).iloc[-1])
        if current <= self.max_drawdown:
            return []
        return [
            self._breach(
                state,
                f"drawdown {current:.1%} exceeds {self.max_drawdown:.1%}",
                current,
                self.max_drawdown,
            )
        ]


class VolLimit(RiskLimit):
    """Trailing annualized volatility must not exceed ``max_vol``.

    Silent during the warmup while fewer than ``window`` returns exist —
    there is not yet enough data to estimate the quantity being limited.
    """

    name = "volatility"

    def __init__(self, max_vol: float, window: int = 63) -> None:
        if max_vol <= 0:
            raise DataError(f"max_vol must be positive, got {max_vol}")
        if window < 2:
            raise DataError(f"window must be >= 2, got {window}")
        self.max_vol = max_vol
        self.window = window

    def check(self, state: PortfolioState) -> list[LimitBreach]:
        returns = self._require_returns(state).dropna()
        if len(returns) < self.window:
            return []
        vol = annualized_vol(returns.iloc[-self.window :])
        if vol <= self.max_vol:
            return []
        return [
            self._breach(
                state,
                f"{self.window}-bar vol {vol:.1%} exceeds {self.max_vol:.1%}",
                vol,
                self.max_vol,
            )
        ]


@dataclass(frozen=True, slots=True)
class RiskMonitor:
    """Evaluate a list of limits against portfolio states.

    With ``raise_on_breach=True`` any breach raises
    :class:`~quantlab.exceptions.RiskLimitError`; otherwise breaches are
    returned for the caller to log or report.
    """

    limits: list[RiskLimit] = field(default_factory=list)
    raise_on_breach: bool = False

    def evaluate(self, state: PortfolioState) -> list[LimitBreach]:
        breaches = [b for limit in self.limits for b in limit.check(state)]
        if breaches and self.raise_on_breach:
            detail = "; ".join(str(b) for b in breaches)
            raise RiskLimitError(f"{len(breaches)} risk limit breach(es): {detail}")
        return breaches
