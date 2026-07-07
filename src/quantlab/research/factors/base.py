"""Factor base class.

A factor maps a :class:`~quantlab.data.MarketDataset` to a wide frame of
real-valued scores where **higher = more attractive** under the factor's
hypothesis. Scores indexed at date ``t`` may use information up to and
including the close of ``t`` and nothing later; the evaluation and ML layers
align scores at ``t`` against *forward* returns over ``(t, t+h]``.

Factors are stateless: all configuration lives in constructor arguments,
echoed via :attr:`Factor.params` so experiment tracking can record exactly
what produced a score frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["Factor"]


class Factor(ABC):
    """Base class for all alpha factors.

    Subclasses set :attr:`name`, ensure :attr:`min_history` reflects the
    number of bars needed before the first non-NaN score, and implement
    :meth:`compute`. Rows earlier than ``min_history`` should be NaN, never
    silently backfilled.
    """

    #: Short identifier used in evaluation reports and experiment logs.
    name: str = "factor"

    #: Bars of history required before the first valid score.
    min_history: int = 1

    @abstractmethod
    def compute(self, dataset: MarketDataset) -> WideFrame:
        """Compute the score frame (same index/columns as the dataset)."""

    @property
    def params(self) -> dict[str, Any]:
        """Constructor configuration (public instance attributes)."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def _require_history(self, dataset: MarketDataset) -> None:
        """Raise if the dataset is shorter than ``min_history`` bars."""
        if dataset.n_periods < self.min_history:
            raise DataError(
                f"Factor '{self.name}' needs >= {self.min_history} bars; "
                f"dataset has {dataset.n_periods}"
            )

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
