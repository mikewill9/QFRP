"""Volatility factors.

All estimators return annualized volatility internally; with the default
``invert=True`` the emitted score is *negative* volatility so that low-vol
names rank high, matching the low-volatility anomaly. Pass ``invert=False``
to obtain the raw (positive) volatility estimate as the score.
"""

from __future__ import annotations

import numpy as np

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = ["EWMAVolatility", "RealizedVolatility"]

_ESTIMATORS = ("cc", "parkinson", "garman_klass")


class RealizedVolatility(Factor):
    """Rolling realized volatility from one of three estimators.

    - ``cc``: close-to-close, root mean of squared log returns.
    - ``parkinson``: high/low range estimator,
      ``(1 / 4 ln 2) * ln(H/L)^2`` per bar; more efficient than ``cc`` when
      bars have clean highs/lows.
    - ``garman_klass``: OHLC estimator,
      ``0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2``.
    """

    name = "realized_vol"

    def __init__(
        self,
        window: int = 63,
        estimator: str = "cc",
        invert: bool = True,
    ) -> None:
        if window < 5:
            raise DataError("window must be >= 5")
        if estimator not in _ESTIMATORS:
            raise DataError(f"estimator must be one of {_ESTIMATORS}")
        self.window = window
        self.estimator = estimator
        self.invert = invert
        # Close-close needs one extra bar to form the first return.
        self.min_history = window + (1 if estimator == "cc" else 0)

    def _per_bar_variance(self, dataset: MarketDataset) -> WideFrame:
        if self.estimator == "cc":
            return dataset.returns(method="log") ** 2
        hl = np.log(dataset.high / dataset.low)
        if self.estimator == "parkinson":
            return hl**2 / (4.0 * np.log(2.0))
        co = np.log(dataset.close / dataset.open)
        return 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        ppy = dataset.frequency.periods_per_year
        var = self._per_bar_variance(dataset)
        vol = np.sqrt(var.rolling(self.window).mean() * ppy)
        return -vol if self.invert else vol


class EWMAVolatility(Factor):
    """Exponentially weighted volatility (RiskMetrics-style).

    Reacts faster to regime shifts than an equal-weight window; ``halflife``
    is in bars.
    """

    name = "ewma_vol"

    def __init__(self, halflife: float = 32.0, invert: bool = True) -> None:
        if halflife <= 0:
            raise DataError("halflife must be positive")
        self.halflife = halflife
        self.invert = invert
        self.min_history = int(halflife) + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        ppy = dataset.frequency.periods_per_year
        sq = dataset.returns(method="log") ** 2
        vol = np.sqrt(
            sq.ewm(halflife=self.halflife, min_periods=self.min_history - 1).mean() * ppy
        )
        return -vol if self.invert else vol
