"""Mean-reversion factors."""

from __future__ import annotations

import numpy as np

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = ["PriceToMovingAverage", "ShortTermReversal"]


class ShortTermReversal(Factor):
    """Negative of the trailing short-horizon return.

    Recent losers score high, recent winners low, capturing the well
    documented one-week to one-month reversal effect. This is the mirror of
    the window that :class:`~quantlab.research.factors.momentum.Momentum`
    deliberately skips.
    """

    name = "st_reversal"

    def __init__(self, lookback: int = 5) -> None:
        if lookback < 1:
            raise DataError("lookback must be >= 1")
        self.lookback = lookback
        self.min_history = lookback + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        logp = np.log(dataset.close)
        return -(logp - logp.shift(self.lookback))


class PriceToMovingAverage(Factor):
    """Reversion toward a trailing moving average.

    Score at ``t`` is the negative z-score of the close against its trailing
    ``window`` mean and standard deviation: prices stretched far below their
    average score high (expected snap-back), far above score low.
    """

    name = "price_to_ma"

    def __init__(self, window: int = 21) -> None:
        if window < 3:
            raise DataError("window must be >= 3")
        self.window = window
        self.min_history = window

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        close = dataset.close
        ma = close.rolling(self.window).mean()
        sd = close.rolling(self.window).std()
        return -(close - ma) / sd
