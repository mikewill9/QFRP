"""Momentum factors."""

from __future__ import annotations

import numpy as np

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = ["Momentum", "TimeSeriesMomentum"]


class Momentum(Factor):
    """Classic price momentum: trailing return excluding the recent past.

    Score at ``t`` is the log return from ``t - lookback`` to ``t - skip``.
    The skip window (default one month) removes the short-term reversal
    effect that otherwise contaminates the signal; ``lookback=252, skip=21``
    reproduces the standard 12-1 construction.
    """

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        if lookback <= skip:
            raise DataError("lookback must exceed skip")
        if skip < 0:
            raise DataError("skip must be >= 0")
        self.lookback = lookback
        self.skip = skip
        self.min_history = lookback + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        logp = np.log(dataset.close)
        return logp.shift(self.skip) - logp.shift(self.lookback)


class TimeSeriesMomentum(Factor):
    """Volatility-scaled own-trend momentum.

    Score at ``t`` is the trailing ``lookback`` log return divided by its
    realized volatility over the same window, i.e. a t-statistic-like measure
    of trend strength. Unlike :class:`Momentum` it is meaningful per asset in
    isolation (positive = uptrend), not only cross-sectionally.
    """

    name = "ts_momentum"

    def __init__(self, lookback: int = 126) -> None:
        if lookback < 5:
            raise DataError("lookback must be >= 5")
        self.lookback = lookback
        self.min_history = lookback + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        logp = np.log(dataset.close)
        rets = dataset.returns(method="log")
        total = logp - logp.shift(self.lookback)
        scale = rets.rolling(self.lookback).std() * np.sqrt(self.lookback)
        return total / scale
