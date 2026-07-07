"""Liquidity and trading-cost factors.

Sign conventions (higher score = more attractive):

- :class:`AmihudIlliquidity` scores *illiquid* names high, following the
  illiquidity-premium hypothesis. Flip the sign downstream if you instead
  want a tradability screen.
- :class:`DollarVolume` scores *liquid* names high (log average dollar
  volume); use it for universe construction and capacity-aware tilts.
- :class:`RollSpread` scores *tight-spread* names high (score is the
  negative estimated effective spread).
"""

from __future__ import annotations

import numpy as np

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = ["AmihudIlliquidity", "DollarVolume", "RollSpread"]


class AmihudIlliquidity(Factor):
    """Amihud (2002) price-impact proxy.

    Rolling mean of ``|simple return| / dollar volume``, scaled by 1e6 for
    readable magnitudes. High values mean a dollar of flow moves the price a
    lot, i.e. the name is illiquid.
    """

    name = "amihud"

    def __init__(self, window: int = 21) -> None:
        if window < 5:
            raise DataError("window must be >= 5")
        self.window = window
        self.min_history = window + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        abs_ret = dataset.returns(method="simple").abs()
        dv = dataset.dollar_volume().replace(0.0, np.nan)
        return (abs_ret / dv).rolling(self.window).mean() * 1e6


class DollarVolume(Factor):
    """Log trailing average dollar volume (liquidity/capacity measure)."""

    name = "dollar_volume"

    def __init__(self, window: int = 21) -> None:
        if window < 1:
            raise DataError("window must be >= 1")
        self.window = window
        self.min_history = window

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        adv = dataset.dollar_volume().rolling(self.window).mean()
        return np.log(adv.replace(0.0, np.nan))


class RollSpread(Factor):
    """Roll (1984) effective-spread estimator from price-change covariance.

    Under the Roll model, bid-ask bounce induces negative first-order
    autocovariance in price changes; the implied spread is
    ``2 * sqrt(-cov(dp_t, dp_{t-1}))`` when that covariance is negative
    (clamped at zero otherwise), expressed here as a fraction of price.
    Score is the negative spread, so tight names rank high.
    """

    name = "roll_spread"

    def __init__(self, window: int = 21) -> None:
        if window < 10:
            raise DataError("window must be >= 10 for a usable covariance")
        self.window = window
        self.min_history = window + 2

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        dp = dataset.close.diff()
        lag = dp.shift(1)
        w = self.window
        # Rolling covariance via E[xy] - E[x]E[y] on the paired window.
        exy = (dp * lag).rolling(w).mean()
        ex = dp.rolling(w).mean()
        ey = lag.rolling(w).mean()
        cov = exy - ex * ey
        spread = 2.0 * np.sqrt((-cov).clip(lower=0.0)) / dataset.close
        return -spread
