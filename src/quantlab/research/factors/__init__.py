"""Alpha factor library. All factors implement the ``Factor`` contract."""

from quantlab.research.factors.base import Factor
from quantlab.research.factors.liquidity import (
    AmihudIlliquidity,
    DollarVolume,
    RollSpread,
)
from quantlab.research.factors.mean_reversion import (
    PriceToMovingAverage,
    ShortTermReversal,
)
from quantlab.research.factors.momentum import Momentum, TimeSeriesMomentum
from quantlab.research.factors.statistical import MarketBeta, PCAResidualReversal
from quantlab.research.factors.volatility import EWMAVolatility, RealizedVolatility

__all__ = [
    "AmihudIlliquidity",
    "DollarVolume",
    "EWMAVolatility",
    "Factor",
    "MarketBeta",
    "Momentum",
    "PCAResidualReversal",
    "PriceToMovingAverage",
    "RealizedVolatility",
    "RollSpread",
    "ShortTermReversal",
    "TimeSeriesMomentum",
]
