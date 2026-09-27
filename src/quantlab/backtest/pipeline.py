"""The shared single-factor long/short backtest convention.

Extracted from ``notebooks/factor_research.ipynb`` so the notebook and
scripts (e.g. ``scripts/run_permutations.py``) run the *same* code rather
than drifting copies. Contents:

- the universe and portfolio/cost constants the notebook used;
- ``PRICE_FACTOR_BUILDERS``: the price-factor catalog (fundamental factors
  stay in the notebook — they need a ``FundamentalDataset`` and have no
  real history before ~2025 on free data);
- ``normalization()``: the per-date winsorize + z-score pipeline;
- ``run_ls_backtest``: monthly top/bottom-quintile, equal weight per side,
  costed (volume-impact slippage, per-share commission, 5% participation
  cap), decisions at the close and fills at the next open.

Lives in ``backtest`` (not ``research``) because of the layering rule in
docs/architecture.md: ``backtest`` may import ``research`` and ``data``,
never the reverse.
"""

from __future__ import annotations

from collections.abc import Callable

from quantlab.backtest.costs import (
    PerShareCommission,
    VolumeImpactSlippage,
    ZeroCommission,
    ZeroSlippage,
)
from quantlab.backtest.engine import BacktestEngine
from quantlab.backtest.execution import ExecutionSimulator
from quantlab.backtest.results import BacktestResult
from quantlab.backtest.strategy import TargetWeightStrategy, signal_to_weights
from quantlab.data.dataset import MarketDataset
from quantlab.data.universe import US_LARGE_CAP_150
from quantlab.exceptions import BacktestError
from quantlab.research import (
    AmihudIlliquidity,
    DollarVolume,
    EWMAVolatility,
    Factor,
    MarketBeta,
    Momentum,
    PCAResidualReversal,
    PriceToMovingAverage,
    RealizedVolatility,
    RollSpread,
    ShortTermReversal,
    SignalPipeline,
    TimeSeriesMomentum,
    Winsorize,
    ZScore,
)
from quantlab.types import WideFrame

__all__ = [
    "COMMISSION_MINIMUM",
    "COMMISSION_RATE",
    "DATA_START",
    "GROSS",
    "IMPACT_COEFF",
    "INITIAL_CAPITAL",
    "LONG_FRAC",
    "MAX_PARTICIPATION",
    "PRICE_FACTOR_BUILDERS",
    "REBALANCE_EVERY",
    "SHORT_FRAC",
    "SPREAD_BPS",
    "TICKERS",
    "TOLERANCE",
    "make_execution",
    "normalization",
    "run_ls_backtest",
]

# --- universe ---------------------------------------------------------------
# MMC: Yahoo data issue. EA: delisted; Yahoo no longer serves its history.
# Survivors only -- every result on this list carries the survivorship caveat.
TICKERS: list[str] = [t for t in US_LARGE_CAP_150 if t not in ("MMC", "EA")]

# Prices are fetched from here regardless of the backtest start, so every
# factor is fully warmed up (12-1 momentum needs 273 bars) on day one of
# any backtest window from 2005 on. Factors are computed on the full history
# and then trimmed to the window, so the same date gets the same score no
# matter which window is being tested.
DATA_START = "2004-01-01"

# --- portfolio --------------------------------------------------------------
REBALANCE_EVERY = 21  # trade monthly
LONG_FRAC, SHORT_FRAC, GROSS = 0.2, 0.2, 1.0
INITIAL_CAPITAL = 1_000_000
TOLERANCE = 0.001  # skip trades smaller than 0.1% of equity

# --- costs (the 1x setting) ---------------------------------------------------
SPREAD_BPS = 5.0  # full quoted spread; half paid per fill
IMPACT_COEFF = 0.1  # square-root market impact scale
COMMISSION_RATE = 0.005  # $ per share
COMMISSION_MINIMUM = 1.0  # $ per order
MAX_PARTICIPATION = 0.05  # never more than 5% of a bar's volume

# --- price-factor catalog -----------------------------------------------------
PRICE_FACTOR_BUILDERS: dict[str, Callable[[], Factor]] = {
    "momentum": lambda: Momentum(lookback=252, skip=21),
    "ts_momentum": lambda: TimeSeriesMomentum(lookback=126),
    "st_reversal": lambda: ShortTermReversal(lookback=5),
    "price_to_ma": lambda: PriceToMovingAverage(window=21),
    "realized_vol": lambda: RealizedVolatility(window=63),
    "ewma_vol": lambda: EWMAVolatility(halflife=32.0),
    "market_beta": lambda: MarketBeta(window=63),
    "pca_reversal": lambda: PCAResidualReversal(),
    "amihud_illiquidity": lambda: AmihudIlliquidity(window=21),
    "dollar_volume": lambda: DollarVolume(window=21),
    "roll_spread": lambda: RollSpread(window=21),
}


def normalization() -> SignalPipeline:
    """Per-date winsorize (1%/99%) then z-score: comparable cross-sectional scores."""
    return SignalPipeline([Winsorize(0.01, 0.99), ZScore()])


def make_execution(
    cost_multiplier: float = 1.0, *, frictionless: bool = False
) -> ExecutionSimulator:
    """Execution simulator at ``cost_multiplier`` times the 1x cost setting.

    Scales the spread and the commission (per-share rate and per-order
    minimum). The market-impact coefficient and participation cap are left
    unchanged: impact is a model of the stock's liquidity, not a fee. So
    ``cost_multiplier=0`` still charges impact.

    ``frictionless=True`` removes every cost -- spread, commission AND
    impact -- for gross (pre-cost) returns. The participation cap is kept:
    it limits what can be traded, it doesn't charge for trading.
    """
    if frictionless:
        if cost_multiplier != 1.0:
            raise BacktestError("pass either frictionless=True or a cost_multiplier, not both")
        return ExecutionSimulator(
            slippage=ZeroSlippage(),
            commission=ZeroCommission(),
            max_participation=MAX_PARTICIPATION,
        )
    if cost_multiplier < 0:
        raise BacktestError("cost_multiplier must be >= 0")
    return ExecutionSimulator(
        slippage=VolumeImpactSlippage(
            spread_bps=SPREAD_BPS * cost_multiplier, impact_coeff=IMPACT_COEFF
        ),
        commission=PerShareCommission(
            rate=COMMISSION_RATE * cost_multiplier,
            minimum=COMMISSION_MINIMUM * cost_multiplier,
        ),
        max_participation=MAX_PARTICIPATION,
    )


def run_ls_backtest(
    signal_frame: WideFrame,
    ds: MarketDataset,
    *,
    cost_multiplier: float = 1.0,
    frictionless: bool = False,
) -> BacktestResult:
    """Long/short backtest of ``signal_frame`` on ``ds``.

    Costed at ``cost_multiplier`` x the 1x setting, or cost-free with
    ``frictionless=True`` (see :func:`make_execution`).
    """
    weights = signal_to_weights(
        signal_frame,
        long_frac=LONG_FRAC,
        short_frac=SHORT_FRAC,
        gross=GROSS,
        rebalance_every=REBALANCE_EVERY,
    )
    strategy = TargetWeightStrategy(weights=weights, tolerance=TOLERANCE)
    return BacktestEngine(
        ds,
        strategy,
        make_execution(cost_multiplier, frictionless=frictionless),
        initial_capital=INITIAL_CAPITAL,
    ).run()
