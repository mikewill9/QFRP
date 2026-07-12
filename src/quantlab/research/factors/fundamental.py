"""Fundamental factors: value, quality, and growth.

Every factor here follows the standard :class:`Factor` contract —
``compute(dataset) -> WideFrame`` of scores where higher = more attractive,
using nothing later than the close of each date — but consumes a
:class:`~quantlab.data.fundamentals.FundamentalDataset` supplied at
construction alongside the price dataset passed to ``compute``.

Point-in-time discipline
------------------------
All statement values enter through the fundamental dataset's
publication-date machinery: the value a factor sees at date ``t`` is from
the newest fiscal quarter *published* on or before ``t`` — never the
quarter that merely *ended* by ``t``. Trailing-twelve-month (TTM) and
growth constructions become usable at the latest publication date among
their constituent quarters. A factor value at ``t`` therefore changes only
when information published on or before ``t`` changes; the test suite pins
this with statements whose publication straddles the evaluation date.

Because statements arrive quarterly, these scores are step functions:
they jump on publication dates and are flat in between (except where the
price enters, e.g. yields moving daily with market cap). Signal turnover
is correspondingly far lower than for price factors.

Coverage semantics
------------------
Symbols in the price dataset without fundamental coverage score NaN (a
universe can legitimately be partially covered); zero overlap between the
two universes raises. ``min_history`` is 1 for every factor here: the
warmup is gated by statement availability and publication dates, not by
price-bar count, so early rows are NaN exactly until enough statements are
public.
"""

from __future__ import annotations

from quantlab.data.dataset import MarketDataset
from quantlab.data.fundamentals import (
    FundamentalDataset,
    combine_events,
    pit_align,
    trailing_change,
    trailing_stat,
)
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = [
    "Accruals",
    "BookToMarket",
    "EarningsGrowth",
    "EarningsYield",
    "EstimateRevision",
    "FCFYield",
    "FundamentalFactor",
    "GrossMarginStability",
    "ReturnOnEquity",
    "RevenueGrowth",
]


class FundamentalFactor(Factor):
    """Shared machinery for factors that consume fundamental statements.

    The fundamental dataset is stored privately so :attr:`Factor.params`
    keeps echoing only the small configuration knobs (windows, flags) that
    experiment tracking should record, not a data object.
    """

    min_history = 1

    def __init__(self, fundamentals: FundamentalDataset) -> None:
        if not isinstance(fundamentals, FundamentalDataset):
            raise DataError(
                f"{type(self).__name__} requires a FundamentalDataset, "
                f"got {type(fundamentals).__name__}"
            )
        self._fundamentals = fundamentals

    @property
    def fundamentals(self) -> FundamentalDataset:
        """The point-in-time statement panel this factor reads from."""
        return self._fundamentals

    def _check_overlap(self, dataset: MarketDataset) -> None:
        covered = set(self._fundamentals.symbols)
        if not any(sym in covered for sym in dataset.symbols):
            raise DataError(
                f"Factor '{self.name}': no overlap between price symbols "
                f"{dataset.symbols[:5]} and fundamental symbols "
                f"{self._fundamentals.symbols[:5]}"
            )

    def _pit(self, item: str, dataset: MarketDataset, *, ttm: bool = False) -> WideFrame:
        """Latest published value of ``item`` aligned to the price panel."""
        return self._fundamentals.pit_panel(
            item, dataset.index, symbols=dataset.symbols, ttm=ttm
        )

    def _market_cap(self, dataset: MarketDataset) -> WideFrame:
        """Close price times latest published shares outstanding.

        Uses point-in-time shares, so buybacks/issuance enter only once the
        balance sheet reporting them is public. NaN where shares are
        unknown.
        """
        return dataset.close * self._pit("shares_outstanding", dataset)


# --------------------------------------------------------------------- value
class EarningsYield(FundamentalFactor):
    """Trailing-twelve-month net income over market capitalization.

    The classic value ratio (inverse P/E) in yield form so that it is
    well-behaved through zero: loss-makers score negative, cheap earners
    score high. Moves daily with price between statement publications.
    """

    name = "earnings_yield"

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        earnings = self._pit("net_income", dataset, ttm=True)
        return earnings / self._market_cap(dataset)


class BookToMarket(FundamentalFactor):
    """Latest published book equity over market capitalization.

    High book-to-market = value. Firms with negative book equity score
    negative (deeply unattractive under the value hypothesis) rather than
    NaN; exclude them upstream with a universe filter if preferred.
    """

    name = "book_to_market"

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        equity = self._pit("total_equity", dataset)
        return equity / self._market_cap(dataset)


class FCFYield(FundamentalFactor):
    """Trailing-twelve-month free cash flow over market capitalization.

    Free cash flow is ``operating_cash_flow + capital_expenditure`` — capex
    is stored as reported (negative = outflow), so the sum subtracts
    investment. Both items must be present in a quarter for it to count
    toward the TTM window.
    """

    name = "fcf_yield"

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        f = self._fundamentals
        fcf_quarters = combine_events(
            f.quarterly("operating_cash_flow"),
            f.quarterly("capital_expenditure"),
            lambda ocf, capex: ocf + capex,
        )
        fcf_ttm = trailing_stat(fcf_quarters, quarters=4, stat="sum")
        fcf = pit_align(fcf_ttm, dataset.index, dataset.symbols)
        return fcf / self._market_cap(dataset)


# ------------------------------------------------------------------- quality
class ReturnOnEquity(FundamentalFactor):
    """Trailing-twelve-month net income over latest published book equity.

    Profitability quality: high-ROE names rank high. Firms with
    non-positive equity are NaN — dividing earnings by negative equity
    produces a sign-flipped nonsense ratio, so the honest score is
    "undefined" rather than a fabricated number.
    """

    name = "roe"

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        earnings = self._pit("net_income", dataset, ttm=True)
        equity = self._pit("total_equity", dataset)
        return earnings / equity.where(equity > 0)


class GrossMarginStability(FundamentalFactor):
    """Negative variability of the gross margin across trailing quarters.

    Stable-margin businesses (pricing power, low input-cost sensitivity)
    rank high: the score is minus the sample standard deviation of
    ``gross_profit / revenue`` over the trailing ``quarters`` fiscal
    quarters. Needs ``quarters`` consecutive statements — with vendors
    serving only ~5 quarters of history (Yahoo), the default 8-quarter
    window emits NaN until the cache accumulates more.
    """

    name = "gross_margin_stability"

    def __init__(self, fundamentals: FundamentalDataset, quarters: int = 8) -> None:
        super().__init__(fundamentals)
        if quarters < 3:
            raise DataError("quarters must be >= 3 for a meaningful stability estimate")
        self.quarters = quarters

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        f = self._fundamentals
        margins = combine_events(
            f.quarterly("gross_profit"),
            f.quarterly("revenue"),
            lambda gp, rev: gp / rev.where(rev != 0),
        )
        stability = trailing_stat(margins, quarters=self.quarters, stat="std")
        return -pit_align(stability, dataset.index, dataset.symbols)


class Accruals(FundamentalFactor):
    """Negative accrual intensity (Sloan 1996 style).

    Accruals are the gap between reported earnings and cash actually
    generated: TTM ``net_income - operating_cash_flow``, scaled by the
    latest published total assets. High accruals historically predict
    earnings reversals, so the emitted score is the **negative** of the
    ratio — low-accrual (cash-backed earnings) names rank high. Scaling
    uses latest point-in-time assets rather than the academic average of
    beginning/ending assets; the simplification is deliberate and small.
    """

    name = "accruals"

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        f = self._fundamentals
        accrual_quarters = combine_events(
            f.quarterly("net_income"),
            f.quarterly("operating_cash_flow"),
            lambda ni, ocf: ni - ocf,
        )
        accruals_ttm = trailing_stat(accrual_quarters, quarters=4, stat="sum")
        accruals = pit_align(accruals_ttm, dataset.index, dataset.symbols)
        assets = self._pit("total_assets", dataset)
        return -accruals / assets.where(assets > 0)


# -------------------------------------------------------------------- growth
class RevenueGrowth(FundamentalFactor):
    """Year-over-year revenue growth.

    With ``ttm=True`` (default) growth compares trailing-twelve-month
    revenue against the TTM one year earlier — smooth and
    seasonality-free, but needing 8 consecutive quarters of statements.
    ``ttm=False`` compares single quarters against the same quarter one
    year back (5 quarters needed), the right choice on shallow vendor
    history. The denominator enters in absolute value so the growth sign
    stays meaningful for negative bases; zero bases are NaN.
    """

    name = "revenue_growth"

    _item = "revenue"

    def __init__(self, fundamentals: FundamentalDataset, ttm: bool = True) -> None:
        super().__init__(fundamentals)
        self.ttm = bool(ttm)

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        self._check_overlap(dataset)
        f = self._fundamentals
        base = f.ttm(self._item) if self.ttm else f.quarterly(self._item)
        growth = trailing_change(base, quarters=4)
        return pit_align(growth, dataset.index, dataset.symbols)


class EarningsGrowth(RevenueGrowth):
    """Year-over-year net income growth.

    Same construction as :class:`RevenueGrowth` on ``net_income``. Earnings
    cross zero routinely, which is why the change is scaled by the
    **absolute** prior value: a swing from -10 to +10 scores +2.0 (strong
    improvement), from +10 to -10 scores -2.0. Bases of exactly zero are
    NaN.
    """

    name = "earnings_growth"

    _item = "net_income"


# ------------------------------------------------------------------ interface
class EstimateRevision(Factor):
    """Analyst estimate-revision factor — interface awaiting real data.

    Hypothesis: upgrades in consensus estimates continue to drift prices
    for weeks (post-revision drift), so names whose consensus rose over the
    trailing ``window`` bars rank high.

    **This factor cannot be backtested with yfinance data.** Yahoo exposes
    only the *current* snapshot of analyst estimates, and a backtest needs
    what the consensus was on every historical date. Feeding today's
    estimates into the past is lookahead, so this class raises
    :class:`~quantlab.exceptions.DataError` unless it is given a genuinely
    point-in-time estimates panel. It activates unchanged when such a
    source arrives.

    Expected data shape: a :class:`FundamentalDataset` holding one row per
    (symbol, snapshot date) with ``period_end == publication_date ==``
    snapshot date and an ``estimate_item`` column carrying the vendor's
    *rolled* consensus (e.g. next-twelve-months mean EPS). Rolling the
    fiscal-period targets into a continuous series is the vendor adapter's
    job — comparing raw next-quarter consensus across a quarter roll would
    mix two different fiscal periods.

    Parameters
    ----------
    estimates:
        Point-in-time estimates panel, or None (raises on compute).
    estimate_item:
        Item column holding the rolled consensus level.
    window:
        Revision horizon in price bars (default ~3 months).
    """

    name = "estimate_revision"

    min_history = 1

    def __init__(
        self,
        estimates: FundamentalDataset | None = None,
        estimate_item: str = "eps_estimate_mean",
        window: int = 63,
    ) -> None:
        if window < 1:
            raise DataError("window must be >= 1")
        self._estimates = estimates
        self.estimate_item = estimate_item
        self.window = window

    def compute(self, dataset: MarketDataset) -> WideFrame:
        if self._estimates is None or self.estimate_item not in self._estimates.items:
            raise DataError(
                "EstimateRevision requires a point-in-time analyst-estimates panel "
                f"with item '{self.estimate_item}'. yfinance provides only the "
                "current snapshot of estimates, which cannot be used historically "
                "without lookahead; connect a point-in-time estimates vendor to "
                "activate this factor."
            )
        self._require_history(dataset)
        covered = set(self._estimates.symbols)
        if not any(sym in covered for sym in dataset.symbols):
            raise DataError(
                f"Factor '{self.name}': no overlap between price symbols and "
                "estimate symbols"
            )
        consensus = self._estimates.pit_panel(
            self.estimate_item, dataset.index, symbols=dataset.symbols
        )
        prior = consensus.shift(self.window)
        return (consensus - prior) / prior.abs().where(prior.abs() > 0)
