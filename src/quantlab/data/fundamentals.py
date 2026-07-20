"""Point-in-time fundamental data.

Why this module is shaped differently from price data
------------------------------------------------------
A price bar is usable the moment it prints. A quarterly report is not: a
statement for the quarter ending March 31 was typically *filed* weeks later,
and until that filing date the numbers simply did not exist for anyone
trading. Aligning fundamentals on ``period_end`` alone therefore leaks the
future into a backtest. To keep the framework's no-lookahead guarantee
structural, every fundamental observation here carries **two** dates:

- ``period_end``: the fiscal period the numbers describe;
- ``publication_date``: the first date the observation was usable.

All alignment against price indexes goes through :func:`pit_align`, which
only ever reveals a statement on or after its ``publication_date``.

Publication-lag assumption (read this)
--------------------------------------
Free sources (yfinance) do **not** provide true filing dates, and they
silently restate history — you always see today's view of past statements.
:class:`YahooFundamentalSource` therefore *assumes* a configurable
publication lag (default **45 calendar days** after period end) and stamps
``metadata["publication_basis"] = "assumed_lag"``. 45 days covers the SEC
10-Q deadlines (40-45 days), so interim quarters are treated
conservatively; fiscal fourth quarters arrive via the 10-K (60-90 day
deadlines), for which 45 days is optimistic — raise the lag for stricter
tests. A point-in-time vendor with real filing dates sets
``publication_basis = "reported"`` and everything downstream is unchanged.

Data shapes
-----------
Statement frame (the container's storage)
    Long frame, one row per ``(symbol, period_end, publication_date)`` with
    one column per item. Multiple rows for the same ``(symbol, period_end)``
    are allowed and mean *restatements*: later ``publication_date`` = newer
    version of the same fiscal quarter.

Event frame (the alignment currency)
    Long frame with exactly the columns
    ``["symbol", "period_end", "publication_date", "value"]`` — one item's
    observations, ready for :func:`pit_align` / :func:`trailing_stat` /
    :func:`trailing_change`.

Known limitations (documented, not hidden)
------------------------------------------
- yfinance returns only the trailing ~4-6 quarters of statements. Long
  fundamental backtests on free data are impossible today; the cache
  *accumulates* quarters across fetches over time, which is the honest way
  to build point-in-time history from a snapshot vendor going forward.
- Restatements are invisible in a single yfinance snapshot; the container
  supports multi-version rows so a better vendor slots in unchanged.
- Trailing-twelve-month values are gated by the *latest* publication date
  among their constituent quarters. When restatement history exists this is
  slightly conservative (the pre-restatement TTM was usable earlier), never
  lookahead.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.cache import CachedSource
from quantlab.data.sources.base import normalize_date, validate_request
from quantlab.exceptions import CacheError, DataError
from quantlab.log import get_logger
from quantlab.types import LongFrame, WideFrame

__all__ = [
    "FUNDAMENTAL_ID_COLUMNS",
    "FUNDAMENTAL_ITEMS",
    "CachedFundamentalSource",
    "FundamentalDataset",
    "FundamentalSource",
    "SyntheticFundamentalSource",
    "YahooFundamentalSource",
    "combine_events",
    "pit_align",
    "trailing_change",
    "trailing_stat",
]

logger = get_logger(__name__)

#: Identity columns every statement frame must carry.
FUNDAMENTAL_ID_COLUMNS: tuple[str, ...] = ("symbol", "period_end", "publication_date")

#: Canonical quarterly line items. Sources map vendor names onto these; the
#: container itself accepts any numeric item columns (e.g. estimate items),
#: so this tuple is a vocabulary, not a constraint. Sign conventions:
#: ``capital_expenditure`` is stored **as reported** in the cash-flow
#: statement (negative = cash outflow), so free cash flow is
#: ``operating_cash_flow + capital_expenditure``.
FUNDAMENTAL_ITEMS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "shares_outstanding",
    "operating_cash_flow",
    "capital_expenditure",
)

#: Column layout of an event frame (one item's observations).
EVENT_COLUMNS: tuple[str, ...] = ("symbol", "period_end", "publication_date", "value")

# Calendar-quarter spacing tolerance, in days per quarter step. A window of
# k quarter steps must span [k * _MIN_QTR_DAYS, k * _MAX_QTR_DAYS] days or
# the derived value is dropped (a silently longer window would compare the
# wrong fiscal periods, e.g. across a missing quarter).
_MIN_QTR_DAYS = 80
_MAX_QTR_DAYS = 100


def _empty_events() -> LongFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "publication_date": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype=float),
        }
    )


def _latest_versions(events: LongFrame) -> LongFrame:
    """Collapse restatement versions to each quarter's latest publication."""
    if events.empty:
        return events
    out = events.sort_values(["symbol", "period_end", "publication_date"])
    return out.drop_duplicates(["symbol", "period_end"], keep="last")


def _check_datetime_column(frame: pd.DataFrame, col: str) -> pd.Series:
    series = frame[col]
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise DataError(f"Column '{col}' must be tz-naive; strip timezones in the source")
    try:
        parsed = pd.to_datetime(series)
    except (ValueError, TypeError) as exc:
        raise DataError(f"Column '{col}' is not parseable as datetimes: {exc}") from exc
    if len(parsed) == 0:
        return parsed.astype("datetime64[ns]")
    if parsed.dtype == object or isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise DataError(f"Column '{col}' must be tz-naive homogeneous datetimes")
    return parsed


# ---------------------------------------------------------------- container
@dataclass(frozen=True, slots=True)
class FundamentalDataset:
    """Point-in-time panel of fundamental statements.

    Parameters
    ----------
    statements:
        Long frame with the :data:`FUNDAMENTAL_ID_COLUMNS` plus one numeric
        column per item. One row per ``(symbol, period_end,
        publication_date)``; several rows sharing ``(symbol, period_end)``
        are restatement versions, newest publication = current view.
    metadata:
        Provenance. Conventional keys: ``"source"``,
        ``"publication_basis"`` (``"reported"`` or ``"assumed_lag"``),
        ``"publication_lag_days"``.

    Construction normalizes (parses dates, casts symbols to ``str``, sorts
    rows) and validates structure: ``publication_date < period_end`` is
    impossible for a filed statement and raises, as do duplicate rows,
    tz-aware dates, and non-numeric item columns.
    """

    statements: LongFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stmts = self.statements
        if not isinstance(stmts, pd.DataFrame):
            raise DataError("statements must be a DataFrame")
        missing = [c for c in FUNDAMENTAL_ID_COLUMNS if c not in stmts.columns]
        if missing:
            raise DataError(f"statements missing required columns {missing}")
        item_cols = [c for c in stmts.columns if c not in FUNDAMENTAL_ID_COLUMNS]
        if not item_cols:
            raise DataError("statements contain no item columns")

        work = stmts.copy()
        work["symbol"] = work["symbol"].astype(str)
        for col in ("period_end", "publication_date"):
            work[col] = _check_datetime_column(work, col)
        for col in item_cols:
            try:
                work[col] = pd.to_numeric(work[col])
            except (ValueError, TypeError) as exc:
                raise DataError(f"Item column '{col}' is not numeric: {exc}") from exc

        if len(work):
            ids = work[list(FUNDAMENTAL_ID_COLUMNS)]
            if ids.isna().any().any():
                raise DataError("statements have missing symbol/period_end/publication_date")
            early = work["publication_date"] < work["period_end"]
            if early.any():
                bad = work.loc[early, ["symbol", "period_end", "publication_date"]].head(3)
                raise DataError(
                    "publication_date precedes period_end (a statement cannot be "
                    f"public before its quarter ends), e.g.:\n{bad.to_string(index=False)}"
                )
            dupes = work.duplicated(list(FUNDAMENTAL_ID_COLUMNS))
            if dupes.any():
                bad = work.loc[dupes, list(FUNDAMENTAL_ID_COLUMNS)].head(3)
                raise DataError(
                    f"Duplicate (symbol, period_end, publication_date) rows, e.g.:\n"
                    f"{bad.to_string(index=False)}"
                )

        work = work.sort_values(["symbol", "period_end", "publication_date"])
        work = work[list(FUNDAMENTAL_ID_COLUMNS) + item_cols].reset_index(drop=True)
        object.__setattr__(self, "statements", work)
        object.__setattr__(self, "metadata", dict(self.metadata))

    # ------------------------------------------------------------ properties
    @property
    def symbols(self) -> list[str]:
        """Symbols with at least one statement, sorted."""
        return sorted(self.statements["symbol"].unique())

    @property
    def items(self) -> list[str]:
        """Item columns, in stored order."""
        return [c for c in self.statements.columns if c not in FUNDAMENTAL_ID_COLUMNS]

    @property
    def n_statements(self) -> int:
        return len(self.statements)

    @property
    def start(self) -> pd.Timestamp:
        """Earliest period end (NaT when empty)."""
        return self.statements["period_end"].min()

    @property
    def end(self) -> pd.Timestamp:
        """Latest period end (NaT when empty)."""
        return self.statements["period_end"].max()

    # ------------------------------------------------------------- selection
    def select(self, symbols: Sequence[str]) -> FundamentalDataset:
        """Restrict to ``symbols``; unknown symbols raise."""
        have = set(self.statements["symbol"])
        missing = [s for s in symbols if s not in have]
        if missing:
            raise DataError(f"Symbols not in fundamental dataset: {missing[:10]}")
        kept = self.statements[self.statements["symbol"].isin(set(symbols))]
        return FundamentalDataset(kept, metadata=dict(self.metadata))

    def as_of(self, date: pd.Timestamp | str) -> FundamentalDataset:
        """Everything usable on ``date``: rows with ``publication_date <= date``.

        A statement published *on* ``date`` is usable *at* ``date`` — it is
        public information by that day's close, matching the framework's
        signal-at-``t`` convention.
        """
        ts = normalize_date(date)
        kept = self.statements[self.statements["publication_date"] <= ts]
        return FundamentalDataset(
            kept, metadata={**self.metadata, "as_of": str(ts.date())}
        )

    # ------------------------------------------------------------ derivation
    def quarterly(self, item: str) -> LongFrame:
        """One item's per-quarter history as an event frame.

        **All statement versions are kept**: when a quarter was restated,
        the original value is the answer between the two publication dates
        and the restated value afterwards — :func:`pit_align` resolves this
        per date. Rows whose value is NaN are dropped, so an event frame
        contains only observations that actually carry a value for this
        item.
        """
        if item not in self.items:
            raise DataError(f"Unknown fundamental item '{item}'; have {self.items}")
        cols = [*FUNDAMENTAL_ID_COLUMNS, item]
        events = self.statements[cols].rename(columns={item: "value"})
        events = events.dropna(subset=["value"])
        events = events.sort_values(["symbol", "period_end", "publication_date"])
        return events.reset_index(drop=True)

    def ttm(self, item: str, quarters: int = 4) -> LongFrame:
        """Trailing sum of ``item`` over ``quarters`` fiscal quarters.

        The result becomes usable only at the **latest** publication date of
        its constituent quarters, and is NaN-dropped where the window spans
        an irregular calendar range (missing quarters). Only meaningful for
        flow items (income and cash-flow statement lines); a trailing sum of
        a balance-sheet stock is nonsense.
        """
        return trailing_stat(self.quarterly(item), quarters=quarters, stat="sum")

    def pit_panel(
        self,
        item: str,
        index: pd.DatetimeIndex,
        symbols: Sequence[str] | None = None,
        *,
        ttm: bool = False,
    ) -> WideFrame:
        """Wide frame of the latest *published* value of ``item`` per date.

        The value at ``(t, symbol)`` comes from the newest fiscal quarter
        whose (latest) publication date is on or before ``t`` — exactly what
        a researcher standing at ``t`` could have known. Dates before the
        first publication are NaN, never backfilled.
        """
        events = self.ttm(item) if ttm else self.quarterly(item)
        cols = list(symbols) if symbols is not None else self.symbols
        return pit_align(events, index, cols)

    def coverage_table(
        self,
        item: str,
        index: pd.DatetimeIndex,
        symbols: Sequence[str] | None = None,
        *,
        ttm: bool = False,
    ) -> pd.DataFrame:
        """Per-date coverage of ``item`` across a universe, for reporting.

        One row per date of ``index`` with columns ``n_covered`` (symbols
        whose latest-published value of ``item`` is available at that
        date), ``n_universe`` (size of ``symbols``, or this dataset's
        symbols when omitted), and ``pct_covered`` (their ratio, 0-100).

        Use this to make sparse fundamental data visible instead of
        silently dropped: panels drop rows with any missing feature, so a
        universe that is 20% covered produces a panel a fifth the size —
        report the coverage next to any result that includes fundamental
        factors.
        """
        panel = self.pit_panel(item, index, symbols, ttm=ttm)
        n_universe = panel.shape[1]
        if n_universe == 0:
            raise DataError("coverage_table needs at least one symbol")
        n_covered = panel.notna().sum(axis=1)
        return pd.DataFrame(
            {
                "n_covered": n_covered.astype(int),
                "n_universe": n_universe,
                "pct_covered": 100.0 * n_covered / n_universe,
            },
            index=index,
        )

    # -------------------------------------------------------------- plumbing
    def with_metadata(self, **updates: Any) -> FundamentalDataset:
        """Return a copy with metadata keys merged in."""
        return FundamentalDataset(self.statements, metadata={**self.metadata, **updates})

    def equals(self, other: FundamentalDataset) -> bool:
        """Statement-frame equality (NaN == NaN), ignoring metadata."""
        return self.statements.equals(other.statements)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        span = (
            "empty" if self.n_statements == 0 else f"{self.start.date()} .. {self.end.date()}"
        )
        return (
            f"FundamentalDataset({len(self.symbols)} symbols, "
            f"{self.n_statements} statements, {len(self.items)} items, {span})"
        )


# ------------------------------------------------------- alignment machinery
def pit_align(
    events: LongFrame, index: pd.DatetimeIndex, symbols: Sequence[str]
) -> WideFrame:
    """Project an event frame onto a price index without lookahead.

    For each date ``t`` and symbol, the emitted value belongs to the row
    with the greatest ``period_end`` among rows published on or before
    ``t`` (publication ties and restatements resolve to the latest
    publication). A restated *old* quarter never displaces a newer quarter
    that is already public. Dates before the first publication are NaN.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise DataError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise DataError("index must be tz-naive")
    cols = [str(s) for s in symbols]
    out_empty = pd.DataFrame(np.nan, index=index, columns=cols)
    if events.empty or not cols:
        return out_empty
    ev = events[events["symbol"].isin(set(cols))]
    if ev.empty:
        return out_empty
    ev = ev.sort_values(["symbol", "publication_date", "period_end"])
    # A publication only changes what is known if its period_end is at least
    # as new as everything already published for that symbol.
    newest = ev.groupby("symbol")["period_end"].cummax()
    ev = ev[ev["period_end"].eq(newest)]
    ev = ev.drop_duplicates(["symbol", "publication_date"], keep="last")
    wide = ev.pivot(index="publication_date", columns="symbol", values="value")
    union = index.union(pd.DatetimeIndex(wide.index))
    aligned = wide.reindex(union).ffill().reindex(index)
    return aligned.reindex(columns=cols)


def trailing_stat(
    events: LongFrame, quarters: int = 4, stat: str = "sum"
) -> LongFrame:
    """Rolling per-symbol statistic over consecutive fiscal quarters.

    ``stat="sum"`` gives trailing totals (TTM with ``quarters=4``);
    ``stat="std"`` gives trailing variability (sample std, ddof=1). Each
    output row inherits the window's last ``period_end`` and becomes usable
    at the **maximum** publication date inside the window. Windows whose
    period ends do not span ``quarters - 1`` roughly-quarterly steps are
    dropped: a gap in the statement history must surface as a missing value,
    not as a silently mis-sized window.

    Restatement versions collapse to each quarter's latest publication
    before rolling — a multi-quarter window built from mixed vintages is
    not meaningful. Combined with max-publication gating this is slightly
    conservative (the pre-restatement window was usable earlier), never
    lookahead.
    """
    if quarters < 2:
        raise DataError("quarters must be >= 2")
    if stat not in ("sum", "std"):
        raise DataError(f"Unknown stat '{stat}' (use 'sum' or 'std')")
    events = _latest_versions(events)
    if events.empty:
        return _empty_events()
    lo, hi = _MIN_QTR_DAYS * (quarters - 1), _MAX_QTR_DAYS * (quarters - 1)
    parts: list[pd.DataFrame] = []
    for _, group in events.groupby("symbol", sort=False):
        g = group.sort_values("period_end")
        roll = g["value"].rolling(quarters)
        value = roll.sum() if stat == "sum" else roll.std(ddof=1)
        pubs = g["publication_date"].to_numpy()
        window_pub = np.array(
            [
                pubs[i - quarters + 1 : i + 1].max()
                if i >= quarters - 1
                else np.datetime64("NaT", "ns")
                for i in range(len(g))
            ],
            dtype="datetime64[ns]",
        )
        span = (g["period_end"] - g["period_end"].shift(quarters - 1)).dt.days
        keep = span.between(lo, hi) & value.notna()
        parts.append(
            pd.DataFrame(
                {
                    "symbol": g["symbol"],
                    "period_end": g["period_end"],
                    "publication_date": window_pub,
                    "value": value,
                }
            ).loc[keep]
        )
    if not parts:
        return _empty_events()
    out = pd.concat(parts, ignore_index=True)
    return out if len(out) else _empty_events()


def trailing_change(events: LongFrame, quarters: int = 4) -> LongFrame:
    """Per-symbol relative change versus ``quarters`` fiscal quarters ago.

    ``value = (x_t - x_{t-q}) / |x_{t-q}|`` — the absolute-value denominator
    keeps the sign meaningful when the base is negative (e.g. earnings
    recovering from a loss); a zero base yields NaN. Usable at the later of
    the two constituent publication dates; windows spanning an irregular
    calendar range (missing quarters) are dropped. Restatement versions
    collapse to each quarter's latest publication first (see
    :func:`trailing_stat`).
    """
    if quarters < 1:
        raise DataError("quarters must be >= 1")
    events = _latest_versions(events)
    if events.empty:
        return _empty_events()
    lo, hi = _MIN_QTR_DAYS * quarters, _MAX_QTR_DAYS * quarters
    parts: list[pd.DataFrame] = []
    for _, group in events.groupby("symbol", sort=False):
        g = group.sort_values("period_end")
        prev = g["value"].shift(quarters)
        denom = prev.abs()
        value = (g["value"] - prev) / denom.where(denom > 0)
        pub = np.maximum(
            g["publication_date"].to_numpy(),
            g["publication_date"].shift(quarters).to_numpy(),
        )
        span = (g["period_end"] - g["period_end"].shift(quarters)).dt.days
        keep = span.between(lo, hi) & value.notna()
        parts.append(
            pd.DataFrame(
                {
                    "symbol": g["symbol"],
                    "period_end": g["period_end"],
                    "publication_date": pub,
                    "value": value,
                }
            ).loc[keep]
        )
    out = pd.concat(parts, ignore_index=True)
    return out if len(out) else _empty_events()


def combine_events(
    left: LongFrame,
    right: LongFrame,
    func: Callable[[pd.Series, pd.Series], pd.Series],
) -> LongFrame:
    """Combine two event frames row-wise on ``(symbol, period_end)``.

    Only quarters present in **both** frames survive (inner join); the
    result is usable at the later of the two publication dates. Rows where
    ``func`` yields NaN (e.g. a ratio with a zero denominator) are dropped.
    Restatement versions collapse to each quarter's latest publication
    first — combining mixed vintages of the same quarter is not meaningful.
    """
    merged = _latest_versions(left).merge(
        _latest_versions(right), on=["symbol", "period_end"], how="inner", suffixes=("_l", "_r")
    )
    if merged.empty:
        return _empty_events()
    out = pd.DataFrame(
        {
            "symbol": merged["symbol"],
            "period_end": merged["period_end"],
            "publication_date": np.maximum(
                merged["publication_date_l"].to_numpy(),
                merged["publication_date_r"].to_numpy(),
            ),
            "value": func(merged["value_l"], merged["value_r"]),
        }
    ).dropna(subset=["value"])
    out = out.sort_values(["symbol", "period_end"]).reset_index(drop=True)
    return out if len(out) else _empty_events()


# ------------------------------------------------------------------- sources
class FundamentalSource(ABC):
    """Abstract quarterly-fundamentals source.

    ``fetch`` returns statements whose ``period_end`` lies in
    ``[start, end]`` inclusive. Because trailing-twelve-month and growth
    factors need history *before* their first evaluation date, callers
    should request fundamentals starting 6-8 quarters before the first
    backtest date. Sources must stamp ``metadata["publication_basis"]``:
    ``"reported"`` for true filing dates, ``"assumed_lag"`` when publication
    dates are period end plus an assumed lag.
    """

    #: Stable identifier; keys the cache directory.
    name: str = "fundamental_base"

    @abstractmethod
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> FundamentalDataset:
        """Fetch quarterly statements for ``symbols`` over ``[start, end]``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


_FUND_ASSET_STREAM = 3_000
_FUND_QUARTER_STREAM = 4_000_000_000


@dataclass(frozen=True, slots=True)
class _FundamentalParams:
    """Per-asset structural parameters (drawn once from the seed)."""

    revenue0: np.ndarray  # initial quarterly revenue
    growth_annual: np.ndarray  # annual revenue log-drift
    revenue_vol: np.ndarray  # quarterly revenue shock sd
    gross_margin: np.ndarray  # base gross margin
    margin_vol: np.ndarray  # gross-margin shock sd (stability dispersion)
    opex_ratio: np.ndarray  # opex as a fraction of base gross profit
    retention: np.ndarray  # earnings retained into equity
    leverage: np.ndarray  # liabilities scale relative to equity0
    equity_mult: np.ndarray  # equity0 as a multiple of quarterly revenue
    accrual_mu: np.ndarray  # mean accrual intensity (fraction of revenue)
    capex_frac: np.ndarray  # capex as a fraction of revenue
    shares: np.ndarray  # constant shares outstanding


class SyntheticFundamentalSource(FundamentalSource):
    """Deterministic synthetic quarterly statements.

    Companion to :class:`~quantlab.data.sources.synthetic.SyntheticSource`:
    same ``SYN000..`` symbol naming, so with matching ``n_assets`` the two
    universes line up. Statements are internally consistent by construction:

    - ``total_assets = total_liabilities + total_equity`` exactly;
    - equity evolves by retained earnings
      (``equity_q = equity_{q-1} + retention * net_income_q``);
    - ``gross_profit = margin * revenue`` with ``margin`` in (0.05, 0.95);
    - ``operating_cash_flow = net_income - accruals`` with accruals a small
      per-asset fraction of revenue, so the accruals factor has real
      cross-sectional dispersion at income-statement scale;
    - ``capital_expenditure`` is negative (as-reported convention).

    Fundamentals are drawn **independently of synthetic prices**, so the
    statements themselves carry no return signal: factors built purely from
    statement values (e.g. an earnings *level*) have ICs statistically at
    zero. Price-scaled ratios (earnings yield, book-to-market) inherit a
    mildly **negative** IC on this market — the price in their denominator
    makes them anti-momentum, and the synthetic market's permanent
    per-asset drift makes momentum persistent. That is an artifact of the
    price model (mirroring the real value/momentum anti-correlation), not
    alpha. A strongly profitable fundamental backtest on synthetic data
    indicates a bug, not a discovery.

    Determinism: per-asset parameters come from ``(seed, asset stream)`` and
    per-quarter shocks from ``(seed, quarter ordinal)``, so any two fetches
    agree exactly on overlapping quarters regardless of the requested
    window or symbol subset (same contract as the price source).

    Publication dates are ``period_end + publication_lag_days`` plus a
    deterministic per-(asset, quarter) jitter of up to
    ``publication_jitter_days``.
    """

    name = "synthetic_fundamentals"

    def __init__(
        self,
        n_assets: int = 50,
        seed: int = 7,
        inception: pd.Timestamp | str = "2015-01-02",
        publication_lag_days: int = 45,
        publication_jitter_days: int = 10,
    ) -> None:
        if n_assets < 1:
            raise DataError("n_assets must be >= 1")
        if publication_lag_days < 0:
            raise DataError("publication_lag_days must be >= 0")
        if publication_jitter_days < 0:
            raise DataError("publication_jitter_days must be >= 0")
        self.n_assets = int(n_assets)
        self.seed = int(seed)
        self.inception = normalize_date(inception)
        self.publication_lag_days = int(publication_lag_days)
        self.publication_jitter_days = int(publication_jitter_days)
        self._params = self._draw_params()

    @property
    def symbols(self) -> list[str]:
        """All symbols in the universe (matches ``SyntheticSource`` naming)."""
        return [f"SYN{i:03d}" for i in range(self.n_assets)]

    # -------------------------------------------------------------- internals
    def _draw_params(self) -> _FundamentalParams:
        """Draw per-asset parameters in a fixed, documented order.

        Revenue scales with shares outstanding so per-share economics land
        in plausible bands (quarterly revenue per share around a few
        dollars), keeping ratios like earnings yield and book-to-market in
        realistic magnitudes even though fundamentals stay independent of
        the synthetic *price* paths.
        """
        n = self.n_assets
        rng = np.random.default_rng([self.seed, _FUND_ASSET_STREAM])
        shares = np.exp(rng.normal(17.5, 1.0, size=n))
        return _FundamentalParams(
            shares=shares,
            revenue0=shares * np.exp(rng.normal(0.8, 0.6, size=n)),
            growth_annual=rng.normal(0.08, 0.06, size=n),
            revenue_vol=rng.uniform(0.01, 0.05, size=n),
            gross_margin=rng.uniform(0.25, 0.65, size=n),
            margin_vol=rng.uniform(0.002, 0.04, size=n),
            opex_ratio=rng.uniform(0.40, 0.95, size=n),
            retention=rng.uniform(0.5, 1.0, size=n),
            leverage=rng.uniform(0.1, 1.5, size=n),
            equity_mult=rng.uniform(4.0, 16.0, size=n),
            accrual_mu=rng.normal(0.0, 0.03, size=n),
            capex_frac=rng.uniform(0.02, 0.12, size=n),
        )

    def _quarter_shocks(
        self, period_end: pd.Timestamp
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw all shocks for one quarter in a fixed, documented order."""
        ordinal = period_end.year * 4 + (period_end.quarter - 1)
        rng = np.random.default_rng([self.seed, _FUND_QUARTER_STREAM + ordinal])
        z_revenue = rng.standard_normal(self.n_assets)
        z_margin = rng.standard_normal(self.n_assets)
        z_accrual = rng.standard_normal(self.n_assets)
        u_jitter = rng.random(self.n_assets)
        return z_revenue, z_margin, z_accrual, u_jitter

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> FundamentalDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        if start_ts < self.inception:
            raise DataError(
                f"start {start_ts.date()} precedes inception {self.inception.date()}"
            )
        idx_map = {sym: i for i, sym in enumerate(self.symbols)}
        unknown = [s for s in symbols_list if s not in idx_map]
        if unknown:
            raise DataError(f"Unknown synthetic symbols: {unknown[:10]}")
        quarter_ends = pd.date_range(self.inception, end_ts, freq="QE")
        if len(quarter_ends) == 0:
            raise DataError(
                f"No completed fiscal quarters in [{self.inception.date()}, {end_ts.date()}]"
            )

        p = self._params
        n_q, n = len(quarter_ends), self.n_assets
        items = {name: np.empty((n_q, n)) for name in FUNDAMENTAL_ITEMS}
        pub = np.empty((n_q, n), dtype="datetime64[ns]")

        log_revenue = np.log(p.revenue0)
        equity = p.equity_mult * p.revenue0
        equity0 = equity.copy()
        for q, period_end in enumerate(quarter_ends):
            z_rev, z_gm, z_acc, u_jit = self._quarter_shocks(period_end)
            log_revenue = log_revenue + p.growth_annual / 4.0 + p.revenue_vol * z_rev
            revenue = np.exp(log_revenue)
            margin = np.clip(p.gross_margin + p.margin_vol * z_gm, 0.05, 0.95)
            gross_profit = margin * revenue
            opex = p.opex_ratio * p.gross_margin * revenue
            net_income = (gross_profit - opex) * (1.0 - 0.21)
            equity = equity + p.retention * net_income
            liabilities = p.leverage * equity0 * np.sqrt(revenue / p.revenue0)
            assets = equity + liabilities
            accruals = (p.accrual_mu + 0.01 * z_acc) * revenue
            items["revenue"][q] = revenue
            items["gross_profit"][q] = gross_profit
            items["net_income"][q] = net_income
            items["total_assets"][q] = assets
            items["total_liabilities"][q] = liabilities
            items["total_equity"][q] = equity
            items["shares_outstanding"][q] = p.shares
            items["operating_cash_flow"][q] = net_income - accruals
            items["capital_expenditure"][q] = -p.capex_frac * revenue
            lag_days = self.publication_lag_days + np.floor(
                u_jit * (self.publication_jitter_days + 1)
            ).astype(int)
            pub[q] = period_end.to_datetime64() + lag_days * np.timedelta64(1, "D")

        cols = np.array([idx_map[s] for s in symbols_list], dtype=int)
        in_range = (quarter_ends >= start_ts) & (quarter_ends <= end_ts)
        kept = np.flatnonzero(in_range)
        if len(kept) == 0:
            raise DataError(
                f"No completed fiscal quarters in [{start_ts.date()}, {end_ts.date()}]"
            )
        rows: dict[str, Any] = {
            "symbol": np.repeat(symbols_list, len(kept)),
            "period_end": np.tile(quarter_ends[kept].to_numpy(), len(cols)),
            "publication_date": pub[np.ix_(kept, cols)].T.reshape(-1),
        }
        for name in FUNDAMENTAL_ITEMS:
            rows[name] = items[name][np.ix_(kept, cols)].T.reshape(-1)
        return FundamentalDataset(
            pd.DataFrame(rows),
            metadata={
                "source": self.name,
                "seed": self.seed,
                "inception": str(self.inception.date()),
                "publication_basis": "assumed_lag",
                "publication_lag_days": self.publication_lag_days,
            },
        )


class YahooFundamentalSource(FundamentalSource):
    """Quarterly statements from Yahoo Finance via ``yfinance``.

    Requires the optional ``yfinance`` dependency
    (``pip install quantlab[data]``); imported inside ``fetch``, never at
    module load. Always wrap in :class:`CachedFundamentalSource`.

    Honesty constraints of this source — read before backtesting:

    - **Assumed publication dates.** Yahoo does not expose filing dates, so
      ``publication_date = period_end + publication_lag_days`` (default 45).
      SEC 10-Q deadlines are 40-45 days, so 45 is conservative for interim
      quarters — but fiscal fourth quarters arrive via the 10-K (60-90 day
      deadlines), for which 45 days is *optimistic*. Use 60+ days for a
      stricter test, or a point-in-time vendor for real filing dates.
    - **Shallow history.** Yahoo returns only the trailing ~4-6 quarters.
      Multi-year fundamental backtests are impossible on a fresh fetch; the
      cache accumulates quarters across periodic fetches over time.
    - **Silent restatements.** Yahoo shows today's view of past quarters.
      A backtest through the cache sees each quarter's values as of the
      first fetch that captured them — an approximation of point-in-time,
      exact only for quarters captured near their filing dates.

    Parameters
    ----------
    publication_lag_days:
        Calendar days after period end at which a statement is assumed
        public.
    strict:
        If True (default), symbols returning no statements raise
        :class:`~quantlab.exceptions.DataError`; otherwise they are dropped
        with a warning.
    """

    name = "yahoo_fundamentals"

    #: canonical item -> (statement, vendor row-label candidates in priority order)
    _ITEM_CANDIDATES: dict[str, tuple[str, tuple[str, ...]]] = {
        "revenue": ("income", ("Total Revenue", "Operating Revenue")),
        "gross_profit": ("income", ("Gross Profit",)),
        "net_income": (
            "income",
            (
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income Continuous Operations",
            ),
        ),
        "total_assets": ("balance", ("Total Assets",)),
        "total_liabilities": (
            "balance",
            ("Total Liabilities Net Minority Interest", "Total Liabilities"),
        ),
        "total_equity": (
            "balance",
            (
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
            ),
        ),
        "shares_outstanding": ("balance", ("Ordinary Shares Number", "Share Issued")),
        "operating_cash_flow": (
            "cashflow",
            ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
        ),
        "capital_expenditure": (
            "cashflow",
            ("Capital Expenditure", "Net PPE Purchase And Sale"),
        ),
    }

    def __init__(self, publication_lag_days: int = 45, strict: bool = True) -> None:
        if publication_lag_days < 0:
            raise DataError("publication_lag_days must be >= 0")
        self.publication_lag_days = int(publication_lag_days)
        self.strict = bool(strict)

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> FundamentalDataset:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DataError(
                "YahooFundamentalSource requires 'yfinance'; install quantlab[data]"
            ) from exc

        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        pieces: list[pd.DataFrame] = []
        for sym in symbols_list:
            try:
                ticker = yf.Ticker(sym)
                mapped = self._map_statements(
                    ticker.quarterly_income_stmt,
                    ticker.quarterly_balance_sheet,
                    ticker.quarterly_cashflow,
                )
            except Exception as exc:  # vendor/network failures vary widely
                if self.strict:
                    raise DataError(f"Yahoo fundamentals failed for '{sym}': {exc}") from exc
                logger.warning("Dropping %s: Yahoo fundamentals failed (%s)", sym, exc)
                continue
            if mapped.empty:
                if self.strict:
                    raise DataError(f"Yahoo returned no fundamental statements for '{sym}'")
                logger.warning("Dropping symbol with no Yahoo fundamentals: %s", sym)
                continue
            mapped = mapped.reset_index().rename(columns={"index": "period_end"})
            mapped.insert(0, "symbol", sym)
            mapped["publication_date"] = mapped["period_end"] + pd.Timedelta(
                days=self.publication_lag_days
            )
            pieces.append(mapped)
        if not pieces:
            raise DataError("Yahoo returned no fundamental statements for any symbol")
        stmts = pd.concat(pieces, ignore_index=True)
        stmts = stmts[
            (stmts["period_end"] >= start_ts) & (stmts["period_end"] <= end_ts)
        ]
        if stmts.empty:
            raise DataError(
                f"No fundamental statements with period end in "
                f"[{start_ts.date()}, {end_ts.date()}] (Yahoo only serves the "
                "trailing ~4-6 quarters)"
            )
        return FundamentalDataset(
            stmts,
            metadata={
                "source": self.name,
                "publication_basis": "assumed_lag",
                "publication_lag_days": self.publication_lag_days,
            },
        )

    @classmethod
    def _map_statements(
        cls,
        income: pd.DataFrame | None,
        balance: pd.DataFrame | None,
        cashflow: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Map yfinance statement frames (rows = vendor line items, columns =
        period ends) to a frame indexed by period end with canonical item
        columns. Pure so it is testable offline against fixture frames.
        """
        frames = {"income": income, "balance": balance, "cashflow": cashflow}
        cleaned: dict[str, pd.DataFrame] = {}
        periods: pd.DatetimeIndex = pd.DatetimeIndex([])
        for key, frame in frames.items():
            if frame is None or frame.empty:
                cleaned[key] = pd.DataFrame()
                continue
            idx = pd.DatetimeIndex(pd.to_datetime(frame.columns))
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            work = frame.copy()
            work.columns = idx.normalize()
            work = work.loc[:, ~work.columns.duplicated(keep="first")]
            cleaned[key] = work
            periods = periods.union(work.columns)
        if len(periods) == 0:
            return pd.DataFrame()
        out = pd.DataFrame(index=periods.sort_values(), columns=list(cls._ITEM_CANDIDATES))
        for item, (statement, candidates) in cls._ITEM_CANDIDATES.items():
            frame = cleaned[statement]
            if frame.empty:
                continue
            for candidate in candidates:
                if candidate in frame.index:
                    values = pd.to_numeric(frame.loc[candidate], errors="coerce")
                    out[item] = values.reindex(out.index)
                    break
        out = out.astype(float).dropna(how="all")
        return out


# --------------------------------------------------------------------- cache
class CachedFundamentalSource(FundamentalSource):
    """Parquet cache around any :class:`FundamentalSource`.

    Follows the market-data cache machinery: per-symbol parquet files plus a
    covered-range JSON sidecar under
    ``{cache_dir}/{source.name}/quarterly/``, atomic writes, and corrupt
    files degrading to a refetch. Covered ranges track *requested* windows
    (period-end dates), so a range the vendor simply has no statements for
    is remembered and not re-fetched.

    Merging keeps every previously cached row that a refetch does not
    re-deliver. For snapshot vendors (Yahoo) this means the cache
    **accumulates** quarters across periodic fetches — the honest way to
    build point-in-time history going forward from a source that only
    serves a trailing window.
    """

    def __init__(
        self, source: FundamentalSource, cache_dir: Path | str | None = None
    ) -> None:
        if cache_dir is None:
            from quantlab.config import get_settings

            cache_dir = get_settings().cache_dir
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.name = f"cached[{source.name}]"

    # ----------------------------------------------------------- filesystem
    def _dir(self) -> Path:
        return self.cache_dir / self.source.name / "quarterly"

    def _statement_path(self, symbol: str) -> Path:
        return self._dir() / f"{_safe_symbol(symbol)}.parquet"

    def _meta_path(self, symbol: str) -> Path:
        return self._dir() / f"{_safe_symbol(symbol)}.meta.json"

    def _read_cached(
        self, symbol: str
    ) -> tuple[pd.DataFrame | None, tuple[pd.Timestamp, pd.Timestamp] | None]:
        """Return (statement rows, covered range) or (None, None)."""
        path = self._statement_path(symbol)
        if not path.is_file():
            return None, None
        try:
            frame = pd.read_parquet(path)
            missing = [c for c in FUNDAMENTAL_ID_COLUMNS if c not in frame.columns]
            if missing:
                raise CacheError(f"Cached statements {path} missing columns {missing}")
            for col in ("period_end", "publication_date"):
                frame[col] = pd.to_datetime(frame[col])
        except Exception as exc:
            logger.warning("Corrupt fundamental cache for %s (%s); refetching", symbol, exc)
            return None, None
        covered = None
        meta_path = self._meta_path(symbol)
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                covered = (
                    pd.Timestamp(meta["covered_start"]),
                    pd.Timestamp(meta["covered_end"]),
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                covered = None
        if covered is None and len(frame):
            covered = (frame["period_end"].min(), frame["period_end"].max())
        return frame, covered

    def _write_cached(
        self,
        symbol: str,
        frame: pd.DataFrame,
        covered: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        path = self._statement_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            frame.reset_index(drop=True).to_parquet(tmp)
            os.replace(tmp, path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise CacheError(f"Failed writing fundamental cache for {symbol}: {exc}") from exc
        meta = {
            "covered_start": str(covered[0].date()),
            "covered_end": str(covered[1].date()),
        }
        meta_path = self._meta_path(symbol)
        fd, tmp_name = tempfile.mkstemp(dir=meta_path.parent, suffix=".json.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.write_text(json.dumps(meta))
            os.replace(tmp, meta_path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise CacheError(f"Failed writing cache meta for {symbol}: {exc}") from exc

    # ------------------------------------------------------------------ api
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> FundamentalDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        self._dir().mkdir(parents=True, exist_ok=True)

        rows: dict[str, pd.DataFrame | None] = {}
        covered: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}
        gaps: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
        for sym in symbols_list:
            frame, cov = self._read_cached(sym)
            rows[sym], covered[sym] = frame, cov
            for gap in CachedSource._missing_ranges(cov, start_ts, end_ts):
                gaps.setdefault(gap, []).append(sym)

        for (gap_start, gap_end), gap_symbols in sorted(gaps.items()):
            logger.info(
                "Fundamental cache miss [%s..%s] for %d symbol(s) via %s",
                gap_start.date(),
                gap_end.date(),
                len(gap_symbols),
                self.source.name,
            )
            try:
                fetched = self.source.fetch(gap_symbols, gap_start, gap_end)
                fetched_stmts = fetched.statements
            except DataError:
                # The vendor has nothing in this window (common for snapshot
                # vendors asked for old quarters). Remember the range as
                # covered so it is not re-requested every call.
                fetched_stmts = None
            for sym in gap_symbols:
                if fetched_stmts is not None:
                    new = fetched_stmts[fetched_stmts["symbol"] == sym]
                    rows[sym] = self._merge(rows[sym], new)
                covered[sym] = CachedSource._extend_covered(covered[sym], gap_start, gap_end)

        pieces: list[pd.DataFrame] = []
        for sym in symbols_list:
            frame, cov = rows[sym], covered[sym]
            if frame is None or frame.empty:
                logger.warning("No fundamental statements cached for %s", sym)
                continue
            self._write_cached(sym, frame, cov or (start_ts, end_ts))
            window = frame[
                (frame["period_end"] >= start_ts) & (frame["period_end"] <= end_ts)
            ]
            if len(window):
                pieces.append(window)
        if not pieces:
            raise DataError(
                f"No fundamental statements with period end in "
                f"[{start_ts.date()}, {end_ts.date()}] for any of {symbols_list[:5]}"
            )
        metadata: dict[str, Any] = {"source": self.source.name, "cached": True}
        lag = getattr(self.source, "publication_lag_days", None)
        if lag is not None:
            metadata["publication_basis"] = "assumed_lag"
            metadata["publication_lag_days"] = lag
        return FundamentalDataset(pd.concat(pieces, ignore_index=True), metadata=metadata)

    @staticmethod
    def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
        """Union of statement rows; a re-delivered version replaces the old
        row, while rows the vendor no longer serves are kept (accumulation).
        """
        if old is None or old.empty:
            return new.copy()
        if new.empty:
            return old
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(
            ["period_end", "publication_date"], keep="last"
        )
        return combined.sort_values(["period_end", "publication_date"])

    # ------------------------------------------------------------ utilities
    def cached_range(self, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Covered (requested-and-stored) period-end range for ``symbol``."""
        _, cov = self._read_cached(symbol)
        return cov

    def clear(self, symbols: Sequence[str] | None = None) -> int:
        """Delete cached files; all symbols when ``symbols`` is None.

        Returns the number of parquet files removed.
        """
        d = self._dir()
        if not d.exists():
            return 0
        if symbols is None:
            targets = list(d.glob("*.parquet"))
        else:
            targets = [self._statement_path(s) for s in symbols]
        removed = 0
        for path in targets:
            if path.is_file():
                path.unlink()
                removed += 1
            meta = path.parent / (path.stem + ".meta.json")
            meta.unlink(missing_ok=True)
        return removed


def _safe_symbol(symbol: str) -> str:
    """Filesystem-safe file stem for a symbol."""
    return "".join(c if (c.isalnum() or c in "._-^=") else "_" for c in symbol)
