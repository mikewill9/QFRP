"""Data-quality validation.

Checks are composable objects implementing
``run(dataset) -> list[ValidationIssue]``; a :class:`DataValidator` executes a
suite of them and returns a :class:`ValidationReport`. Severity semantics:

- ``ERROR``: data is unusable as-is (non-positive prices, inconsistent OHLC,
  fully missing symbols). ``report.passed`` is False and
  ``report.raise_for_errors()`` raises.
- ``WARNING``: data is suspicious and should be inspected or repaired
  (stale runs, extreme returns, high missingness).
- ``INFO``: notable but usually benign (calendar gaps around holidays).

Structural invariants (sorted unique tz-naive index, aligned fields) are
enforced by the :class:`~quantlab.data.dataset.MarketDataset` constructor and
are therefore not re-checked here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataValidationError
from quantlab.types import Frequency

__all__ = [
    "DataValidator",
    "ExtremeReturnCheck",
    "FrequencyGapCheck",
    "MissingDataCheck",
    "NonPositivePriceCheck",
    "OHLCConsistencyCheck",
    "Severity",
    "StalePriceCheck",
    "ValidationCheck",
    "ValidationIssue",
    "ValidationReport",
    "ZeroVolumeRunCheck",
    "default_checks",
]


class Severity(IntEnum):
    """Issue severity, ordered so comparisons read naturally."""

    INFO = 10
    WARNING = 20
    ERROR = 30


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single finding from one check on (usually) one symbol."""

    check: str
    severity: Severity
    message: str
    symbol: str | None = None
    count: int = 1
    details: dict[str, Any] = field(default_factory=dict)


class ValidationCheck(ABC):
    """Base class for data-quality checks."""

    #: Human-readable identifier reported on issues.
    name: str = "check"

    @abstractmethod
    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        """Inspect ``dataset`` and return zero or more issues."""


# --------------------------------------------------------------------- utils
def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max())


def _example_dates(index: pd.DatetimeIndex, mask: np.ndarray, k: int = 5) -> list[str]:
    return [str(d.date()) for d in index[mask][:k]]


# -------------------------------------------------------------------- checks
class MissingDataCheck(ValidationCheck):
    """Flag symbols with excessive NaN in the close series.

    ERROR when a symbol has no data at all; WARNING when the missing fraction
    exceeds ``max_missing_frac``.
    """

    name = "missing_data"

    def __init__(self, max_missing_frac: float = 0.10) -> None:
        if not 0.0 <= max_missing_frac < 1.0:
            raise ValueError("max_missing_frac must be in [0, 1)")
        self.max_missing_frac = max_missing_frac

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        frac = dataset.close.isna().mean(axis=0)
        for sym, f in frac.items():
            if f == 1.0:
                issues.append(
                    ValidationIssue(
                        check=self.name,
                        severity=Severity.ERROR,
                        symbol=str(sym),
                        message="close series is entirely missing",
                        count=dataset.n_periods,
                    )
                )
            elif f > self.max_missing_frac:
                issues.append(
                    ValidationIssue(
                        check=self.name,
                        severity=Severity.WARNING,
                        symbol=str(sym),
                        message=f"{f:.1%} of close values missing "
                        f"(threshold {self.max_missing_frac:.1%})",
                        count=int(f * dataset.n_periods),
                    )
                )
        return issues


class NonPositivePriceCheck(ValidationCheck):
    """Zero or negative prices in any price field are always ERROR."""

    name = "non_positive_price"

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for field_name in ("open", "high", "low", "close"):
            frame = dataset.field(field_name)
            bad = frame.le(0.0)
            for sym in frame.columns[bad.any(axis=0)]:
                mask = bad[sym].to_numpy()
                issues.append(
                    ValidationIssue(
                        check=self.name,
                        severity=Severity.ERROR,
                        symbol=str(sym),
                        message=f"non-positive values in '{field_name}'",
                        count=int(mask.sum()),
                        details={"dates": _example_dates(dataset.index, mask)},
                    )
                )
        return issues


class OHLCConsistencyCheck(ValidationCheck):
    """High must dominate open/low/close; low must be dominated by them.

    Violations beyond ``tolerance`` (relative) are ERROR: they usually
    indicate corrupted bars or a bad column mapping.
    """

    name = "ohlc_consistency"

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = tolerance

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        o, h, l_, c = dataset.open, dataset.high, dataset.low, dataset.close
        tol = 1.0 + self.tolerance
        viol = (
            (h * tol < o) | (h * tol < c) | (h * tol < l_)
            | (l_ > o * tol) | (l_ > c * tol)
        )
        viol &= h.notna() & l_.notna() & o.notna() & c.notna()
        issues: list[ValidationIssue] = []
        for sym in viol.columns[viol.any(axis=0)]:
            mask = viol[sym].to_numpy()
            issues.append(
                ValidationIssue(
                    check=self.name,
                    severity=Severity.ERROR,
                    symbol=str(sym),
                    message="OHLC bounds violated (high < max(o,c,l) or low > min(o,c))",
                    count=int(mask.sum()),
                    details={"dates": _example_dates(dataset.index, mask)},
                )
            )
        return issues


class StalePriceCheck(ValidationCheck):
    """Warn on long runs of exactly unchanged close prices.

    Repeated identical closes typically mean a halted listing, a stale feed,
    or forward-filled vendor data masquerading as fresh prints.
    """

    name = "stale_price"

    def __init__(self, max_run: int = 10) -> None:
        if max_run < 2:
            raise ValueError("max_run must be >= 2")
        self.max_run = max_run

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        close = dataset.close
        unchanged = close.diff().eq(0.0) & close.notna()
        for sym in close.columns:
            run = _max_true_run(unchanged[sym].to_numpy())
            if run >= self.max_run:
                issues.append(
                    ValidationIssue(
                        check=self.name,
                        severity=Severity.WARNING,
                        symbol=str(sym),
                        message=f"close unchanged for {run} consecutive bars "
                        f"(threshold {self.max_run})",
                        count=run,
                    )
                )
        return issues


class ExtremeReturnCheck(ValidationCheck):
    """Warn on single-bar simple returns beyond ``max_abs_return``.

    Large one-bar moves are legitimate occasionally; clusters of them usually
    mean unadjusted splits or bad prints, so eyeball the reported dates.
    """

    name = "extreme_return"

    def __init__(self, max_abs_return: float = 0.50) -> None:
        if max_abs_return <= 0:
            raise ValueError("max_abs_return must be positive")
        self.max_abs_return = max_abs_return

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        rets = dataset.returns(method="simple")
        extreme = rets.abs().gt(self.max_abs_return)
        issues: list[ValidationIssue] = []
        for sym in rets.columns[extreme.any(axis=0)]:
            mask = extreme[sym].to_numpy()
            worst = float(rets[sym].abs().max())
            issues.append(
                ValidationIssue(
                    check=self.name,
                    severity=Severity.WARNING,
                    symbol=str(sym),
                    message=f"|return| exceeded {self.max_abs_return:.0%} "
                    f"(worst {worst:.1%}); possible split or bad print",
                    count=int(mask.sum()),
                    details={"dates": _example_dates(dataset.index, mask)},
                )
            )
        return issues


class ZeroVolumeRunCheck(ValidationCheck):
    """Warn on long runs of zero volume (likely halted or dead listings)."""

    name = "zero_volume_run"

    def __init__(self, max_run: int = 5) -> None:
        if max_run < 1:
            raise ValueError("max_run must be >= 1")
        self.max_run = max_run

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        vol = dataset.volume
        if vol.isna().all().all():
            return []
        zero = vol.eq(0.0) & vol.notna()
        issues: list[ValidationIssue] = []
        for sym in vol.columns:
            run = _max_true_run(zero[sym].to_numpy())
            if run >= self.max_run:
                issues.append(
                    ValidationIssue(
                        check=self.name,
                        severity=Severity.WARNING,
                        symbol=str(sym),
                        message=f"volume was zero for {run} consecutive bars",
                        count=run,
                    )
                )
        return issues


class FrequencyGapCheck(ValidationCheck):
    """Report unusually large gaps between consecutive index dates.

    For daily data a handful of 3-4 day gaps (weekends plus holidays) is
    normal; gaps beyond ``max_gap_days`` are surfaced as INFO so the
    researcher can confirm they are exchange closures rather than data holes.
    """

    name = "frequency_gap"

    def __init__(self, max_gap_days: int | None = None) -> None:
        self.max_gap_days = max_gap_days

    def run(self, dataset: MarketDataset) -> list[ValidationIssue]:
        if dataset.n_periods < 2:
            return []
        defaults = {Frequency.DAILY: 5, Frequency.WEEKLY: 15, Frequency.MONTHLY: 45}
        limit = self.max_gap_days or defaults[dataset.frequency]
        deltas = dataset.index.to_series().diff().dt.days.iloc[1:]
        big = deltas[deltas > limit]
        if big.empty:
            return []
        return [
            ValidationIssue(
                check=self.name,
                severity=Severity.INFO,
                message=f"{len(big)} calendar gap(s) exceeding {limit} days "
                f"(largest {int(big.max())} days)",
                count=len(big),
                details={"dates": [str(d.date()) for d in big.index[:5]]},
            )
        ]


# -------------------------------------------------------------------- runner
def default_checks() -> list[ValidationCheck]:
    """The standard suite applied by ``DataValidator()``."""
    return [
        MissingDataCheck(),
        NonPositivePriceCheck(),
        OHLCConsistencyCheck(),
        StalePriceCheck(),
        ExtremeReturnCheck(),
        ZeroVolumeRunCheck(),
        FrequencyGapCheck(),
    ]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregated validation outcome."""

    issues: tuple[ValidationIssue, ...]
    n_symbols: int
    n_periods: int

    @property
    def passed(self) -> bool:
        """True when no ERROR-severity issue was found."""
        return all(i.severity < Severity.ERROR for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def raise_for_errors(self) -> None:
        """Raise :class:`DataValidationError` if any ERROR issues exist."""
        if self.passed:
            return
        errors = [i for i in self.issues if i.severity == Severity.ERROR]
        lines = [f"  [{i.check}] {i.symbol or '*'}: {i.message}" for i in errors[:10]]
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise DataValidationError(
            f"Validation failed with {len(errors)} error(s):\n" + "\n".join(lines) + more
        )

    def to_frame(self) -> pd.DataFrame:
        """Issues as a DataFrame (one row per issue), sorted by severity."""
        if not self.issues:
            return pd.DataFrame(
                columns=["check", "severity", "symbol", "count", "message"]
            )
        rows = [
            {
                "check": i.check,
                "severity": i.severity.name,
                "symbol": i.symbol,
                "count": i.count,
                "message": i.message,
            }
            for i in sorted(self.issues, key=lambda x: (-x.severity, x.check, x.symbol or ""))
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationReport({status}: {self.n_errors} errors, "
            f"{self.n_warnings} warnings, {len(self.issues)} total issues; "
            f"{self.n_symbols} symbols x {self.n_periods} bars)"
        )


class DataValidator:
    """Run a suite of checks against a dataset.

    Parameters
    ----------
    checks:
        Checks to execute; defaults to :func:`default_checks`.
    """

    def __init__(self, checks: Sequence[ValidationCheck] | None = None) -> None:
        self.checks = list(checks) if checks is not None else default_checks()

    def validate(self, dataset: MarketDataset) -> ValidationReport:
        """Execute all checks and aggregate their issues."""
        issues: list[ValidationIssue] = []
        for check in self.checks:
            issues.extend(check.run(dataset))
        return ValidationReport(
            issues=tuple(issues),
            n_symbols=dataset.n_symbols,
            n_periods=dataset.n_periods,
        )
