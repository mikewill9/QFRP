"""Leak-proof cross-validation splitters.

Why ordinary k-fold lies to you here
------------------------------------
A panel row at date ``t`` carries a label spanning ``(t, t+h]``. With
plain k-fold, a training row dated one bar before a test block has a
label window that overlaps the test labels — the model is literally
trained on (part of) the outcome it is later scored on, and the CV score
inflates. Two defenses, both operating on whole dates (groups), never on
individual rows:

- **Purging** removes every training date whose label window overlaps
  any test label window: with a test block spanning date positions
  ``[a, b]``, all dates ``j`` with ``a - h < j < b + h`` are excluded.
- **Embargo** additionally drops ``embargo`` dates *after* the purged
  zone, guarding against serial correlation that leaks information
  beyond the exact label overlap.

:class:`PurgedKFold` scores a model fairly (test blocks are interior, so
"future" data legitimately appears in training folds — fine for model
selection, not a tradeable signal). :class:`WalkForward` only ever
trains on data strictly before the test window (with a purge gap of the
label horizon in between), so its out-of-sample predictions are causal
and may be backtested.

Both splitters yield ``(train_rows, test_rows)`` integer positions into
the panel. Horizon and embargo are measured in panel date steps.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from quantlab.exceptions import ModelError
from quantlab.ml.datasets import PanelDataset

__all__ = ["PurgedKFold", "WalkForward"]

Split = tuple[np.ndarray, np.ndarray]


def _date_codes(panel: PanelDataset) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Unique sorted dates and, per row, the position of its date."""
    dates = panel.dates
    codes = dates.get_indexer(pd.DatetimeIndex(panel.groups))
    return dates, codes


def _rows_for(codes: np.ndarray, date_positions: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(codes, date_positions))


class PurgedKFold:
    """K-fold over dates with purging and embargo (Lopez de Prado).

    Parameters
    ----------
    n_splits:
        Number of contiguous test blocks.
    horizon:
        Label horizon ``h`` in date steps; pass the panel's horizon.
        Training dates within ``h`` of a test block are purged on both
        sides because their label windows overlap the test labels.
    embargo:
        Extra dates dropped after each test block's purge zone.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ModelError(f"n_splits must be >= 2, got {n_splits}")
        if horizon < 1:
            raise ModelError(f"horizon must be >= 1, got {horizon}")
        if embargo < 0:
            raise ModelError(f"embargo must be >= 0, got {embargo}")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, panel: PanelDataset) -> Iterator[Split]:
        """Yield ``(train_rows, test_rows)`` per fold, in date order."""
        dates, codes = _date_codes(panel)
        m = len(dates)
        if m < self.n_splits:
            raise ModelError(
                f"Cannot make {self.n_splits} folds from {m} unique dates"
            )
        blocks = np.array_split(np.arange(m), self.n_splits)
        for block in blocks:
            a, b = int(block[0]), int(block[-1])
            all_positions = np.arange(m)
            # Purge label overlap on both sides, then embargo after.
            excluded = (all_positions > a - self.horizon) & (
                all_positions < b + self.horizon + self.embargo
            )
            train_positions = all_positions[~excluded]
            if len(train_positions) == 0:
                raise ModelError(
                    "Purging removed every training date; reduce n_splits, "
                    "horizon, or embargo"
                )
            yield _rows_for(codes, train_positions), _rows_for(codes, block)


class WalkForward:
    """Rolling train -> gap -> test windows marching forward in time.

    Every window trains on ``train_periods`` dates, skips ``purge``
    dates (defaults to the panel's label horizon, so no training label
    window reaches into the test block), then tests on the next
    ``test_periods`` dates. ``step`` (default ``test_periods``) advances
    the window, so default test blocks tile the sample without overlap
    and the concatenated test predictions form one continuous
    out-of-sample signal.
    """

    def __init__(
        self,
        train_periods: int,
        test_periods: int,
        step: int | None = None,
        purge: int | None = None,
    ) -> None:
        if train_periods < 1:
            raise ModelError(f"train_periods must be >= 1, got {train_periods}")
        if test_periods < 1:
            raise ModelError(f"test_periods must be >= 1, got {test_periods}")
        if step is not None and step < 1:
            raise ModelError(f"step must be >= 1, got {step}")
        if purge is not None and purge < 0:
            raise ModelError(f"purge must be >= 0, got {purge}")
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step = test_periods if step is None else step
        self.purge = purge

    def split(self, panel: PanelDataset) -> Iterator[Split]:
        """Yield ``(train_rows, test_rows)``, strictly past -> future."""
        dates, codes = _date_codes(panel)
        m = len(dates)
        purge = panel.horizon if self.purge is None else self.purge
        first_test = self.train_periods + purge
        if first_test >= m:
            raise ModelError(
                f"Need more than {first_test} unique dates for "
                f"train={self.train_periods} + purge={purge}, got {m}"
            )
        produced = False
        for start in range(0, m, self.step):
            test_start = start + first_test
            if test_start >= m:
                break
            test_end = min(test_start + self.test_periods, m)
            train_positions = np.arange(start, start + self.train_periods)
            test_positions = np.arange(test_start, test_end)
            produced = True
            yield _rows_for(codes, train_positions), _rows_for(codes, test_positions)
        if not produced:  # pragma: no cover - guarded by the check above
            raise ModelError("WalkForward produced no folds")
