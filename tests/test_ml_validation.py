"""Splitter correctness: the no-lookahead guarantees, checked positionally."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import PanelDataset, PurgedKFold, WalkForward


def _panel(n_dates=60, horizon=5, symbols=("A", "B", "C"), seed=1) -> PanelDataset:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_dates)
    prices = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_dates))) for s in symbols},
        index=idx,
    )
    factor = pd.DataFrame(
        rng.normal(size=prices.shape), index=idx, columns=prices.columns
    )
    return PanelDataset.from_factors({"f": factor}, prices, horizon=horizon)


def _date_positions(panel: PanelDataset, rows: np.ndarray) -> np.ndarray:
    """Positions (in the sorted unique-date array) of the given rows' dates."""
    dates = panel.dates
    row_dates = pd.DatetimeIndex(panel.groups.iloc[rows])
    return np.unique(dates.get_indexer(row_dates))


# ------------------------------------------------------------ PurgedKFold


def test_purged_kfold_train_never_touches_the_purge_zone():
    panel = _panel(horizon=5)
    embargo = 3
    for train_rows, test_rows in PurgedKFold(4, horizon=5, embargo=embargo).split(panel):
        train_pos = _date_positions(panel, train_rows)
        test_pos = _date_positions(panel, test_rows)
        a, b = test_pos.min(), test_pos.max()
        # Every training date sits clear of the overlap+embargo zone.
        assert ((train_pos <= a - 5) | (train_pos >= b + 5 + embargo)).all()
        assert len(np.intersect1d(train_rows, test_rows)) == 0


def test_purged_kfold_test_blocks_partition_all_dates():
    panel = _panel()
    seen: list[np.ndarray] = []
    for _, test_rows in PurgedKFold(4, horizon=5).split(panel):
        seen.append(_date_positions(panel, test_rows))
    covered = np.concatenate(seen)
    assert len(covered) == len(set(covered)) == len(panel.dates)


def test_purged_kfold_keeps_whole_dates_together():
    panel = _panel()
    for train_rows, test_rows in PurgedKFold(4, horizon=5).split(panel):
        train_dates = set(panel.groups.iloc[train_rows])
        test_dates = set(panel.groups.iloc[test_rows])
        assert train_dates.isdisjoint(test_dates)


def test_purged_kfold_purge_grows_with_horizon():
    panel = _panel(horizon=5)
    def n_train(horizon: int) -> int:
        splits = PurgedKFold(4, horizon=horizon).split(panel)
        return sum(len(train) for train, _ in splits)

    assert n_train(10) < n_train(1)


def test_purged_kfold_raises_when_purging_eats_everything():
    panel = _panel(n_dates=30, horizon=1)
    with pytest.raises(ModelError, match="removed every training date"):
        list(PurgedKFold(2, horizon=30).split(panel))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_splits": 1}, {"horizon": 0}, {"embargo": -1}],
)
def test_purged_kfold_validation(kwargs):
    with pytest.raises(ModelError):
        PurgedKFold(**{"n_splits": 3, "horizon": 1, "embargo": 0, **kwargs})


def test_purged_kfold_rejects_more_splits_than_dates():
    panel = _panel(n_dates=10, horizon=2)
    with pytest.raises(ModelError, match="folds"):
        list(PurgedKFold(20, horizon=1).split(panel))


# ------------------------------------------------------------ WalkForward


def test_walk_forward_train_strictly_before_test_with_horizon_gap():
    panel = _panel(n_dates=80, horizon=5)
    folds = list(WalkForward(train_periods=20, test_periods=10).split(panel))
    assert len(folds) > 1
    for train_rows, test_rows in folds:
        train_pos = _date_positions(panel, train_rows)
        test_pos = _date_positions(panel, test_rows)
        # Strictly past -> future, with the default purge = panel.horizon.
        assert train_pos.max() < test_pos.min()
        assert test_pos.min() - train_pos.max() - 1 == panel.horizon


def test_walk_forward_default_step_tiles_test_blocks_without_overlap():
    panel = _panel(n_dates=80, horizon=2)
    test_blocks = [
        _date_positions(panel, test)
        for _, test in WalkForward(train_periods=20, test_periods=10).split(panel)
    ]
    for earlier, later in zip(test_blocks, test_blocks[1:], strict=False):
        assert earlier.max() + 1 == later.min()  # contiguous, no overlap


def test_walk_forward_train_window_is_fixed_length():
    panel = _panel(n_dates=80, horizon=2)
    for train_rows, _ in WalkForward(train_periods=15, test_periods=10).split(panel):
        assert len(_date_positions(panel, train_rows)) == 15


def test_walk_forward_explicit_purge_overrides_panel_horizon():
    panel = _panel(n_dates=60, horizon=5)
    folds = list(
        WalkForward(train_periods=10, test_periods=5, purge=0).split(panel)
    )
    train_pos = _date_positions(panel, folds[0][0])
    test_pos = _date_positions(panel, folds[0][1])
    assert test_pos.min() - train_pos.max() == 1  # no gap when purge=0


def test_walk_forward_custom_step_overlaps_test_blocks():
    panel = _panel(n_dates=80, horizon=1)
    blocks = [
        _date_positions(panel, test)
        for _, test in WalkForward(20, 10, step=5).split(panel)
    ]
    assert blocks[0].max() >= blocks[1].min()  # overlap by construction


def test_walk_forward_raises_without_enough_dates():
    panel = _panel(n_dates=20, horizon=2)
    with pytest.raises(ModelError, match="unique dates"):
        list(WalkForward(train_periods=50, test_periods=10).split(panel))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_periods": 0, "test_periods": 5},
        {"train_periods": 5, "test_periods": 0},
        {"train_periods": 5, "test_periods": 5, "step": 0},
        {"train_periods": 5, "test_periods": 5, "purge": -1},
    ],
)
def test_walk_forward_validation(kwargs):
    with pytest.raises(ModelError):
        WalkForward(**kwargs)


# ------------------------------------------------- WalkForward expanding mode


def test_walk_forward_default_mode_is_rolling():
    assert WalkForward(5, 5).mode == "rolling"


def test_walk_forward_expanding_first_fold_matches_rolling():
    panel = _panel(n_dates=80, horizon=2)
    rolling = list(WalkForward(20, 10).split(panel))
    expanding = list(WalkForward(20, 10, mode="expanding").split(panel))
    assert len(rolling) == len(expanding)
    np.testing.assert_array_equal(rolling[0][0], expanding[0][0])
    np.testing.assert_array_equal(rolling[0][1], expanding[0][1])


def test_walk_forward_expanding_train_grows_from_first_date():
    # 80 bars, horizon 2 -> 78 panel dates; train=20, purge=2, test=10 gives
    # folds at start = 0, 10, ..., 50 with the k-th training on 20 + 10k dates.
    panel = _panel(n_dates=80, horizon=2)
    folds = list(
        WalkForward(train_periods=20, test_periods=10, mode="expanding").split(panel)
    )
    assert len(folds) == 6
    for k, (train_rows, test_rows) in enumerate(folds):
        train_pos = _date_positions(panel, train_rows)
        test_pos = _date_positions(panel, test_rows)
        assert train_pos.min() == 0  # window pinned at the first date
        assert len(train_pos) == 20 + 10 * k  # grows by step each fold
        # The purge gap to the test block is preserved exactly.
        assert test_pos.min() - train_pos.max() - 1 == panel.horizon


def test_walk_forward_expanding_stays_strictly_causal():
    panel = _panel(n_dates=80, horizon=5)
    for train_rows, test_rows in WalkForward(20, 10, mode="expanding").split(panel):
        train_pos = _date_positions(panel, train_rows)
        test_pos = _date_positions(panel, test_rows)
        assert train_pos.max() < test_pos.min()


def test_walk_forward_invalid_mode_raises():
    with pytest.raises(ModelError, match="mode"):
        WalkForward(5, 5, mode="growing")
