"""Walk-forward trainer: causality of the OOS signal, planted-alpha recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import (
    LinearModel,
    PanelDataset,
    PurgedKFold,
    WalkForward,
    WalkForwardTrainer,
)


def _planted_panel(
    n_dates=120, symbols=("A", "B", "C", "D", "E"), beta=0.3, noise=0.002, seed=5
) -> tuple[PanelDataset, pd.DataFrame]:
    """Prices whose next-bar log return is ``beta * factor_t`` plus noise.

    The factor at date t is known at t; the return it drives happens over
    (t, t+1]. A causal learner can recover beta; a leaky one would score
    perfectly, which is what the causality assertions guard against.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    factor = pd.DataFrame(
        rng.normal(size=(n_dates, len(symbols))), index=idx, columns=list(symbols)
    )
    drivers = beta * factor.to_numpy() + rng.normal(0, noise, factor.shape)
    log_prices = np.log(100.0) + np.vstack(
        [np.zeros(len(symbols)), np.cumsum(drivers[:-1], axis=0)]
    )
    prices = pd.DataFrame(np.exp(log_prices), index=idx, columns=list(symbols))
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=1)
    return panel, prices


def test_trainer_recovers_planted_alpha_out_of_sample():
    panel, _ = _planted_panel()
    trainer = WalkForwardTrainer(
        lambda: LinearModel(alpha=0.0), WalkForward(train_periods=40, test_periods=20)
    )
    signal = trainer.run(panel)
    stacked = signal.predictions.stack(future_stack=True).dropna()
    realized = panel.y.reindex(stacked.index)
    corr = float(np.corrcoef(stacked.to_numpy(), realized.to_numpy())[0, 1])
    assert corr > 0.9  # OOS predictions track realized forward returns


def test_trainer_folds_are_strictly_causal():
    panel, _ = _planted_panel()
    signal = WalkForwardTrainer(
        lambda: LinearModel(), WalkForward(train_periods=40, test_periods=20)
    ).run(panel)
    for _, fold in signal.folds.iterrows():
        assert fold["train_end"] < fold["test_start"]
    assert signal.n_folds == len(signal.models) > 1


def test_predictions_only_on_test_dates():
    panel, _ = _planted_panel()
    train_periods = 40
    signal = WalkForwardTrainer(
        lambda: LinearModel(), WalkForward(train_periods, test_periods=20)
    ).run(panel)
    first_possible = panel.dates[train_periods + panel.horizon]
    assert signal.predictions.index.min() >= first_possible
    test_dates = set()
    for _, fold in signal.folds.iterrows():
        test_dates |= set(pd.bdate_range(fold["test_start"], fold["test_end"]))
    assert set(signal.predictions.index) <= test_dates


def test_overlapping_test_blocks_keep_later_fold():
    panel, _ = _planted_panel()
    signal = WalkForwardTrainer(
        lambda: LinearModel(), WalkForward(train_periods=40, test_periods=20, step=10)
    ).run(panel)
    assert signal.predictions.index.is_unique
    assert not signal.predictions.dropna(how="all").empty


def test_trainer_accepts_purged_kfold_for_model_selection():
    panel, _ = _planted_panel(n_dates=80)
    signal = WalkForwardTrainer(
        lambda: LinearModel(), PurgedKFold(n_splits=4, horizon=1)
    ).run(panel)
    assert signal.n_folds == 4
    # K-fold predicts every date exactly once.
    assert len(signal.predictions.index) == len(panel.dates)


def test_trainer_metadata_round_trip():
    panel, _ = _planted_panel(n_dates=80)
    signal = WalkForwardTrainer(
        lambda: LinearModel(), WalkForward(train_periods=30, test_periods=15)
    ).run(panel)
    assert signal.horizon == panel.horizon
    assert signal.feature_names == ["f"]
    assert list(signal.folds.columns) == [
        "train_start", "train_end", "test_start", "test_end", "n_train", "n_test",
    ]


def test_trainer_rejects_non_callable_factory():
    with pytest.raises(ModelError, match="callable"):
        WalkForwardTrainer(LinearModel(), WalkForward(10, 5))  # type: ignore[arg-type]
