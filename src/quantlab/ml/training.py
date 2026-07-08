"""Walk-forward training.

:class:`WalkForwardTrainer` turns a model factory plus a splitter into a
wide out-of-sample prediction frame: for every fold it fits a *fresh*
model on the training rows and predicts the test rows, so each
prediction comes from a model that never saw that date (or, with
:class:`~quantlab.ml.validation.WalkForward`, any later date).

The resulting :class:`TrainedSignal.predictions` frame follows the
framework's signal convention — value at date ``t``, higher = more
attractive — and can go straight into ``signal_to_weights`` or a
:class:`~quantlab.backtest.strategy.TargetWeightStrategy`. Only
predictions produced by a :class:`WalkForward` splitter are causal
enough to backtest; purged k-fold predictions are for model selection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from quantlab.exceptions import ModelError
from quantlab.ml.datasets import PanelDataset
from quantlab.ml.models import ModelWrapper
from quantlab.ml.validation import PurgedKFold, WalkForward
from quantlab.types import WideFrame

__all__ = ["TrainedSignal", "WalkForwardTrainer"]


@dataclass(frozen=True, slots=True)
class TrainedSignal:
    """Out-of-sample predictions plus everything needed to audit them.

    Attributes
    ----------
    predictions:
        Wide frame (dates x symbols) of OOS predictions; NaN where no
        fold predicted (warmup dates, purge gaps, missing panel rows).
    folds:
        One row per fold: train/test date ranges and row counts.
    models:
        The fitted model of each fold (same order as ``folds``).
    horizon:
        Label horizon of the panel the signal was trained on.
    feature_names:
        Features the models were fitted with.
    """

    predictions: WideFrame
    folds: pd.DataFrame
    models: list[ModelWrapper] = field(repr=False)
    horizon: int
    feature_names: list[str]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TrainedSignal({self.n_folds} folds, "
            f"{self.predictions.notna().any(axis=1).sum()} predicted dates, "
            f"horizon={self.horizon})"
        )


class WalkForwardTrainer:
    """Fit one fresh model per fold and assemble the OOS signal.

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning an *unfitted* model wrapper;
        called once per fold so folds never share fitted state.
    splitter:
        :class:`WalkForward` (causal signal) or :class:`PurgedKFold`
        (model selection). Anything with the same ``split(panel)``
        surface also works.
    """

    def __init__(
        self,
        model_factory: Callable[[], ModelWrapper],
        splitter: WalkForward | PurgedKFold,
    ) -> None:
        if not callable(model_factory):
            raise ModelError("model_factory must be callable")
        self.model_factory = model_factory
        self.splitter = splitter

    def run(self, panel: PanelDataset) -> TrainedSignal:
        """Train across all folds and return the assembled signal."""
        prediction_parts: list[pd.Series] = []
        fold_rows: list[dict[str, object]] = []
        models: list[ModelWrapper] = []

        for train_rows, test_rows in self.splitter.split(panel):
            x_train = panel.X.iloc[train_rows]
            y_train = panel.y.iloc[train_rows]
            x_test = panel.X.iloc[test_rows]
            model = self.model_factory()
            model.fit(x_train, y_train)
            preds = pd.Series(model.predict(x_test), index=x_test.index)

            train_dates = panel.groups.iloc[train_rows]
            test_dates = panel.groups.iloc[test_rows]
            fold_rows.append(
                {
                    "train_start": train_dates.min(),
                    "train_end": train_dates.max(),
                    "test_start": test_dates.min(),
                    "test_end": test_dates.max(),
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                }
            )
            models.append(model)
            prediction_parts.append(preds)

        long_preds = pd.concat(prediction_parts)
        # Overlapping test blocks (step < test_periods) predict some rows
        # twice; keep the later fold's value (trained on more recent data).
        long_preds = long_preds[~long_preds.index.duplicated(keep="last")]
        predictions = long_preds.unstack("symbol").sort_index()

        return TrainedSignal(
            predictions=predictions,
            folds=pd.DataFrame(fold_rows),
            models=models,
            horizon=panel.horizon,
            feature_names=panel.feature_names,
        )
