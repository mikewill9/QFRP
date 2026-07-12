"""Model selection with leak-proof cross-validation.

Compares candidate models on identical purged folds. This module is for
SELECTION ONLY: its scores are interior-fold numbers and must never be
quoted as out-of-sample results — produce those with WalkForwardTrainer
on the single chosen configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from quantlab.exceptions import ModelError
from quantlab.ml.datasets import PanelDataset
from quantlab.ml.evaluation import regression_metrics
from quantlab.ml.models import ModelWrapper
from quantlab.ml.validation import PurgedKFold, WalkForward

__all__ = ["compare_models"]

_MIN_NAMES_PER_DATE = 3


def compare_models(
    panel: PanelDataset,
    model_factories: Mapping[str, Callable[[], ModelWrapper]],
    splitter: PurgedKFold | WalkForward,
) -> pd.DataFrame:
    """Fit each candidate per fold; return one row per model.

    Columns: ``mean_r2``, ``std_r2``, ``mean_mse``, ``mean_rank_ic``
    (Spearman correlation between fold predictions and fold targets,
    computed per test date and averaged), ``n_folds``. Index = model name,
    sorted by ``mean_rank_ic`` descending. A fresh model is constructed
    per (model, fold) so folds never share fitted state.

    Every candidate sees the identical folds, so scores are directly
    comparable; dates with fewer than three test names are skipped when
    averaging the rank IC (a Spearman correlation on two points is noise).
    """
    if not model_factories:
        raise ModelError("model_factories is empty; provide at least one candidate")

    folds = list(splitter.split(panel))
    rows: list[pd.Series] = []
    for name, factory in model_factories.items():
        r2s: list[float] = []
        mses: list[float] = []
        fold_ics: list[float] = []
        for train_rows, test_rows in folds:
            model = factory()
            model.fit(panel.X.iloc[train_rows], panel.y.iloc[train_rows])
            X_test = panel.X.iloc[test_rows]
            y_test = panel.y.iloc[test_rows]
            preds = pd.Series(model.predict(X_test), index=X_test.index)
            metrics = regression_metrics(y_test, preds)
            r2s.append(float(metrics["r2"]))
            mses.append(float(metrics["mse"]))

            date_ics: list[float] = []
            test_dates = panel.groups.iloc[test_rows]
            for _, rows_of_date in test_dates.groupby(test_dates).groups.items():
                if len(rows_of_date) < _MIN_NAMES_PER_DATE:
                    continue
                p = preds.loc[rows_of_date]
                t = y_test.loc[rows_of_date]
                # Constant predictions (or targets) have no rank ordering;
                # the correlation is undefined, not zero.
                if p.nunique() <= 1 or t.nunique() <= 1:
                    continue
                ic = p.corr(t, method="spearman")
                if not np.isnan(ic):
                    date_ics.append(float(ic))
            fold_ics.append(float(np.mean(date_ics)) if date_ics else np.nan)

        rows.append(
            pd.Series(
                {
                    "mean_r2": float(np.mean(r2s)),
                    "std_r2": float(np.std(r2s)),
                    "mean_mse": float(np.mean(mses)),
                    "mean_rank_ic": float(np.nanmean(fold_ics))
                    if not all(np.isnan(ic) for ic in fold_ics)
                    else float("nan"),
                    "n_folds": float(len(folds)),
                },
                name=name,
            )
        )

    table = pd.DataFrame(rows)
    table.index.name = "model"
    return table.sort_values("mean_rank_ic", ascending=False)
