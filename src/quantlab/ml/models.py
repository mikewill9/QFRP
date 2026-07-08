"""Model wrappers.

Thin, uniform wrappers around scikit-learn (and optionally LightGBM)
regressors so the trainer, evaluation, and registry never care which
library sits underneath. The shared surface is the
:class:`ModelWrapper` protocol:

- ``fit(X, y) -> self`` with a DataFrame/Series pair,
- ``predict(X) -> np.ndarray``,
- ``feature_importances_`` returning a Series (or ``None`` when the
  model has no natural notion of importance).

``lightgbm`` is an optional extra: :class:`GradientBoostingModel` tries
it *inside* ``fit`` — never at module load — and falls back to
scikit-learn's ``GradientBoostingRegressor``.

Wrappers remember the training feature names and refuse to predict on a
frame with different columns; silently reordering or missing features is
how subtle garbage happens.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quantlab.config import get_settings
from quantlab.exceptions import ModelError

__all__ = [
    "EnsembleModel",
    "GradientBoostingModel",
    "LinearModel",
    "ModelWrapper",
    "TreeModel",
]


@runtime_checkable
class ModelWrapper(Protocol):
    """Structural type every model in the ML layer satisfies."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelWrapper: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @property
    def feature_importances_(self) -> pd.Series | None: ...


def _resolve_seed(random_state: int | None) -> int:
    return get_settings().random_seed if random_state is None else random_state


class _BaseWrapper:
    """Shared fit/predict plumbing and feature-name bookkeeping."""

    def __init__(self) -> None:
        self._estimator: Any = None
        self._feature_names: list[str] | None = None

    def _make_estimator(self) -> Any:
        raise NotImplementedError

    def _check_xy(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0:
            raise ModelError("Cannot fit on an empty feature matrix")
        if len(X) != len(y):
            raise ModelError(f"X has {len(X)} rows but y has {len(y)}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _BaseWrapper:
        self._check_xy(X, y)
        self._feature_names = [str(c) for c in X.columns]
        self._estimator = self._make_estimator()
        self._estimator.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._estimator is None or self._feature_names is None:
            raise ModelError(f"{type(self).__name__} is not fitted")
        if [str(c) for c in X.columns] != self._feature_names:
            raise ModelError(
                f"Prediction features {list(X.columns)[:5]} do not match "
                f"training features {self._feature_names[:5]}"
            )
        return np.asarray(self._estimator.predict(X.to_numpy(dtype=float)), dtype=float)

    @property
    def feature_importances_(self) -> pd.Series | None:
        if self._estimator is None or self._feature_names is None:
            raise ModelError(f"{type(self).__name__} is not fitted")
        raw = getattr(self._estimator, "feature_importances_", None)
        if raw is None:
            return None
        return pd.Series(np.asarray(raw, dtype=float), index=self._feature_names)


class LinearModel(_BaseWrapper):
    """Ridge regression (``alpha=0`` is plain least squares).

    ``feature_importances_`` reports ``|coefficient|`` — meaningful only
    when features share a scale (factor pipelines usually z-score them).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        if alpha < 0:
            raise ModelError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.alpha)

    @property
    def coef_(self) -> pd.Series:
        """Fitted coefficients by feature name."""
        if self._estimator is None or self._feature_names is None:
            raise ModelError("LinearModel is not fitted")
        return pd.Series(self._estimator.coef_, index=self._feature_names)

    @property
    def feature_importances_(self) -> pd.Series | None:
        return self.coef_.abs()


class TreeModel(_BaseWrapper):
    """Random forest regressor with conservative defaults.

    Financial panels have tiny signal-to-noise; shallow trees and large
    leaves are regularization, not timidity.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 5,
        min_samples_leaf: int = 20,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = _resolve_seed(random_state)
        # Default single-threaded: parallel reduction reorders float sums,
        # so n_jobs != 1 is fast but not bit-exact run to run.
        self.n_jobs = n_jobs

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class GradientBoostingModel(_BaseWrapper):
    """Gradient-boosted trees: LightGBM when installed, else scikit-learn.

    The backend is chosen at ``fit`` time (``backend`` reports which);
    neither library is imported when this module loads.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = _resolve_seed(random_state)
        self._backend: str | None = None

    @property
    def backend(self) -> str | None:
        """``"lightgbm"`` or ``"sklearn"`` once fitted, else ``None``."""
        return self._backend

    def _make_estimator(self) -> Any:
        try:
            from lightgbm import LGBMRegressor

            self._backend = "lightgbm"
            return LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbosity=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self._backend = "sklearn"
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )


class EnsembleModel:
    """Weighted average of other wrappers' predictions.

    Averaging models with uncorrelated errors reduces variance the same
    way diversifying a portfolio does. Weights are normalized to sum to
    one; the default is equal weighting.
    """

    def __init__(
        self, models: list[ModelWrapper], weights: list[float] | None = None
    ) -> None:
        if not models:
            raise ModelError("EnsembleModel needs at least one model")
        if weights is not None:
            if len(weights) != len(models):
                raise ModelError(
                    f"{len(weights)} weights for {len(models)} models"
                )
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ModelError("weights must be non-negative and sum > 0")
            total = float(sum(weights))
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(models)] * len(models)
        self.models = models
        self.weights = weights
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        for model in self.models:
            model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ModelError("EnsembleModel is not fitted")
        stacked = np.stack([model.predict(X) for model in self.models])
        return np.asarray(np.average(stacked, axis=0, weights=self.weights))

    @property
    def feature_importances_(self) -> pd.Series | None:
        if not self._fitted:
            raise ModelError("EnsembleModel is not fitted")
        parts: list[pd.Series] = []
        for model in self.models:
            importances = model.feature_importances_
            if importances is None:
                return None
            parts.append(importances)
        blended = self.weights[0] * parts[0]
        for weight, part in zip(self.weights[1:], parts[1:], strict=True):
            blended = blended + weight * part
        return blended
