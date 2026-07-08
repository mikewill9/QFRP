"""Model registry: round trips, versioning, manifests, error handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import LinearModel, ModelRegistry


def _fitted_model(seed=0) -> tuple[LinearModel, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=50), "f2": rng.normal(size=50)})
    y = 1.5 * X["f1"] - 0.5 * X["f2"]
    return LinearModel(alpha=0.0).fit(X, y), X


def test_save_load_round_trip_predicts_identically(tmp_path):
    model, X = _fitted_model()
    registry = ModelRegistry(tmp_path)
    version = registry.save(model, "alpha", manifest={"note": "test"})
    assert version == "v0001"
    loaded, manifest = registry.load("alpha")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    assert manifest["note"] == "test"
    assert manifest["model_class"] == "LinearModel"
    assert manifest["version"] == "v0001"
    assert "created_utc" in manifest


def test_versions_auto_increment_and_latest_wins(tmp_path):
    registry = ModelRegistry(tmp_path)
    first, _ = _fitted_model(seed=1)
    second, _ = _fitted_model(seed=2)
    assert registry.save(first, "alpha") == "v0001"
    assert registry.save(second, "alpha") == "v0002"
    loaded, manifest = registry.load("alpha")  # no version -> latest
    assert manifest["version"] == "v0002"
    assert loaded.coef_["f1"] == pytest.approx(second.coef_["f1"])


def test_load_specific_version(tmp_path):
    registry = ModelRegistry(tmp_path)
    first, _ = _fitted_model(seed=1)
    registry.save(first, "alpha")
    registry.save(_fitted_model(seed=2)[0], "alpha")
    loaded, manifest = registry.load("alpha", "v0001")
    assert manifest["version"] == "v0001"
    assert loaded.coef_["f1"] == pytest.approx(first.coef_["f1"])


def test_list_reports_all_versions(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(_fitted_model()[0], "alpha")
    registry.save(_fitted_model()[0], "alpha")
    registry.save(_fitted_model()[0], "beta", manifest={"ic": 0.05})
    frame = registry.list()
    assert list(frame.columns) == ["name", "version", "created_utc", "model_class"]
    assert len(frame) == 3
    assert sorted(frame["name"].unique()) == ["alpha", "beta"]


def test_list_empty_registry(tmp_path):
    frame = ModelRegistry(tmp_path / "nowhere").list()
    assert frame.empty
    assert list(frame.columns) == ["name", "version", "created_utc", "model_class"]


def test_incomplete_save_directory_is_ignored(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(_fitted_model()[0], "alpha")
    # Simulate a crashed save: version dir without a manifest.
    (tmp_path / "alpha" / "v0002").mkdir()
    assert registry.save(_fitted_model()[0], "alpha") != "v0002"
    _, manifest = registry.load("alpha")
    assert manifest["version"] == "v0003"
    assert len(registry.list()) == 2  # v0002 never counts


def test_missing_model_and_version_raise(tmp_path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ModelError, match="No model named"):
        registry.load("ghost")
    registry.save(_fitted_model()[0], "alpha")
    with pytest.raises(ModelError, match="no version"):
        registry.load("alpha", "v0099")


@pytest.mark.parametrize("name", ["", "a/b", "a\\b", "a b"])
def test_invalid_names_rejected(tmp_path, name):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ModelError, match="Invalid model name"):
        registry.save(_fitted_model()[0], name)
