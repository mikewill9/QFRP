"""Experiment tracker: run lifecycle, persistence, comparison."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantlab.exceptions import ExperimentError
from quantlab.research import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(root=tmp_path)


def test_run_lifecycle_and_manifest(tracker):
    run = tracker.start_run("momentum_sweep", {"lookback": 126, "skip": 21})
    run.log_metrics({"ic_mean": 0.031, "sharpe": 1.2})
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    run.log_frame("quantiles", frame)
    run.finish()

    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["config"]["lookback"] == 126
    assert manifest["metrics"]["sharpe"] == 1.2
    assert manifest["frames"] == ["quantiles"]

    reloaded = tracker.get_run("momentum_sweep", run.run_id)
    pd.testing.assert_frame_equal(reloaded.load_frame("quantiles"), frame)


def test_finished_run_rejects_logging(tracker):
    run = tracker.start_run("exp", {})
    run.finish()
    with pytest.raises(ExperimentError, match="already completed"):
        run.log_metrics({"x": 1.0})
    with pytest.raises(ExperimentError, match="already completed"):
        run.log_frame("f", pd.DataFrame())


def test_runs_table_and_compare(tracker):
    for lb, ic in ((63, 0.01), (126, 0.04), (252, 0.02)):
        run = tracker.start_run("sweep", {"lookback": lb})
        run.log_metrics({"ic_mean": ic})
        run.finish()
    failed = tracker.start_run("sweep", {"lookback": 5})
    failed.log_metrics({"ic_mean": 9.9})
    failed.finish("failed")

    table = tracker.runs("sweep")
    assert len(table) == 4
    assert "metric.ic_mean" in table.columns
    assert "config.lookback" in table.columns

    best = tracker.compare("ic_mean", "sweep")
    assert len(best) == 3  # failed run excluded
    assert best.iloc[0]["config.lookback"] == 126

    with pytest.raises(ExperimentError, match="No run has logged"):
        tracker.compare("nonexistent")


def test_same_config_distinct_run_ids_share_hash(tracker):
    a = tracker.start_run("exp", {"x": 1})
    b = tracker.start_run("exp", {"x": 1})
    c = tracker.start_run("exp", {"x": 2})
    assert a.run_id != b.run_id
    assert a.run_id.split("-")[-1] == b.run_id.split("-")[-1]
    assert a.run_id.split("-")[-1] != c.run_id.split("-")[-1]


def test_invalid_names_rejected(tracker):
    with pytest.raises(ExperimentError):
        tracker.start_run("bad/name")
    run = tracker.start_run("ok")
    with pytest.raises(ExperimentError):
        run.log_frame("../escape", pd.DataFrame())


def test_empty_tracker_runs_table(tracker):
    table = tracker.runs()
    assert table.empty
    assert "run_id" in table.columns
