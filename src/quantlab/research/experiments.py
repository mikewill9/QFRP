"""Experiment tracking.

A lightweight local registry so research runs are reproducible and
comparable without external services. Each run gets a directory

    {root}/{experiment_name}/{run_id}/
        manifest.json      # config, metrics, artifact list, status
        frames/*.parquet   # logged DataFrames

``run_id`` embeds a UTC timestamp plus a short hash of the config, so two
runs of the same configuration are distinguishable yet visibly related.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.config import get_settings
from quantlab.exceptions import ExperimentError

__all__ = ["ExperimentTracker", "Run"]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return str(value)


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(_jsonable(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


@dataclass
class Run:
    """One tracked run. Create via :meth:`ExperimentTracker.start_run`."""

    experiment: str
    run_id: str
    path: Path
    config: dict[str, Any]
    metrics: dict[str, float] = dc_field(default_factory=dict)
    frames: list[str] = dc_field(default_factory=list)
    status: str = "running"
    created_utc: str = ""

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record scalar metrics (later calls overwrite matching keys)."""
        if self.status != "running":
            raise ExperimentError(f"Run {self.run_id} is already {self.status}")
        for key, value in metrics.items():
            self.metrics[str(key)] = float(value)
        self._write_manifest()

    def log_frame(self, name: str, frame: pd.DataFrame) -> Path:
        """Persist a DataFrame artifact under ``frames/{name}.parquet``."""
        if self.status != "running":
            raise ExperimentError(f"Run {self.run_id} is already {self.status}")
        if not name or any(ch in name for ch in "/\\"):
            raise ExperimentError(f"Invalid frame name {name!r}")
        frames_dir = self.path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out = frames_dir / f"{name}.parquet"
        frame.to_parquet(out)
        if name not in self.frames:
            self.frames.append(name)
        self._write_manifest()
        return out

    def load_frame(self, name: str) -> pd.DataFrame:
        path = self.path / "frames" / f"{name}.parquet"
        if not path.exists():
            raise ExperimentError(f"Run {self.run_id} has no frame {name!r}")
        return pd.read_parquet(path)

    def finish(self, status: str = "completed") -> None:
        """Mark the run finished; further logging raises."""
        if status not in ("completed", "failed"):
            raise ExperimentError("status must be 'completed' or 'failed'")
        self.status = status
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "status": self.status,
            "config": _jsonable(self.config),
            "metrics": self.metrics,
            "frames": self.frames,
        }
        tmp = self.path / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp.replace(self.path / "manifest.json")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Run({self.experiment}/{self.run_id}, {self.status}, {len(self.metrics)} metrics)"


class ExperimentTracker:
    """Registry of experiments under a root directory.

    Parameters
    ----------
    root:
        Storage root; defaults to ``{settings.results_dir}/experiments``.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            root = get_settings().results_dir / "experiments"
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str, config: dict[str, Any] | None = None) -> Run:
        """Create a new run directory and return the live :class:`Run`."""
        if not name or any(ch in name for ch in "/\\"):
            raise ExperimentError(f"Invalid experiment name {name!r}")
        config = dict(config or {})
        now = datetime.now(UTC)
        run_id = f"{now.strftime('%Y%m%dT%H%M%S%f')}-{_config_hash(config)}"
        path = self.root / name / run_id
        path.mkdir(parents=True, exist_ok=False)
        run = Run(
            experiment=name,
            run_id=run_id,
            path=path,
            config=config,
            created_utc=now.isoformat(),
        )
        run._write_manifest()
        return run

    def runs(self, name: str | None = None) -> pd.DataFrame:
        """All run manifests as a flat DataFrame (metrics become columns)."""
        rows = []
        experiments = [name] if name else sorted(
            p.name for p in self.root.iterdir() if p.is_dir()
        )
        for exp in experiments:
            exp_dir = self.root / exp
            if not exp_dir.is_dir():
                continue
            for run_dir in sorted(exp_dir.iterdir()):
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                m = json.loads(manifest_path.read_text())
                row = {
                    "experiment": m["experiment"],
                    "run_id": m["run_id"],
                    "created_utc": m["created_utc"],
                    "status": m["status"],
                }
                row.update({f"metric.{k}": v for k, v in m.get("metrics", {}).items()})
                row.update({f"config.{k}": v for k, v in m.get("config", {}).items()
                            if isinstance(v, (str, int, float, bool))})
                rows.append(row)
        if not rows:
            return pd.DataFrame(
                columns=["experiment", "run_id", "created_utc", "status"]
            )
        return pd.DataFrame(rows).sort_values("created_utc").reset_index(drop=True)

    def compare(self, metric: str, name: str | None = None) -> pd.DataFrame:
        """Completed runs sorted by ``metric`` (best first)."""
        frame = self.runs(name)
        col = f"metric.{metric}"
        if col not in frame.columns:
            raise ExperimentError(f"No run has logged metric {metric!r}")
        keep = frame[frame["status"] == "completed"].dropna(subset=[col])
        return keep.sort_values(col, ascending=False).reset_index(drop=True)

    def get_run(self, name: str, run_id: str) -> Run:
        """Rehydrate a finished run for artifact access."""
        path = self.root / name / run_id
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise ExperimentError(f"No run {name}/{run_id} under {self.root}")
        m = json.loads(manifest_path.read_text())
        return Run(
            experiment=m["experiment"],
            run_id=m["run_id"],
            path=path,
            config=m.get("config", {}),
            metrics=m.get("metrics", {}),
            frames=m.get("frames", []),
            status=m.get("status", "completed"),
            created_utc=m.get("created_utc", ""),
        )
