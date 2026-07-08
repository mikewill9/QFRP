"""Model registry.

Versioned on-disk storage so a model used in research (or, one day,
production) can be reloaded exactly, with a manifest recording where it
came from. Layout mirrors the experiment tracker::

    {root}/{name}/v0001/
        model.pkl          # pickled wrapper
        manifest.json      # user manifest + created_utc, model_class, ...

Versions auto-increment per name; ``load`` without a version returns the
latest. Models are serialized with :mod:`pickle` — only load registries
you (or code you trust) wrote, the standard pickle caveat.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.exceptions import ModelError

__all__ = ["ModelRegistry"]

_VERSION_RE = re.compile(r"^v(\d{4})$")
_LIST_COLUMNS = ["name", "version", "created_utc", "model_class"]


class ModelRegistry:
    """Named, versioned model store rooted at ``root``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _check_name(name: str) -> str:
        if not name or not re.fullmatch(r"[A-Za-z0-9_\-\.]+", name):
            raise ModelError(
                f"Invalid model name {name!r}; use letters, digits, '_', '-', '.'"
            )
        return name

    def _version_dirs(self, name: str) -> list[str]:
        """All ``vNNNN`` directories, including crashed/incomplete saves."""
        model_dir = self.root / name
        if not model_dir.is_dir():
            return []
        return sorted(
            child.name
            for child in model_dir.iterdir()
            if child.is_dir() and _VERSION_RE.match(child.name)
        )

    def _versions(self, name: str) -> list[str]:
        """Complete (manifest-bearing) versions only."""
        return [
            v
            for v in self._version_dirs(name)
            if (self.root / name / v / "manifest.json").is_file()
        ]

    # ------------------------------------------------------------ API

    def save(
        self, model: Any, name: str, manifest: dict[str, Any] | None = None
    ) -> str:
        """Store ``model`` under ``name``; returns the new version id."""
        self._check_name(name)
        # Number past every existing directory (even incomplete ones), so a
        # crashed save can never block or be silently overwritten.
        dirs = self._version_dirs(name)
        next_number = int(dirs[-1][1:]) + 1 if dirs else 1
        version = f"v{next_number:04d}"
        target = self.root / name / version
        target.mkdir(parents=True, exist_ok=False)

        with (target / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh)
        record = {
            **(manifest or {}),
            "created_utc": datetime.now(UTC).isoformat(),
            "model_class": type(model).__name__,
            "name": name,
            "version": version,
        }
        # Manifest is written last: a directory without one is treated as
        # a failed save and ignored by _versions/list/load.
        (target / "manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str)
        )
        return version

    def load(self, name: str, version: str | None = None) -> tuple[Any, dict[str, Any]]:
        """Return ``(model, manifest)``; latest version when unspecified."""
        self._check_name(name)
        versions = self._versions(name)
        if not versions:
            raise ModelError(f"No model named {name!r} in registry at {self.root}")
        if version is None:
            version = versions[-1]
        elif version not in versions:
            raise ModelError(
                f"Model {name!r} has no version {version!r}; available: {versions}"
            )
        target = self.root / name / version
        with (target / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        manifest = json.loads((target / "manifest.json").read_text())
        return model, manifest

    def list(self) -> pd.DataFrame:
        """All stored models: one row per (name, version), newest last."""
        rows = []
        if self.root.is_dir():
            for model_dir in sorted(self.root.iterdir()):
                if not model_dir.is_dir():
                    continue
                for version in self._versions(model_dir.name):
                    manifest = json.loads(
                        (model_dir / version / "manifest.json").read_text()
                    )
                    rows.append({c: manifest.get(c) for c in _LIST_COLUMNS})
        return pd.DataFrame(rows, columns=_LIST_COLUMNS)
