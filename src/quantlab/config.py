"""Configuration management.

Settings are resolved from three layers, later layers overriding earlier ones:

1. Built-in defaults (see :class:`Settings`).
2. A TOML file: the first of ``$QUANTLAB_CONFIG``, ``./quantlab.toml``,
   ``~/.config/quantlab/config.toml`` that exists. Keys live under a
   ``[quantlab]`` table (or at the top level).
3. Environment variables ``QUANTLAB_<FIELD>`` (e.g. ``QUANTLAB_CACHE_DIR``).

``get_settings()`` returns a process-wide cached instance; call
``reset_settings()`` in tests to force re-resolution.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from quantlab.exceptions import ConfigError

__all__ = ["Settings", "get_settings", "reset_settings"]

_ENV_PREFIX = "QUANTLAB_"
_CONFIG_ENV_VAR = "QUANTLAB_CONFIG"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable framework settings.

    Attributes
    ----------
    data_dir:
        Root for downloaded/raw data artifacts.
    cache_dir:
        Root for the parquet market-data cache.
    results_dir:
        Root for experiment outputs, model registries, and reports.
    log_level:
        Default logging level name for :func:`quantlab.log.configure_logging`.
    random_seed:
        Default seed used by stochastic components when none is supplied.
    """

    data_dir: Path = Path.home() / ".quantlab" / "data"
    cache_dir: Path = Path.home() / ".quantlab" / "cache"
    results_dir: Path = Path.home() / ".quantlab" / "results"
    log_level: str = "INFO"
    random_seed: int = 7

    def ensure_dirs(self) -> Settings:
        """Create all configured directories (idempotent) and return self."""
        for d in (self.data_dir, self.cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def with_root(self, root: Path | str) -> Settings:
        """Return a copy with ``data/cache/results`` nested under ``root``.

        Convenient for tests and sandboxed runs::

            settings = Settings().with_root(tmp_path).ensure_dirs()
        """
        root = Path(root)
        return replace(
            self,
            data_dir=root / "data",
            cache_dir=root / "cache",
            results_dir=root / "results",
        )


def _find_config_file() -> Path | None:
    explicit = os.environ.get(_CONFIG_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"{_CONFIG_ENV_VAR} points to missing file: {path}")
        return path
    for candidate in (
        Path.cwd() / "quantlab.toml",
        Path.home() / ".config" / "quantlab" / "config.toml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_file_overrides(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    table = raw.get("quantlab", raw)
    if not isinstance(table, dict):
        raise ConfigError(f"[quantlab] table in {path} must be a mapping")
    return table


def _coerce(name: str, value: Any, target_type: type) -> Any:
    try:
        if target_type is Path:
            return Path(str(value)).expanduser()
        if target_type is int:
            return int(value)
        if target_type is str:
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Setting '{name}' has invalid value {value!r}") from exc
    return value


def _resolve() -> Settings:
    values: dict[str, Any] = {}
    file_path = _find_config_file()
    file_overrides = _load_file_overrides(file_path) if file_path else {}

    for f in fields(Settings):
        target_type = {"data_dir": Path, "cache_dir": Path, "results_dir": Path}.get(f.name, type(getattr(Settings(), f.name)))
        if f.name in file_overrides:
            values[f.name] = _coerce(f.name, file_overrides[f.name], target_type)
        env_val = os.environ.get(_ENV_PREFIX + f.name.upper())
        if env_val is not None:
            values[f.name] = _coerce(f.name, env_val, target_type)

    unknown = set(file_overrides) - {f.name for f in fields(Settings)}
    if unknown:
        raise ConfigError(f"Unknown settings in {file_path}: {sorted(unknown)}")
    return Settings(**values)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` (resolved once, then cached)."""
    global _settings
    if _settings is None:
        _settings = _resolve()
    return _settings


def reset_settings() -> None:
    """Discard the cached settings; the next ``get_settings()`` re-resolves."""
    global _settings
    _settings = None
