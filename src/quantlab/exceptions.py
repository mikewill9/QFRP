"""Exception taxonomy for quantlab.

Every error raised by the framework derives from :class:`QuantLabError`, so
callers can catch framework failures with a single ``except`` clause while
still discriminating by subsystem when they need to.
"""

from __future__ import annotations

__all__ = [
    "BacktestError",
    "CacheError",
    "ConfigError",
    "DataError",
    "DataValidationError",
    "ExperimentError",
    "ModelError",
    "OptimizationError",
    "QuantLabError",
    "RiskLimitError",
]


class QuantLabError(Exception):
    """Base class for all quantlab exceptions."""


class ConfigError(QuantLabError):
    """Invalid or missing configuration."""


class DataError(QuantLabError):
    """Problems fetching, storing, or shaping market data."""


class DataValidationError(DataError):
    """A dataset failed validation with one or more ERROR-severity issues."""


class CacheError(DataError):
    """Local cache is corrupt or could not be read/written."""


class BacktestError(QuantLabError):
    """Backtest engine misconfiguration or runtime failure."""


class OptimizationError(QuantLabError):
    """Portfolio optimizer failed to converge or received bad inputs."""


class RiskLimitError(QuantLabError):
    """A hard risk limit was breached and configured to raise."""


class ModelError(QuantLabError):
    """ML model training, prediction, or persistence failure."""


class ExperimentError(QuantLabError):
    """Experiment tracking registry failure."""
