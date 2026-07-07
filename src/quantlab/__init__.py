"""quantlab: an institutional-style quantitative research framework.

Subsystems (import from subpackages; the top-level namespace is kept light so
that ``import quantlab`` never drags in matplotlib or scikit-learn):

- ``quantlab.data``       market data ingestion, caching, validation, preprocessing
- ``quantlab.research``   factors, signal transforms, alpha evaluation, experiments
- ``quantlab.backtest``   event-driven backtesting engine
- ``quantlab.portfolio``  covariance estimation, optimizers, rebalancing
- ``quantlab.risk``       exposures, drawdowns, limits, risk reporting
- ``quantlab.ml``         feature pipelines, purged CV, model training/registry
- ``quantlab.analytics``  performance statistics and attribution
- ``quantlab.viz``        plotting and tearsheets
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from quantlab.config import Settings, get_settings
from quantlab.log import configure_logging, get_logger

try:  # pragma: no cover - depends on installation mode
    __version__ = version("quantlab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "Settings",
    "__version__",
    "configure_logging",
    "get_logger",
    "get_settings",
]
