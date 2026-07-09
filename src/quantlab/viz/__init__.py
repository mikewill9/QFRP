"""Visualization: matplotlib plotting and tearsheets.

All functions accept and return matplotlib objects (an ``Axes`` in,
optionally, a ``Figure`` out); none of them call ``plt.show()``, so
callers decide how and whether to display or save.
"""

from quantlab.viz.factors import plot_factor_correlation, plot_ic, plot_quantile_returns
from quantlab.viz.performance import (
    plot_equity_and_drawdown,
    plot_monthly_heatmap,
    plot_rolling_stats,
)
from quantlab.viz.portfolio import plot_contributions, plot_weights
from quantlab.viz.style import PALETTE, apply_style
from quantlab.viz.tearsheet import tearsheet

__all__ = [
    "PALETTE",
    "apply_style",
    "plot_contributions",
    "plot_equity_and_drawdown",
    "plot_factor_correlation",
    "plot_ic",
    "plot_monthly_heatmap",
    "plot_quantile_returns",
    "plot_rolling_stats",
    "plot_weights",
    "tearsheet",
]
