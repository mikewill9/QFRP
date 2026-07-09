"""Combined one-page performance/risk tearsheet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quantlab.viz.performance import (
    plot_equity_and_drawdown,
    plot_monthly_heatmap,
    plot_rolling_stats,
)
from quantlab.viz.portfolio import plot_weights

if TYPE_CHECKING:
    from quantlab.backtest.results import BacktestResult
    from quantlab.risk.report import RiskReport

__all__ = ["tearsheet"]


def tearsheet(result: BacktestResult, risk_report: RiskReport | None = None) -> Figure:
    """One-page summary: equity/drawdown, rolling stats, monthly heatmap,
    position weights, and a headline performance table.

    ``risk_report`` is optional — a plain backtest result tearsheets
    fine on its own; passing a :class:`~quantlab.risk.report.RiskReport`
    appends a risk-metrics section to the summary text without making
    ``quantlab.risk`` a hard dependency of this function's signature.
    """
    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(5, 2, height_ratios=[2, 1, 1.4, 1.4, 1.6], hspace=0.55, wspace=0.3)

    ax_equity = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_equity)
    plot_equity_and_drawdown(result.returns, axes=(ax_equity, ax_dd))

    ax_roll = fig.add_subplot(gs[2, 0])
    plot_rolling_stats(result.returns, ax=ax_roll)

    ax_heat = fig.add_subplot(gs[2, 1])
    plot_monthly_heatmap(result.returns, ax=ax_heat)

    ax_weights = fig.add_subplot(gs[3, :])
    plot_weights(result.weights, ax=ax_weights)

    ax_text = fig.add_subplot(gs[4, :])
    ax_text.axis("off")
    lines = ["Performance summary", "-" * 21]
    lines += [f"{key:<20} {value:>10.4f}" for key, value in result.summary().items()]
    if risk_report is not None:
        lines += ["", "Risk summary", "-" * 13]
        lines += [f"{key:<24} {value:>10.4f}" for key, value in risk_report.summary.items()]
    ax_text.text(
        0.0, 1.0, "\n".join(lines), transform=ax_text.transAxes,
        va="top", ha="left", family="monospace", fontsize=8,
    )

    fig.suptitle(f"quantlab tearsheet — {result!r}", fontsize=12, fontweight="bold")
    return fig
