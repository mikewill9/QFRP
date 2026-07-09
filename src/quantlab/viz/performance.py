"""Performance charts: equity/drawdown, rolling stats, monthly heatmap."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from quantlab.analytics.performance import rolling_sharpe
from quantlab.risk.drawdown import drawdown_series
from quantlab.risk.metrics import rolling_vol

__all__ = ["plot_equity_and_drawdown", "plot_monthly_heatmap", "plot_rolling_stats"]


def plot_equity_and_drawdown(
    returns: pd.Series, axes: tuple[Axes, Axes] | None = None
) -> Figure:
    """Equity curve on top, drawdown below, sharing the date axis.

    ``axes`` lets callers (:func:`quantlab.viz.tearsheet.tearsheet`)
    place this panel inside a larger figure; without it a fresh 2-row
    figure is created.
    """
    r = returns.dropna()
    created_own_fig = axes is None
    if axes is None:
        fig, (ax_equity, ax_dd) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6),
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        ax_equity, ax_dd = axes
        fig = cast(Figure, ax_equity.figure)

    equity = (1.0 + r).cumprod()
    ax_equity.plot(equity.index, equity.to_numpy(), linewidth=1.5)
    ax_equity.set_title("Equity curve")
    ax_equity.set_ylabel("Growth of 1")

    dd = drawdown_series(r)
    ax_dd.fill_between(dd.index, dd.to_numpy(), 0.0, color="#dc2626", alpha=0.4)
    ax_dd.plot(dd.index, dd.to_numpy(), color="#dc2626", linewidth=0.8)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

    if created_own_fig:
        fig.tight_layout()
    return fig


def plot_rolling_stats(
    returns: pd.Series, window: int = 126, ax: Axes | None = None
) -> Figure:
    """Rolling annualized Sharpe (left axis) and volatility (right axis).

    ``window`` is in bars; overlaying the two makes it easy to tell a
    genuine Sharpe change from one that's just tracking a vol regime.
    """
    r = returns.dropna()
    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = cast(Figure, ax.figure)

    sharpe = rolling_sharpe(r, window=window)
    vol = rolling_vol(r, window=window)

    (line_sharpe,) = ax.plot(
        sharpe.index, sharpe.to_numpy(), color="#2563eb",
        label=f"Rolling Sharpe ({window}d)",
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_ylabel("Sharpe")
    ax.set_title("Rolling risk/return")

    ax2 = ax.twinx()
    (line_vol,) = ax2.plot(
        vol.index, vol.to_numpy(), color="#d97706", alpha=0.7,
        label=f"Rolling vol ({window}d)",
    )
    ax2.set_ylabel("Annualized vol")
    ax2.grid(False)

    handles: list[Line2D] = [line_sharpe, line_vol]
    labels: list[str] = [str(line_sharpe.get_label()), str(line_vol.get_label())]
    ax.legend(handles, labels, loc="upper left", fontsize=8)
    if created_own_fig:
        fig.tight_layout()
    return fig


def plot_monthly_heatmap(returns: pd.Series, ax: Axes | None = None) -> Figure:
    """Calendar heatmap of monthly compounded returns, year x month.

    The classic tearsheet visual: rows are years, columns are months, so
    seasonality and bad stretches are visible at a glance.
    """
    r = returns.dropna()
    monthly = (1.0 + r).resample("BME").prod() - 1.0
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(2.0, 0.4 * len(pivot))))
    else:
        fig = cast(Figure, ax.figure)

    data = pivot.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly returns")
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=mticker.PercentFormatter(1.0))
    if created_own_fig:
        fig.tight_layout()
    return fig
