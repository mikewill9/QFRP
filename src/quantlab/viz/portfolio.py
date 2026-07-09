"""Portfolio composition charts: weights over time, contribution breakdowns."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quantlab.types import WideFrame
from quantlab.viz.style import PALETTE

__all__ = ["plot_contributions", "plot_weights"]


def plot_weights(weights: WideFrame, top_n: int = 10, ax: Axes | None = None) -> Figure:
    """Stacked-area view of position weights over time.

    Books with dozens of names are unreadable stacked in full; the
    ``top_n`` symbols by average absolute weight get their own band and
    everything else is grouped into a single "other" band so the
    picture stays legible without hiding gross/net exposure.
    """
    w = weights.fillna(0.0)
    ranked = w.abs().mean().sort_values(ascending=False)
    kept = list(ranked.index[:top_n])
    remainder = w.drop(columns=kept)
    has_other = remainder.shape[1] > 0

    frame = w[kept].copy()
    if has_other:
        frame["other"] = remainder.sum(axis=1)
    columns = list(frame.columns)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(columns))]

    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = cast(Figure, ax.figure)

    positive = frame.clip(lower=0.0)
    negative = frame.clip(upper=0.0)
    ax.stackplot(frame.index, positive.to_numpy().T, labels=columns, colors=colors, alpha=0.85)
    ax.stackplot(frame.index, negative.to_numpy().T, colors=colors, alpha=0.85)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Position weights")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    if created_own_fig:
        fig.tight_layout()
    return fig


def plot_contributions(
    contributions: WideFrame, top_n: int = 15, ax: Axes | None = None
) -> Figure:
    """Total contribution per column, sorted (top and bottom ``top_n``).

    Works on either
    :func:`quantlab.analytics.attribution.asset_contributions` (per
    symbol) or a :class:`~quantlab.analytics.attribution.FactorAttribution`
    ``.contributions`` frame (alpha/factor/residual) — any per-column
    time series of return contributions. Column sums are the same
    linear-attribution approximation the source functions document.
    """
    totals = contributions.sum().sort_values()
    if len(totals) > 2 * top_n:
        totals = pd.concat([totals.iloc[:top_n], totals.iloc[-top_n:]])

    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.3 * len(totals))))
    else:
        fig = cast(Figure, ax.figure)

    colors = ["#dc2626" if v < 0 else "#16a34a" for v in totals.to_numpy()]
    ax.barh(totals.index.astype(str), totals.to_numpy(), color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Total return contribution")
    ax.set_xlabel("Summed per-period contribution")
    if created_own_fig:
        fig.tight_layout()
    return fig
