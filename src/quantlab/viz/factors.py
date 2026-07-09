"""Factor/signal diagnostic charts: IC, quantile spreads, correlations."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["plot_factor_correlation", "plot_ic", "plot_quantile_returns"]


def plot_ic(ic: pd.Series, rolling_window: int = 21, ax: Axes | None = None) -> Figure:
    """Per-date IC as bars, with a rolling mean overlay.

    Individual dates are noisy; the rolling mean shows whether a
    signal's predictive power is stable, decaying, or regime-dependent.
    """
    s = ic.dropna()
    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = cast(Figure, ax.figure)

    ax.bar(s.index, s.to_numpy(), width=1.0, color="#94a3b8", alpha=0.6, label="Daily IC")
    if len(s) >= rolling_window:
        rolling = s.rolling(rolling_window).mean()
        ax.plot(
            rolling.index, rolling.to_numpy(), color="#2563eb", linewidth=1.5,
            label=f"{rolling_window}-period mean",
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    sd = s.std()
    ir = float(s.mean() / sd) if sd else float("nan")
    ax.set_title(f"Information coefficient (mean={s.mean():.3f}, IR={ir:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    if created_own_fig:
        fig.tight_layout()
    return fig


def plot_quantile_returns(quantile_rets: pd.DataFrame, ax: Axes | None = None) -> Figure:
    """Mean forward return by signal quantile.

    A predictive signal produces a monotonic staircase from quantile 1
    (lowest scores) to the top quantile. Takes the output of
    :func:`quantlab.research.evaluation.quantile_returns`.
    """
    means = quantile_rets.mean()
    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = cast(Figure, ax.figure)

    colors = ["#dc2626" if v < 0 else "#16a34a" for v in means.to_numpy()]
    ax.bar(means.index.astype(str), means.to_numpy(), color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Quantile (1 = lowest score)")
    ax.set_ylabel("Mean forward return")
    ax.set_title("Return by signal quantile")
    if created_own_fig:
        fig.tight_layout()
    return fig


def plot_factor_correlation(
    factors: dict[str, WideFrame], ax: Axes | None = None
) -> Figure:
    """Pairwise correlation of factor scores across all (date, symbol) pairs.

    Highly correlated factors add turnover and estimation noise without
    adding independent information; this heatmap is the first check
    before combining factors into one model.
    """
    if len(factors) < 2:
        raise DataError("Need at least 2 factors to compare")
    stacked = pd.DataFrame(
        {name: frame.stack(future_stack=True) for name, frame in factors.items()}
    )
    corr = stacked.corr()

    created_own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2 * len(corr) + 2, 1.2 * len(corr) + 2))
    else:
        fig = cast(Figure, ax.figure)

    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Factor correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if created_own_fig:
        fig.tight_layout()
    return fig
