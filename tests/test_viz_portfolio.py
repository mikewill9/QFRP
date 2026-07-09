"""Portfolio charts: weight bands, contribution bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quantlab.viz.portfolio import plot_contributions, plot_weights


def _weights(n_symbols=15, n_dates=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_dates)
    cols = [f"S{i}" for i in range(n_symbols)]
    return pd.DataFrame(rng.normal(0, 0.05, (n_dates, n_symbols)), index=idx, columns=cols)


def test_plot_weights_groups_small_positions_into_other():
    w = _weights(n_symbols=15)
    fig = plot_weights(w, top_n=5)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "other" in legend_labels
    assert len(legend_labels) == 6  # top 5 + other


def test_plot_weights_keeps_all_symbols_when_under_top_n():
    w = _weights(n_symbols=4)
    fig = plot_weights(w, top_n=10)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "other" not in legend_labels
    assert set(legend_labels) == set(w.columns)


def test_plot_weights_ranks_by_average_absolute_weight():
    idx = pd.bdate_range("2021-01-04", periods=10)
    w = pd.DataFrame(
        {"big": np.full(10, 0.5), "small": np.full(10, 0.01), "medium": np.full(10, 0.2)},
        index=idx,
    )
    fig = plot_weights(w, top_n=2)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(legend_labels) == {"big", "medium", "other"}


def test_plot_weights_returns_figure():
    assert isinstance(plot_weights(_weights()), Figure)


def test_plot_contributions_signs_and_ordering():
    contributions = pd.DataFrame(
        {"A": [0.02, 0.01], "B": [-0.03, -0.01], "C": [0.001, 0.002]}
    )
    fig = plot_contributions(contributions)
    ax = fig.axes[0]
    totals = contributions.sum().sort_values()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == list(totals.index.astype(str))
    widths = [p.get_width() for p in ax.patches]
    np.testing.assert_allclose(widths, totals.to_numpy())


def test_plot_contributions_truncates_to_top_and_bottom():
    n = 40
    contributions = pd.DataFrame(
        {f"f{i}": [float(i - n // 2)] for i in range(n)}
    )
    fig = plot_contributions(contributions, top_n=5)
    ax = fig.axes[0]
    assert len(ax.patches) == 10  # 5 lowest + 5 highest
