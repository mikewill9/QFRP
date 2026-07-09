"""Performance charts: figure/axes contracts, content sanity, no plt.show()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from quantlab.viz.performance import (
    plot_equity_and_drawdown,
    plot_monthly_heatmap,
    plot_rolling_stats,
)


def _returns(n=300, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx)


def test_plot_equity_and_drawdown_returns_figure_with_two_axes():
    fig = plot_equity_and_drawdown(_returns())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_equity_and_drawdown_draws_equity_curve_data():
    r = _returns()
    fig = plot_equity_and_drawdown(r)
    ax_equity, ax_dd = fig.axes
    assert len(ax_equity.lines) == 1
    equity = (1.0 + r).cumprod()
    np.testing.assert_allclose(ax_equity.lines[0].get_ydata(), equity.to_numpy())
    assert len(ax_dd.lines) == 1  # drawdown line (fill_between isn't a Line2D)


def test_plot_equity_and_drawdown_accepts_existing_axes():
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1)
    returned = plot_equity_and_drawdown(_returns(), axes=(ax1, ax2))
    assert returned is fig


def test_plot_rolling_stats_returns_figure_with_twin_axis():
    fig = plot_rolling_stats(_returns(), window=60)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # primary + twinx


def test_plot_rolling_stats_matches_computed_series():
    from quantlab.analytics.performance import rolling_sharpe

    r = _returns()
    fig = plot_rolling_stats(r, window=50)
    sharpe = rolling_sharpe(r, window=50)
    ax = fig.axes[0]
    plotted = ax.lines[0].get_ydata()
    valid = ~np.isnan(sharpe.to_numpy())
    np.testing.assert_allclose(plotted[valid], sharpe.to_numpy()[valid])


def test_plot_monthly_heatmap_returns_figure_with_image():
    fig = plot_monthly_heatmap(_returns(n=500))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.images) == 1


def test_plot_monthly_heatmap_grid_shape_matches_years():
    r = _returns(n=500)
    fig = plot_monthly_heatmap(r)
    ax = fig.axes[0]
    n_years = r.index.year.nunique()
    image_data = ax.images[0].get_array()
    assert image_data.shape == (n_years, 12)


@pytest.mark.parametrize(
    "plot_fn", [plot_equity_and_drawdown, plot_rolling_stats, plot_monthly_heatmap]
)
def test_no_show_calls(plot_fn, monkeypatch):
    import matplotlib.pyplot as plt

    def _fail_show(*args, **kwargs):
        raise AssertionError("plt.show() must never be called by viz functions")

    monkeypatch.setattr(plt, "show", _fail_show)
    plot_fn(_returns())
