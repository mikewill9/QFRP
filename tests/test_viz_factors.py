"""Factor charts: IC, quantile returns, factor correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from quantlab.exceptions import DataError
from quantlab.viz.factors import plot_factor_correlation, plot_ic, plot_quantile_returns


def _ic_series(n=200, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(rng.normal(0.03, 0.1, n), index=idx)


def test_plot_ic_returns_figure_with_bars_and_rolling_line():
    ic = _ic_series()
    fig = plot_ic(ic, rolling_window=21)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == len(ic)  # one bar per date
    assert len(ax.lines) == 2  # rolling mean + zero line


def test_plot_ic_title_reports_mean_and_ir():
    ic = _ic_series()
    fig = plot_ic(ic)
    ax = fig.axes[0]
    assert f"mean={ic.mean():.3f}" in ax.get_title()


def test_plot_ic_skips_rolling_line_when_too_short():
    ic = _ic_series(n=5)
    fig = plot_ic(ic, rolling_window=21)
    ax = fig.axes[0]
    assert len(ax.lines) == 1  # only the zero line


def test_plot_quantile_returns_bar_heights_match_means():
    quantiles = pd.DataFrame(
        {1: [-0.01, -0.02], 2: [0.0, 0.01], 3: [0.02, 0.015], 4: [0.03, 0.025]}
    )
    fig = plot_quantile_returns(quantiles)
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(sorted(heights), sorted(quantiles.mean().to_numpy()))


def test_plot_quantile_returns_colors_negative_bars_differently():
    quantiles = pd.DataFrame({1: [-0.05], 2: [0.05]})
    fig = plot_quantile_returns(quantiles)
    ax = fig.axes[0]
    colors = {tuple(p.get_facecolor()) for p in ax.patches}
    assert len(colors) == 2


def test_plot_factor_correlation_matches_stacked_corr():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-04", periods=100)
    cols = ["A", "B", "C"]
    f1 = pd.DataFrame(rng.normal(size=(100, 3)), index=idx, columns=cols)
    f2 = f1 * 2.0 + rng.normal(0, 0.01, (100, 3))
    factors = {"momentum": f1, "reversal": f2}
    fig = plot_factor_correlation(factors)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.images) == 1
    expected = pd.DataFrame(
        {"momentum": f1.stack(future_stack=True), "reversal": f2.stack(future_stack=True)}
    ).corr()
    np.testing.assert_allclose(ax.images[0].get_array().data, expected.to_numpy(), atol=1e-10)


def test_plot_factor_correlation_requires_at_least_two_factors():
    with pytest.raises(DataError, match="at least 2"):
        plot_factor_correlation({"only_one": pd.DataFrame({"A": [1.0, 2.0]})})
