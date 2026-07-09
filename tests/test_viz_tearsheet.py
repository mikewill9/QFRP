"""Tearsheet: composes every panel, optional RiskReport, no plt.show()."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quantlab.backtest.results import BacktestResult
from quantlab.risk import RiskReport
from quantlab.viz import tearsheet


def _result(n=300, seed=0) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    returns = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    equity = 1_000_000 * (1 + returns).cumprod()
    weights = pd.DataFrame(
        {f"S{i}": rng.normal(0, 0.05, n) for i in range(12)}, index=idx
    )
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        weights=weights,
        positions=weights * 0,
        trades=pd.DataFrame(),
        costs_paid=pd.Series(0.0, index=idx),
        initial_capital=1_000_000,
    )


def test_tearsheet_returns_figure_with_all_panels():
    fig = tearsheet(_result())
    assert isinstance(fig, Figure)
    # equity, drawdown, rolling(+twin), heatmap(+colorbar), weights, text = 8 axes
    assert len(fig.axes) == 8


def test_tearsheet_without_risk_report_has_no_risk_section():
    fig = tearsheet(_result())
    text_ax = fig.axes[-1]
    text = text_ax.texts[0].get_text()
    assert "Performance summary" in text
    assert "Risk summary" not in text


def test_tearsheet_with_risk_report_adds_risk_section():
    result = _result()
    report = RiskReport.from_backtest(result)
    fig = tearsheet(result, risk_report=report)
    text_ax = fig.axes[-1]
    text = text_ax.texts[0].get_text()
    assert "Risk summary" in text
    assert "annualized_vol" in text


def test_tearsheet_title_includes_result_repr():
    result = _result()
    fig = tearsheet(result)
    assert repr(result) in fig._suptitle.get_text()


def test_tearsheet_never_calls_show(monkeypatch):
    import matplotlib.pyplot as plt

    def _fail_show(*args, **kwargs):
        raise AssertionError("plt.show() must never be called by viz functions")

    monkeypatch.setattr(plt, "show", _fail_show)
    tearsheet(_result(n=200))
