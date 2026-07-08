"""Exposures, the limit framework, and the consolidated risk report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.results import BacktestResult
from quantlab.exceptions import DataError, RiskLimitError
from quantlab.risk import (
    DrawdownLimit,
    GrossLeverageLimit,
    PortfolioState,
    PositionLimit,
    RiskMonitor,
    RiskReport,
    VolLimit,
    factor_exposures,
    gross_net_exposure,
    sector_exposures,
)


def _weights_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=3)
    return pd.DataFrame(
        {"A": [0.5, 0.6, 0.4], "B": [-0.2, -0.1, 0.0], "C": [0.0, 0.1, 0.2]},
        index=idx,
    )


# ------------------------------------------------------------ exposures


def test_gross_net_exposure_hand_case():
    out = gross_net_exposure(_weights_frame())
    assert out["gross"].iloc[0] == pytest.approx(0.7)
    assert out["net"].iloc[0] == pytest.approx(0.3)
    assert out["gross"].iloc[2] == pytest.approx(0.6)
    assert out["net"].iloc[2] == pytest.approx(0.6)


def test_gross_net_exposure_treats_nan_as_flat():
    frame = _weights_frame()
    frame.loc[frame.index[0], "A"] = np.nan
    out = gross_net_exposure(frame)
    assert out["gross"].iloc[0] == pytest.approx(0.2)


def test_sector_exposures_hand_case():
    sectors = {"A": "tech", "B": "energy", "C": "tech"}
    out = sector_exposures(_weights_frame(), sectors)
    assert list(out.columns) == ["energy", "tech"]
    assert out["tech"].iloc[1] == pytest.approx(0.7)  # 0.6 + 0.1
    assert out["energy"].iloc[1] == pytest.approx(-0.1)


def test_sector_exposures_rejects_unmapped_held_symbol():
    with pytest.raises(DataError, match="without a sector"):
        sector_exposures(_weights_frame(), {"A": "tech", "B": "energy"})


def test_sector_exposures_ignores_never_held_symbols():
    frame = _weights_frame()
    frame["D"] = 0.0  # never held; needs no mapping
    out = sector_exposures(frame, {"A": "tech", "B": "energy", "C": "tech"})
    assert list(out.columns) == ["energy", "tech"]


def test_factor_exposures_recover_known_loadings():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=400)
    factors = pd.DataFrame(
        {"mkt": rng.normal(0, 0.01, 400), "size": rng.normal(0, 0.005, 400)},
        index=idx,
    )
    returns = 0.0002 + 1.5 * factors["mkt"] - 0.5 * factors["size"]
    out = factor_exposures(returns, factors)
    assert out["alpha"] == pytest.approx(0.0002, abs=1e-12)
    assert out["mkt"] == pytest.approx(1.5)
    assert out["size"] == pytest.approx(-0.5)


def test_factor_exposures_reject_collinear_factors():
    idx = pd.bdate_range("2020-01-01", periods=100)
    base = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    factors = pd.DataFrame({"f1": base, "f2": 2.0 * base})
    with pytest.raises(DataError, match="collinear"):
        factor_exposures(base, factors)


def test_factor_exposures_reject_short_overlap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    factors = pd.DataFrame({"f1": [0.01, 0.02, 0.01], "f2": [0.0, 0.01, 0.02]}, index=idx)
    returns = pd.Series([0.01, 0.0, 0.01], index=idx)
    with pytest.raises(DataError, match="overlapping"):
        factor_exposures(returns, factors)


# ------------------------------------------------------------ limits


def _state(weights: dict[str, float], returns: list[float] | None = None):
    r = None
    if returns is not None:
        r = pd.Series(returns, index=pd.bdate_range("2022-01-03", periods=len(returns)))
    return PortfolioState(weights=pd.Series(weights), returns=r)


def test_gross_leverage_limit():
    limit = GrossLeverageLimit(max_gross=1.0)
    assert limit.check(_state({"A": 0.6, "B": -0.4})) == []
    breaches = limit.check(_state({"A": 0.8, "B": -0.5}))
    assert len(breaches) == 1
    assert breaches[0].value == pytest.approx(1.3)
    assert breaches[0].threshold == 1.0


def test_position_limit_reports_each_offender():
    limit = PositionLimit(max_weight=0.25)
    breaches = limit.check(_state({"A": 0.30, "B": -0.40, "C": 0.10}))
    assert len(breaches) == 2
    assert sorted(b.value for b in breaches) == pytest.approx([0.30, 0.40])


def test_drawdown_limit():
    # Wealth falls 25% from its peak.
    returns = [0.10, -0.25, 0.01]
    assert DrawdownLimit(max_drawdown=0.30).check(_state({"A": 1.0}, returns)) == []
    breaches = DrawdownLimit(max_drawdown=0.20).check(_state({"A": 1.0}, returns))
    assert len(breaches) == 1
    assert breaches[0].value == pytest.approx(1 - 0.75 * 1.01)


def test_history_limits_require_returns():
    with pytest.raises(DataError, match="requires PortfolioState.returns"):
        DrawdownLimit(0.2).check(_state({"A": 1.0}))
    with pytest.raises(DataError, match="requires PortfolioState.returns"):
        VolLimit(0.2).check(_state({"A": 1.0}))


def test_vol_limit_breaches_on_high_vol_and_sleeps_in_warmup():
    calm = [0.001, -0.001] * 40
    wild = [0.05, -0.05] * 40
    limit = VolLimit(max_vol=0.20, window=60)
    assert limit.check(_state({"A": 1.0}, calm)) == []
    assert len(limit.check(_state({"A": 1.0}, wild))) == 1
    # Warmup: fewer returns than the window -> silent.
    assert limit.check(_state({"A": 1.0}, wild[:30])) == []


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (GrossLeverageLimit, {"max_gross": 0.0}),
        (PositionLimit, {"max_weight": -0.1}),
        (DrawdownLimit, {"max_drawdown": 1.5}),
        (VolLimit, {"max_vol": 0.0}),
        (VolLimit, {"max_vol": 0.2, "window": 1}),
    ],
)
def test_limit_parameter_validation(cls, kwargs):
    with pytest.raises(DataError):
        cls(**kwargs)


def test_risk_monitor_collects_across_limits():
    monitor = RiskMonitor([GrossLeverageLimit(1.0), PositionLimit(0.25)])
    breaches = monitor.evaluate(_state({"A": 0.9, "B": -0.5}))
    assert {b.limit for b in breaches} == {"gross_leverage", "position"}


def test_risk_monitor_raises_when_configured():
    monitor = RiskMonitor([GrossLeverageLimit(1.0)], raise_on_breach=True)
    with pytest.raises(RiskLimitError, match="gross_leverage"):
        monitor.evaluate(_state({"A": 1.5}))
    assert monitor.evaluate(_state({"A": 0.5})) == []


# ------------------------------------------------------------ RiskReport


def _fake_result(n=300, seed=9) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    returns = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    equity = 1_000_000 * (1 + returns).cumprod()
    weights = pd.DataFrame(
        {"A": 0.5 + 0.1 * rng.standard_normal(n) * 0.1, "B": -0.2, "C": 0.3},
        index=idx,
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


def test_risk_report_summary_contents():
    report = RiskReport.from_backtest(_fake_result())
    s = report.summary
    assert s["annualized_vol"] > 0
    assert s["var_95%_hist"] > 0
    assert s["es_95%_hist"] >= s["var_95%_hist"]
    assert s["max_drawdown"] <= 0
    assert s["avg_gross_exposure"] == pytest.approx(1.0, abs=0.05)
    assert s["n_limit_breaches"] == 0


def test_risk_report_includes_beta_and_sectors():
    result = _fake_result()
    benchmark = result.returns * 0.5  # true beta of returns on benchmark = 2
    sectors = {"A": "tech", "B": "energy", "C": "tech"}
    report = RiskReport.from_backtest(result, benchmark=benchmark, sectors=sectors)
    assert report.summary["beta"] == pytest.approx(2.0)
    assert report.sectors is not None
    assert list(report.sectors.columns) == ["energy", "tech"]


def test_risk_report_evaluates_limits_on_final_state():
    report = RiskReport.from_backtest(
        _fake_result(), limits=[PositionLimit(max_weight=0.25)]
    )
    # B (-0.2) is fine; A (~0.5) and C (0.3) breach.
    assert report.summary["n_limit_breaches"] == 2
    assert {b.limit for b in report.breaches} == {"position"}


def test_risk_report_text_rendering():
    report = RiskReport.from_backtest(
        _fake_result(),
        sectors={"A": "tech", "B": "energy", "C": "tech"},
        limits=[GrossLeverageLimit(0.5)],
    )
    text = str(report)
    assert "Risk report" in text
    assert "annualized_vol" in text
    assert "Worst drawdowns:" in text
    assert "Limit breaches:" in text
    assert "Final sector exposures:" in text
