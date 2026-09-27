"""Tests for app/api.py against small fake results files (no real run needed)."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from app.api import (
    FAIL_ABOVE,
    PASS_BELOW,
    ResultNotComputedError,
    get_analysis,
    list_factors,
    list_periods,
    verdict_for,
)

from quantlab.analytics.performance import performance_summary
from quantlab.backtest.pipeline import PRICE_FACTOR_BUILDERS

FACTOR, PERIOD = "momentum", "2020_2024"


def fake_record(p_value: float = 0.01, **overrides) -> dict:
    """The fields run_permutations.py writes, with tiny made-up values."""
    record = {
        "factor": FACTOR,
        "period": PERIOD,
        "start": "2020-01-01",
        "end": "2024-12-31",
        "base_seed": 1,
        "n_shuffles": 4,
        "p_value": p_value,
        "n_dropped": 1,
        "sharpe_gross": 0.9,
        "null_sharpes_gross": [0.1, -0.2, None, 0.3],
        "sharpe_1x": 0.5,
        "sharpe_2x": 0.3,
        "sharpe_3x": 0.1,
        "total_return_1x": 0.05,
        "max_drawdown_1x": -0.02,
        "hit_rate_1x": 0.5,
        "turnover_annualized_1x": 5.5,
        "turnover_annualized_gross": 5.4,
        "null_turnover_annualized_mean": 11.4,
        "equity_curve_1x": {
            "dates": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
            "values": [1_000_000.0, 1_010_000.0, 1_000_000.0, 1_050_000.0],
        },
        "names_per_year": {"2020": 140.0},
        "settings": {
            "n_tickers": 148,
            "spread_bps_1x": 5.0,
            "commission_rate_1x": 0.005,
            "commission_minimum_1x": 1.0,
            "max_participation": 0.05,
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QFRP_RESULTS_DIR", str(tmp_path))
    return tmp_path


def write(results_dir, record: dict) -> None:
    path = results_dir / f"{record['factor']}_{record['period']}.json"
    path.write_text(json.dumps(record))


# ------------------------------------------------------------------ menus
def test_menus_come_from_the_runner_sources():
    assert [f.key for f in list_factors()] == list(PRICE_FACTOR_BUILDERS)
    assert len(list_factors()) == 11
    assert [p.key for p in list_periods()] == [
        "2005_2024",
        "2005_2009",
        "2010_2014",
        "2015_2019",
        "2020_2024",
    ]
    last = list_periods()[-1]
    assert (last.label, last.start_date, last.end_date) == (
        "2020-2024",
        "2020-01-01",
        "2024-12-31",
    )


# --------------------------------------------------------------- analysis
def test_get_analysis_maps_fields(results_dir):
    write(results_dir, fake_record(p_value=0.2))
    r = get_analysis(FACTOR, PERIOD)
    assert r.sharpe_gross == 0.9
    assert r.sharpe_1x == 0.5
    assert [(c.multiplier, c.sharpe) for c in r.cost_stress] == [(1.0, 0.5), (2.0, 0.3), (3.0, 0.1)]
    assert r.null_sharpes_gross == [0.1, -0.2, 0.3]  # None (NaN) draw removed
    assert (r.n_permutations, r.n_dropped) == (4, 1)
    assert r.turnover == 5.5
    assert r.null_turnover == 11.4
    assert r.p_value == r.adjusted_p_value == 0.2  # k defaults to 1
    assert list(r.equity_curve.columns) == ["strategy"]
    assert r.equity_curve["strategy"].iloc[0] == 1.0
    assert r.equity_curve["strategy"].iloc[-1] == pytest.approx(1.05)
    assert r.limitations


def test_annualized_vol_matches_backtest_convention(results_dir):
    record = fake_record()
    write(results_dir, record)
    curve = record["equity_curve_1x"]
    equity = pd.Series(curve["values"], index=pd.DatetimeIndex(curve["dates"]))
    expected = performance_summary(equity.pct_change().fillna(0.0))["annualized_vol"]
    assert get_analysis(FACTOR, PERIOD).annualized_vol == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p", "k", "expected"),
    [(0.01, 1, 0.01), (0.01, 3, 0.03), (0.02, 10, 0.2), (0.3, 5, 1.0)],
)
def test_adjusted_p_uses_k(results_dir, p, k, expected):
    write(results_dir, fake_record(p_value=p))
    r = get_analysis(FACTOR, PERIOD, k=k)
    assert r.k == k
    assert r.p_value == p
    assert r.adjusted_p_value == pytest.approx(expected)
    v = verdict_for(r, k)
    assert v.raw_p_value == p
    assert v.adjusted_p_value == pytest.approx(expected)
    assert v.n_configs_viewed == k


# ---------------------------------------------------------------- verdict
@pytest.mark.parametrize(
    ("adjusted_p", "label"),
    [
        (0.001, "PASS"),
        (0.0499, "PASS"),
        (0.05, "INCONCLUSIVE"),  # exactly at the PASS cutoff
        (0.10, "INCONCLUSIVE"),
        (0.20, "INCONCLUSIVE"),  # exactly at the FAIL cutoff
        (0.2001, "FAIL"),
        (0.9, "FAIL"),
    ],
)
def test_verdict_bands(results_dir, adjusted_p, label):
    write(results_dir, fake_record())
    r = replace(get_analysis(FACTOR, PERIOD), p_value=adjusted_p)
    assert verdict_for(r, 1).label == label


def test_verdict_cutoffs_reached_through_k(results_dir):
    write(results_dir, fake_record())
    base = get_analysis(FACTOR, PERIOD)
    assert verdict_for(replace(base, p_value=0.024), 2).label == "PASS"  # 0.048
    assert verdict_for(replace(base, p_value=0.025), 2).label == "INCONCLUSIVE"  # 0.05
    assert verdict_for(replace(base, p_value=0.05), 4).label == "INCONCLUSIVE"  # 0.20
    assert verdict_for(replace(base, p_value=0.05), 5).label == "FAIL"  # 0.25


def test_cutoff_constants():
    assert (PASS_BELOW, FAIL_ABOVE) == (0.05, 0.20)


def test_verdict_flags_losing_after_costs(results_dir):
    write(results_dir, fake_record(sharpe_1x=-0.1))
    assert "zero or negative" in verdict_for(get_analysis(FACTOR, PERIOD), 1).explanation
    write(results_dir, fake_record(sharpe_1x=0.4))
    assert "zero or negative" not in verdict_for(get_analysis(FACTOR, PERIOD), 1).explanation


# ------------------------------------------------------------- bad inputs
def test_unknown_factor_or_period(results_dir):
    write(results_dir, fake_record())
    with pytest.raises(KeyError, match="unknown factor_key"):
        get_analysis("not_a_factor", PERIOD)
    with pytest.raises(KeyError, match="unknown period_key"):
        get_analysis(FACTOR, "1990_1995")


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(results_dir, k):
    write(results_dir, fake_record())
    with pytest.raises(ValueError, match="k must be >= 1"):
        get_analysis(FACTOR, PERIOD, k=k)
    with pytest.raises(ValueError, match="n_configs_viewed must be >= 1"):
        verdict_for(get_analysis(FACTOR, PERIOD), k)


@pytest.mark.parametrize("k", [1.5, True, "2"])
def test_non_integer_k_raises(results_dir, k):
    write(results_dir, fake_record())
    with pytest.raises(TypeError, match="integer"):
        get_analysis(FACTOR, PERIOD, k=k)


def test_missing_json_raises_not_computed(results_dir):
    with pytest.raises(ResultNotComputedError, match="not computed yet"):
        get_analysis(FACTOR, PERIOD)
    # still a KeyError, so existing "not found" handling keeps working
    with pytest.raises(KeyError):
        get_analysis(FACTOR, PERIOD)


def test_mismatched_file_contents_raise(results_dir):
    record = fake_record(factor="st_reversal")
    (results_dir / f"{FACTOR}_{PERIOD}.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contains st_reversal"):
        get_analysis(FACTOR, PERIOD)
