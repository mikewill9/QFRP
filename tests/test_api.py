"""Tests for app/api.py against small fake results files (no real run needed)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from app import api
from app.api import (
    EXCLUDED_FACTORS,
    FAIL_ABOVE,
    PASS_BELOW,
    ResultNotComputedError,
    get_analysis,
    list_factors,
    list_periods,
    verdict_for,
)

from quantlab.analytics.performance import performance_summary
from quantlab.analytics.significance import permutation_pvalue
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
    assert EXCLUDED_FACTORS == ("roll_spread",)
    expected = [k for k in PRICE_FACTOR_BUILDERS if k not in EXCLUDED_FACTORS]
    assert [f.key for f in list_factors()] == expected
    assert len(list_factors()) == 10
    assert "roll_spread" in PRICE_FACTOR_BUILDERS  # hidden, not deleted
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


# ----------------------------------------------------------- verdict text
ONLY_TEST = " If this were your only test, the verdict would be"
COST_WARNING = " After realistic trading costs, this strategy lost money."


def explain(results_dir, p_value, k, **overrides):
    write(results_dir, fake_record(**overrides))
    return verdict_for(replace(get_analysis(FACTOR, PERIOD), p_value=p_value), k).explanation


def test_pass_text_single_test(results_dir):
    # fake nulls (NaN dropped): 0.1, -0.2, 0.3 -- all below sharpe_gross 0.9
    assert explain(results_dir, 0.01, 1) == (
        "This strategy beat 100% of 4 random versions of itself. Even counting "
        "the 1 test you've run this session, luck alone would rarely produce a "
        "result this strong. That's evidence the signal is real, before trading "
        "costs."
    )


def test_pass_share_counts_nulls_strictly_below_and_rounds_down(results_dir):
    # vs sharpe_gross 0.9: 2 of 4 below (a tie at 0.9 doesn't count) -> "50"
    text = explain(results_dir, 0.01, 1, null_sharpes_gross=[0.1, 0.9, 1.0, -0.5])
    assert text.startswith("This strategy beat 50% of 4 random versions")
    # 2 of 3 below -> 66.67% rounded DOWN -> "66.6"
    text = explain(results_dir, 0.01, 1, null_sharpes_gross=[0.1, 1.0, -0.5])
    assert text.startswith("This strategy beat 66.6% of 4 random versions")


def test_pass_text_with_k(results_dir):
    text = explain(results_dir, 0.01, 3)  # adjusted 0.03 -> PASS
    assert "Even counting the 3 tests you've run this session" in text
    assert text.endswith(f"{ONLY_TEST} PASS.")


def test_inconclusive_text(results_dir):
    assert explain(results_dir, 0.10, 1) == (
        "This strategy did better than most random versions, but not by enough "
        "to rule out luck given the 1 test you've run. It might be real. This "
        "data can't tell."
    )
    text = explain(results_dir, 0.03, 2)  # adjusted 0.06 -> INCONCLUSIVE
    assert "given the 2 tests you've run." in text
    assert text.endswith(f"{ONLY_TEST} PASS.")  # 0.03 on its own passes


def test_fail_text(results_dir):
    assert explain(results_dir, 0.5, 1) == (
        "Random versions of this strategy did about as well or better. There's "
        "no evidence the signal is anything more than luck."
    )
    assert explain(results_dir, 0.15, 2).endswith(f"{ONLY_TEST} INCONCLUSIVE.")  # 0.30
    assert explain(results_dir, 0.5, 3).endswith(f"{ONLY_TEST} FAIL.")


def test_only_test_line_appears_only_when_k_above_one(results_dir):
    assert ONLY_TEST not in explain(results_dir, 0.01, 1)
    assert ONLY_TEST not in explain(results_dir, 0.5, 1)
    assert ONLY_TEST in explain(results_dir, 0.5, 2)


@pytest.mark.parametrize(
    ("sharpe_1x", "warned"), [(-0.1, True), (0.0, True), (0.4, False), (None, False)]
)
def test_cost_warning(results_dir, sharpe_1x, warned):
    for p_value in (0.01, 0.10, 0.5):  # added to every verdict label
        assert explain(results_dir, p_value, 1, sharpe_1x=sharpe_1x).endswith(COST_WARNING) == warned


def test_cost_warning_comes_after_only_test_line(results_dir):
    text = explain(results_dir, 0.03, 2, sharpe_1x=-0.1)
    assert text.endswith(f"{ONLY_TEST} PASS.{COST_WARNING}")


# ------------------------------------------------------------- bad inputs
def test_unknown_factor_or_period(results_dir):
    write(results_dir, fake_record())
    with pytest.raises(KeyError, match="unknown factor_key"):
        get_analysis("not_a_factor", PERIOD)
    with pytest.raises(KeyError, match="unknown period_key"):
        get_analysis(FACTOR, "1990_1995")


def test_excluded_factor_raises_even_if_results_exist(results_dir):
    write(results_dir, fake_record(factor="roll_spread"))
    with pytest.raises(KeyError, match="excluded from the app"):
        get_analysis("roll_spread", PERIOD)


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


# --------------------------------------------------------- reversed verdict
# 99 nulls evenly spread over [-1, 1] and a real gross Sharpe of -2, below all
# of them. Reverse test: negate everything, so the real value is +2 and no
# negated null is >= 2 -> reverse p = (0 + 1) / (99 + 1) = 0.01.
WORSE_NULLS = [float(x) for x in np.linspace(-1.0, 1.0, 99)]


def worse_record(**overrides):
    return fake_record(
        p_value=1.0, sharpe_gross=-2.0, null_sharpes_gross=WORSE_NULLS, n_shuffles=99,
        **overrides,
    )


def test_reverse_p_uses_permutation_pvalue_on_negated_sharpes(results_dir):
    write(results_dir, worse_record())
    r = get_analysis(FACTOR, PERIOD, k=3)
    expected, _ = permutation_pvalue(2.0, -np.array(WORSE_NULLS))
    assert r.reverse_p_value == expected == pytest.approx(0.01)
    assert r.adjusted_reverse_p_value == pytest.approx(0.03)
    # a tie counts toward the reverse p-value, just like the forward one
    write(results_dir, fake_record(sharpe_gross=0.1, null_sharpes_gross=[0.1, 0.5, 0.9]))
    assert get_analysis(FACTOR, PERIOD).reverse_p_value == pytest.approx(2 / 4)


def test_reversed_fail(results_dir):
    write(results_dir, worse_record())
    v = verdict_for(get_analysis(FACTOR, PERIOD), 1)
    assert v.label == "FAIL"
    assert v.reversed is True
    assert v.headline == "Consistently worse than random"
    assert v.raw_reverse_p_value == v.adjusted_reverse_p_value == pytest.approx(0.01)
    assert v.explanation == (
        "This strategy did consistently worse than random, beating only 0% of 99 "
        "random versions. The opposite strategy would have looked strong on this data."
    )


def test_normal_fail_is_not_reversed(results_dir):
    write(results_dir, fake_record(p_value=0.5))  # reverse p = 1.0 for these nulls
    v = verdict_for(get_analysis(FACTOR, PERIOD), 1)
    assert v.label == "FAIL"
    assert v.reversed is False
    assert v.headline == "Indistinguishable from luck"
    assert v.raw_reverse_p_value == 1.0
    assert v.explanation.startswith("Random versions of this strategy did about as well")


def test_reversed_at_k1_but_not_at_higher_k(results_dir):
    write(results_dir, worse_record())
    r = get_analysis(FACTOR, PERIOD)
    assert verdict_for(r, 4).reversed is True  # 0.04 < 0.05
    v = verdict_for(r, 5)  # 0.05 is not < 0.05
    assert v.adjusted_reverse_p_value == pytest.approx(0.05)
    assert v.label == "FAIL"
    assert v.reversed is False
    assert v.headline == "Indistinguishable from luck"
    assert v.explanation == (
        "Random versions of this strategy did about as well or better. There's no "
        "evidence the signal is anything more than luck. If this were your only "
        "test, the verdict would be FAIL (consistently worse than random)."
    )


def test_reversed_verdict_keeps_cost_warning(results_dir):
    write(results_dir, worse_record(sharpe_1x=-0.5))
    v = verdict_for(get_analysis(FACTOR, PERIOD), 2)
    assert v.reversed is True
    assert v.explanation.endswith(
        "If this were your only test, the verdict would be FAIL (consistently worse "
        "than random). After realistic trading costs, this strategy lost money."
    )


# ------------------------------------------------------ survivorship hook
def test_survivorship_note_empty_by_default(results_dir):
    assert api.SURVIVORSHIP_NOTE == ""
    text = explain(results_dir, 0.01, 1)
    assert text.endswith("before trading costs.")  # nothing appended, no stray space


def test_survivorship_note_appends_to_pass_only(results_dir, monkeypatch):
    monkeypatch.setattr(api, "SURVIVORSHIP_NOTE", "Survivorship note here.")
    assert explain(results_dir, 0.01, 1).endswith(" before trading costs. Survivorship note here.")
    assert explain(results_dir, 0.01, 2, sharpe_1x=-0.1).endswith(
        "this strategy lost money. Survivorship note here."
    )
    assert "Survivorship note" not in explain(results_dir, 0.10, 1)  # INCONCLUSIVE
    assert "Survivorship note" not in explain(results_dir, 0.5, 1)  # FAIL
    write(results_dir, worse_record())
    assert "Survivorship note" not in verdict_for(get_analysis(FACTOR, PERIOD), 1).explanation


# ----------------------------------------------------------- shipped results
def test_shipped_results_cover_every_menu_config(monkeypatch):
    """results/permutations/ (the default directory) has a valid file for all 50."""
    monkeypatch.delenv("QFRP_RESULTS_DIR", raising=False)
    configs = [(f.key, p.key) for f in list_factors() for p in list_periods()]
    assert len(configs) == 50
    for factor, period in configs:
        r = get_analysis(factor, period)
        assert r.n_permutations == 2000
        assert len(r.null_sharpes_gross) == 2000 - r.n_dropped
        assert r.equity_curve["strategy"].iloc[0] == 1.0
