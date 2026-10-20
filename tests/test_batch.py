"""Tests for app.api.get_batch_analysis against small fake results files.

Same fixture pattern as tests/test_api.py (fake_record / results_dir).
Every test is xfail(strict) until Michael implements the function: the
moment it works, pytest reports XPASS(strict) as a failure, which is the
signal to delete the ``pytestmark`` line below.
"""

from __future__ import annotations

import pytest
from app.api import ResultNotComputedError, get_analysis, get_batch_analysis, verdict_for

from test_api import fake_record, write


CONFIGS = [
    ("momentum", "2020_2024"),
    ("st_reversal", "2020_2024"),
    ("momentum", "2015_2019"),
]


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QFRP_RESULTS_DIR", str(tmp_path))
    return tmp_path


def write_all(results_dir, configs=CONFIGS, **overrides) -> None:
    for i, (factor, period) in enumerate(configs):
        # A different p-value per config so order mix-ups are visible.
        write(results_dir, fake_record(factor=factor, period=period, p_value=0.01 * (i + 1), **overrides))


# ------------------------------------------------------------------ shape
def test_one_pair_per_config_in_order(results_dir):
    write_all(results_dir)
    pairs = get_batch_analysis(CONFIGS, k=3)
    assert len(pairs) == len(CONFIGS)
    for (factor, period), (analysis, verdict) in zip(CONFIGS, pairs):
        assert analysis.factor_key == factor
        assert analysis.period_label == get_analysis(factor, period, k=3).period_label
        assert verdict.label in ("PASS", "INCONCLUSIVE", "FAIL")
    # order preserved: the fake p-values are 0.01, 0.02, 0.03
    assert [a.p_value for a, _ in pairs] == pytest.approx([0.01, 0.02, 0.03])


def test_every_result_uses_the_same_k(results_dir):
    write_all(results_dir)
    k = 7
    for (factor, period), (analysis, verdict) in zip(CONFIGS, get_batch_analysis(CONFIGS, k=k)):
        single = get_analysis(factor, period, k=k)
        assert analysis.k == k
        assert verdict.n_configs_viewed == k
        assert analysis.adjusted_p_value == pytest.approx(single.adjusted_p_value)
        assert verdict.adjusted_p_value == pytest.approx(analysis.adjusted_p_value)
        assert verdict.label == verdict_for(single, k).label


def test_k_equal_to_unique_count_is_allowed(results_dir):
    write_all(results_dir)
    assert len(get_batch_analysis(CONFIGS, k=len(CONFIGS))) == len(CONFIGS)


# ------------------------------------------------------------- bad inputs
def test_k_below_unique_count_raises(results_dir):
    write_all(results_dir)
    with pytest.raises(ValueError):
        get_batch_analysis(CONFIGS, k=len(CONFIGS) - 1)


def test_k_zero_raises(results_dir):
    write_all(results_dir)
    with pytest.raises(ValueError, match="k must be >= 1"):
        get_batch_analysis(CONFIGS[:1], k=0)


def test_bool_k_raises(results_dir):
    write_all(results_dir)
    with pytest.raises(TypeError, match="integer"):
        get_batch_analysis(CONFIGS[:1], k=True)


def test_unknown_factor_raises_key_error(results_dir):
    write_all(results_dir)
    with pytest.raises(KeyError, match="unknown factor_key"):
        get_batch_analysis([("not_a_factor", "2020_2024")], k=1)


def test_missing_results_file_raises_not_computed(results_dir):
    write_all(results_dir, configs=CONFIGS[:1])  # second config has no file
    with pytest.raises(ResultNotComputedError, match="not computed yet"):
        get_batch_analysis(CONFIGS[:2], k=2)
