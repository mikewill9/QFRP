import math

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.significance import (
    bonferroni_adjust,
    permutation_pvalue,
    permute_within_date,
)


def make_series():
    return pd.Series([1.5, np.nan, -0.5, 2.0, np.nan],
                     index=["AAPL", "MSFT", "XOM", "JPM", "KO"])


def test_values_preserved():
    s = make_series()
    result = permute_within_date(s, np.random.default_rng(0))
    before = np.sort(s.dropna().to_numpy())
    after = np.sort(result.dropna().to_numpy())
    assert np.array_equal(before, after)


def test_nan_positions_unchanged():
    s = make_series()
    for seed in range(50):
        result = permute_within_date(s, np.random.default_rng(seed))
        assert (result.isna() == s.isna()).all()


def test_input_not_modified():
    s = make_series()
    original = s.copy()
    permute_within_date(s, np.random.default_rng(0))
    pd.testing.assert_series_equal(s, original)


def test_same_seed_same_result():
    s = make_series()
    a = permute_within_date(s, np.random.default_rng(7))
    b = permute_within_date(s, np.random.default_rng(7))
    pd.testing.assert_series_equal(a, b)


def test_single_value_unchanged():
    s = pd.Series([np.nan, 3.0, np.nan], index=["A", "B", "C"])
    result = permute_within_date(s, np.random.default_rng(0))
    pd.testing.assert_series_equal(result, s)

def test_pvalue_hand_computed():
    p, dropped = permutation_pvalue(1.0, np.array([0.5, 1.2, np.nan, 1.0, -0.3]))
    assert math.isclose(p, 0.6)
    assert dropped == 1


def test_pvalue_all_ties():
    p, _ = permutation_pvalue(1.0, np.array([1.0, 1.0, 1.0]))
    assert math.isclose(p, 1.0)


def test_pvalue_real_beats_all():
    p, _ = permutation_pvalue(5.0, np.array([0.0, 1.0, 2.0]))
    assert math.isclose(p, 0.25)


def test_pvalue_all_nan_nulls_raises():
    with pytest.raises(ValueError, match="no valid"):
        permutation_pvalue(1.0, np.array([np.nan, np.nan, np.nan]))


def test_pvalue_nan_real_raises():
    with pytest.raises(ValueError, match="NaN"):
        permutation_pvalue(np.nan, np.array([0.5, 1.0]))

def test_bonferroni_basic():
    assert math.isclose(bonferroni_adjust(.02, 5), .10)


def test_bonferroni_caps_at_one():
    assert math.isclose(bonferroni_adjust(.3, 5), 1.0)


def test_bonferroni_k_one_unchanged():
    assert math.isclose(bonferroni_adjust(.02, 1), .02)


def test_bonferroni_bad_k_raises():
    with pytest.raises(ValueError, match="positive integer"):
        bonferroni_adjust(.02, 0)


def test_bonferroni_bad_p_raises():
    for bad_p in [np.nan, -1, 1.5]:
        with pytest.raises(ValueError, match="probability"):
            bonferroni_adjust(bad_p, 5)
