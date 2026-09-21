import numpy as np
import pandas as pd

from quantlab.analytics.significance import permute_within_date


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