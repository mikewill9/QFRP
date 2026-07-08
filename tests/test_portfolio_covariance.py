"""Covariance estimators: exact small cases, properties, error handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import DataError
from quantlab.portfolio import ewma_cov, ledoit_wolf_cc, sample_cov


def _frame(data: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(data.values())))
    return pd.DataFrame(data, index=pd.bdate_range("2022-01-03", periods=n))


def _random_returns(n_obs=500, n_assets=6, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_obs)
    data = {
        f"A{i}": common * (0.5 + 0.2 * i) + rng.normal(0, 0.01, n_obs)
        for i in range(n_assets)
    }
    return _frame({k: list(v) for k, v in data.items()})


# ------------------------------------------------------------ sample_cov


def test_sample_cov_matches_hand_computation():
    r = _frame({"A": [0.01, -0.01, 0.02, 0.00], "B": [0.00, 0.01, -0.02, 0.01]})
    cov = sample_cov(r)
    a = np.array(r["A"]) - np.mean(r["A"])
    b = np.array(r["B"]) - np.mean(r["B"])
    assert cov.loc["A", "A"] == pytest.approx(float(a @ a) / 3)
    assert cov.loc["A", "B"] == pytest.approx(float(a @ b) / 3)
    assert cov.loc["A", "B"] == cov.loc["B", "A"]


def test_sample_cov_matches_pandas():
    r = _random_returns()
    pd.testing.assert_frame_equal(sample_cov(r), r.cov(), rtol=1e-12)


def test_sample_cov_drops_nan_rows():
    r = _frame({"A": [0.01, np.nan, 0.02, 0.00], "B": [0.00, 0.01, -0.02, 0.01]})
    expected = sample_cov(r.dropna())
    pd.testing.assert_frame_equal(sample_cov(r), expected)


def test_sample_cov_psd():
    cov = sample_cov(_random_returns()).to_numpy()
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-12


# ------------------------------------------------------------ ewma_cov


def test_ewma_cov_two_observations_hand_case():
    # halflife=1: weights on (older, newer) are (1/3, 2/3) after normalizing.
    r = _frame({"A": [0.02, -0.01]})
    w = np.array([1.0, 2.0]) / 3.0
    x = np.array([0.02, -0.01])
    mean = w @ x
    expected = float(w @ (x - mean) ** 2)
    assert ewma_cov(r, halflife=1.0).loc["A", "A"] == pytest.approx(expected)


def test_ewma_cov_large_halflife_approaches_equal_weight():
    r = _random_returns(n_obs=200)
    ewma = ewma_cov(r, halflife=1e9)
    x = r.to_numpy() - r.to_numpy().mean(axis=0)
    equal_weight = x.T @ x / len(x)  # ddof=0
    np.testing.assert_allclose(ewma.to_numpy(), equal_weight, rtol=1e-6)


def test_ewma_cov_emphasizes_recent_data():
    # Quiet early sample, volatile late sample: short halflife sees more vol.
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 0.001, 250), rng.normal(0, 0.03, 250)])
    r = _frame({"A": list(vals)})
    short = ewma_cov(r, halflife=10.0).loc["A", "A"]
    long = ewma_cov(r, halflife=5000.0).loc["A", "A"]
    assert short > long * 1.5


def test_ewma_cov_rejects_bad_halflife():
    with pytest.raises(DataError, match="halflife"):
        ewma_cov(_random_returns(), halflife=0.0)


# ------------------------------------------------------------ ledoit_wolf_cc


def test_ledoit_wolf_intensity_in_unit_interval_and_symmetric():
    cov, intensity = ledoit_wolf_cc(_random_returns())
    assert 0.0 <= intensity <= 1.0
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)


def test_ledoit_wolf_preserves_sample_variances():
    r = _random_returns()
    cov, _ = ledoit_wolf_cc(r)
    x = r.to_numpy() - r.to_numpy().mean(axis=0)
    ml_var = (x**2).mean(axis=0)  # 1/T normalization per the estimator
    np.testing.assert_allclose(np.diag(cov.to_numpy()), ml_var, rtol=1e-12)


def test_ledoit_wolf_two_assets_target_equals_sample():
    # With two assets the constant-correlation target IS the sample matrix
    # (the average off-diagonal correlation is the only correlation), so
    # there is nothing to shrink.
    r = _random_returns(n_assets=2)
    cov, intensity = ledoit_wolf_cc(r)
    assert intensity == 0.0
    x = r.to_numpy() - r.to_numpy().mean(axis=0)
    np.testing.assert_allclose(cov.to_numpy(), x.T @ x / len(x), rtol=1e-12)


def test_ledoit_wolf_shrinks_more_with_less_data():
    wide = _random_returns(n_obs=1000, n_assets=8)
    _, low = ledoit_wolf_cc(wide)
    _, high = ledoit_wolf_cc(wide.iloc[:20])
    assert high > low


def test_ledoit_wolf_off_diagonals_move_toward_target():
    r = _random_returns(n_obs=30, n_assets=8, seed=5)
    cov, intensity = ledoit_wolf_cc(r)
    assert intensity > 0.0
    x = r.to_numpy() - r.to_numpy().mean(axis=0)
    s = x.T @ x / len(x)
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    n = s.shape[0]
    r_bar = (corr.sum() - n) / (n * (n - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    expected = intensity * target + (1 - intensity) * s
    np.testing.assert_allclose(cov.to_numpy(), expected, rtol=1e-12)


def test_ledoit_wolf_psd():
    cov, _ = ledoit_wolf_cc(_random_returns(n_obs=25, n_assets=10))
    assert np.linalg.eigvalsh(cov.to_numpy()).min() >= -1e-12


def test_ledoit_wolf_rejects_zero_variance_asset():
    r = _frame({"A": [0.01, -0.01, 0.02, 0.0], "B": [0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(DataError, match="Zero-variance"):
        ledoit_wolf_cc(r)


# ------------------------------------------------------------ shared validation


@pytest.mark.parametrize("estimator", [sample_cov, ewma_cov, ledoit_wolf_cc])
def test_estimators_reject_insufficient_rows(estimator):
    r = _frame({"A": [0.01], "B": [0.02]})
    with pytest.raises(DataError, match="complete return rows"):
        estimator(r)


@pytest.mark.parametrize("estimator", [sample_cov, ewma_cov, ledoit_wolf_cc])
def test_estimators_reject_inf(estimator):
    r = _frame({"A": [0.01, np.inf, 0.0, 0.01], "B": [0.0, 0.01, 0.0, 0.01]})
    with pytest.raises(DataError, match="non-finite"):
        estimator(r)


def test_estimators_reject_empty_frame():
    empty = pd.DataFrame(index=pd.bdate_range("2022-01-03", periods=5))
    with pytest.raises(DataError, match="no columns"):
        sample_cov(empty)
