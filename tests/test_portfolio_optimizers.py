"""Optimizers: closed-form checks, constraint satisfaction, error handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import OptimizationError
from quantlab.portfolio import (
    Constraints,
    EqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    MaxSharpe,
    MeanVariance,
    VolatilityTargeting,
)


def _cov(matrix: list[list[float]], symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=symbols, columns=symbols, dtype=float)


DIAG2 = _cov([[0.04, 0.0], [0.0, 0.01]], ["A", "B"])  # vols 20% and 10%
CORR3 = _cov(
    [
        [0.040, 0.012, 0.004],
        [0.012, 0.090, 0.009],
        [0.004, 0.009, 0.010],
    ],
    ["A", "B", "C"],
)
MU3 = pd.Series({"A": 0.08, "B": 0.05, "C": 0.03})


def _risk_contributions(w: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    vec = w.to_numpy()
    values = cov.to_numpy()
    marginal = values @ vec
    return vec * marginal / float(vec @ values @ vec)


# ------------------------------------------------------------ InverseVolatility


def test_inverse_vol_hand_case():
    w = InverseVolatility().weights(None, DIAG2)
    # 1/0.2 : 1/0.1 = 1 : 2.
    assert w["A"] == pytest.approx(1 / 3)
    assert w["B"] == pytest.approx(2 / 3)
    assert w.sum() == pytest.approx(1.0)


def test_inverse_vol_ignores_correlations():
    corr = _cov([[0.04, 0.018], [0.018, 0.01]], ["A", "B"])
    pd.testing.assert_series_equal(
        InverseVolatility().weights(None, corr),
        InverseVolatility().weights(None, DIAG2),
    )


# ------------------------------------------------------------ EqualRiskContribution


def test_erc_equals_inverse_vol_when_uncorrelated():
    w = EqualRiskContribution().weights(None, DIAG2)
    assert w["A"] == pytest.approx(1 / 3, rel=1e-6)
    assert w["B"] == pytest.approx(2 / 3, rel=1e-6)


def test_erc_risk_contributions_are_equal():
    w = EqualRiskContribution().weights(None, CORR3)
    rc = _risk_contributions(w, CORR3)
    np.testing.assert_allclose(rc, 1 / 3, rtol=1e-6)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_erc_rejects_non_psd():
    bad = _cov([[0.04, -0.05], [-0.05, 0.01]], ["A", "B"])  # |cov| > vol product
    with pytest.raises(OptimizationError, match="not PSD|converge"):
        EqualRiskContribution().weights(None, bad)


# ------------------------------------------------------------ MeanVariance


def test_mean_variance_unconstrained_closed_form():
    w = MeanVariance(risk_aversion=2.0).weights(MU3, CORR3)
    expected = np.linalg.solve(CORR3.to_numpy(), MU3.to_numpy()) / 2.0
    np.testing.assert_allclose(w.to_numpy(), expected, rtol=1e-12)


def test_mean_variance_two_asset_hand_case():
    mu = pd.Series({"A": 0.04, "B": 0.02})
    w = MeanVariance(risk_aversion=1.0).weights(mu, DIAG2)
    # Diagonal cov: w_i = mu_i / (gamma * var_i).
    assert w["A"] == pytest.approx(1.0)
    assert w["B"] == pytest.approx(2.0)


def test_mean_variance_higher_aversion_shrinks_positions():
    w1 = MeanVariance(risk_aversion=1.0).weights(MU3, CORR3)
    w5 = MeanVariance(risk_aversion=5.0).weights(MU3, CORR3)
    np.testing.assert_allclose(w5.to_numpy() * 5.0, w1.to_numpy(), rtol=1e-10)


def test_mean_variance_respects_constraints():
    mu = pd.Series({"A": 0.10, "B": -0.05, "C": 0.02})
    cons = Constraints(long_only=True, max_weight=0.6, max_gross=1.0)
    w = MeanVariance(risk_aversion=0.5, constraints=cons).weights(mu, CORR3)
    assert cons.is_satisfied(w)
    assert w["B"] == pytest.approx(0.0, abs=1e-8)  # negative mu, long-only


def test_mean_variance_constrained_matches_analytic_when_slack():
    # Constraints wide enough not to bind: SLSQP must find the closed form.
    analytic = MeanVariance(risk_aversion=5.0).weights(MU3, CORR3)
    loose = Constraints(max_gross=100.0)
    solved = MeanVariance(risk_aversion=5.0, constraints=loose).weights(MU3, CORR3)
    np.testing.assert_allclose(solved.to_numpy(), analytic.to_numpy(), atol=1e-6)


def test_mean_variance_requires_mu():
    with pytest.raises(OptimizationError, match="requires expected returns"):
        MeanVariance().weights(None, CORR3)


def test_mean_variance_rejects_singular_cov():
    singular = _cov([[0.04, 0.02], [0.02, 0.01]], ["A", "B"])  # rank 1
    with pytest.raises(OptimizationError, match="ill-conditioned"):
        MeanVariance().weights(pd.Series({"A": 0.01, "B": 0.01}), singular)


def test_mean_variance_rejects_incomplete_mu():
    with pytest.raises(OptimizationError, match="missing entries"):
        MeanVariance().weights(pd.Series({"A": 0.01}), CORR3)


# ------------------------------------------------------------ MaxSharpe


def test_max_sharpe_unconstrained_is_tangency():
    w = MaxSharpe().weights(MU3, CORR3)
    tangency = np.linalg.solve(CORR3.to_numpy(), MU3.to_numpy())
    np.testing.assert_allclose(
        w.to_numpy(), tangency / np.abs(tangency).sum(), rtol=1e-12
    )
    assert w.abs().sum() == pytest.approx(1.0)


def test_max_sharpe_constrained_stays_feasible_with_unit_gross():
    cons = Constraints(long_only=True, max_weight=0.5, max_gross=1.0)
    w = MaxSharpe(constraints=cons).weights(MU3, CORR3)
    assert cons.is_satisfied(w)
    assert w.abs().sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_constrained_sharpe_not_above_unconstrained():
    def sharpe(w: pd.Series) -> float:
        vec = w.to_numpy()
        return float(MU3.to_numpy() @ vec) / np.sqrt(
            float(vec @ CORR3.to_numpy() @ vec)
        )

    unconstrained = MaxSharpe().weights(MU3, CORR3)
    cons = Constraints(long_only=True, max_weight=0.5, max_gross=1.0)
    constrained = MaxSharpe(constraints=cons).weights(MU3, CORR3)
    assert sharpe(constrained) <= sharpe(unconstrained) + 1e-9


def test_max_sharpe_requires_mu():
    with pytest.raises(OptimizationError, match="requires expected returns"):
        MaxSharpe().weights(None, CORR3)


# ------------------------------------------------------------ HierarchicalRiskParity


def test_hrp_two_assets_inverse_variance_split():
    w = HierarchicalRiskParity().weights(None, DIAG2)
    # Bisection of two singletons allocates by inverse cluster variance.
    assert w["A"] == pytest.approx(0.01 / 0.05)
    assert w["B"] == pytest.approx(0.04 / 0.05)


def test_hrp_single_asset():
    one = _cov([[0.04]], ["A"])
    assert HierarchicalRiskParity().weights(None, one)["A"] == 1.0


def test_hrp_long_only_and_fully_invested():
    w = HierarchicalRiskParity().weights(None, CORR3)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_hrp_tolerates_singular_cov():
    # Rank-1 matrix breaks mean-variance but HRP never inverts it.
    singular = _cov(
        [[0.04, 0.02, 0.02], [0.02, 0.01, 0.01], [0.02, 0.01, 0.01]],
        ["A", "B", "C"],
    )
    w = HierarchicalRiskParity().weights(None, singular)
    assert w.sum() == pytest.approx(1.0)


def test_hrp_overweights_the_diversifier():
    # A and B are near-duplicates; C is independent with the same variance.
    dup = _cov(
        [
            [0.040, 0.038, 0.0],
            [0.038, 0.040, 0.0],
            [0.0, 0.0, 0.040],
        ],
        ["A", "B", "C"],
    )
    w = HierarchicalRiskParity().weights(None, dup)
    assert w["C"] > w["A"]
    assert w["C"] > w["B"]


# ------------------------------------------------------------ VolatilityTargeting


def test_vol_targeting_hits_target_exactly():
    ppy = 252
    wrapped = VolatilityTargeting(
        EqualRiskContribution(), target=0.10, periods_per_year=ppy
    )
    w = wrapped.weights(None, CORR3)
    vec = w.to_numpy()
    ann_vol = np.sqrt(float(vec @ CORR3.to_numpy() @ vec) * ppy)
    assert ann_vol == pytest.approx(0.10, rel=1e-10)


def test_vol_targeting_max_leverage_caps_gross():
    wrapped = VolatilityTargeting(
        InverseVolatility(), target=5.0, periods_per_year=252, max_leverage=2.0
    )
    w = wrapped.weights(None, CORR3)
    assert w.abs().sum() == pytest.approx(2.0)


def test_vol_targeting_validation():
    with pytest.raises(OptimizationError, match="target vol"):
        VolatilityTargeting(InverseVolatility(), target=0.0)


# ------------------------------------------------------------ shared validation


def test_optimizers_reject_asymmetric_cov():
    bad = pd.DataFrame(
        [[0.04, 0.01], [0.02, 0.01]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(OptimizationError, match="symmetric"):
        InverseVolatility().weights(None, bad)


def test_optimizers_reject_mismatched_labels():
    bad = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "C"]
    )
    with pytest.raises(OptimizationError, match="index and columns"):
        InverseVolatility().weights(None, bad)


def test_optimizers_reject_non_positive_variance():
    bad = _cov([[0.04, 0.0], [0.0, 0.0]], ["A", "B"])
    with pytest.raises(OptimizationError, match="non-positive variances"):
        EqualRiskContribution().weights(None, bad)
