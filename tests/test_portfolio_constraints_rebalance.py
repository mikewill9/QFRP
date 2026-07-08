"""Constraints projection/validation and rebalance policy behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.exceptions import OptimizationError
from quantlab.portfolio import Constraints, RebalancePolicy


def _w(**kwargs: float) -> pd.Series:
    return pd.Series(dict(kwargs), dtype=float)


# ------------------------------------------------------------ Constraints


def test_project_clips_to_bounds():
    cons = Constraints(max_weight=0.25, min_weight=-0.10)
    out = cons.project(_w(A=0.50, B=-0.30, C=0.10))
    assert out["A"] == pytest.approx(0.25)
    assert out["B"] == pytest.approx(-0.10)
    assert out["C"] == pytest.approx(0.10)


def test_project_long_only_zeroes_shorts():
    out = Constraints(long_only=True).project(_w(A=0.6, B=-0.2))
    assert out["A"] == pytest.approx(0.6)
    assert out["B"] == 0.0


def test_project_scales_gross():
    out = Constraints(max_gross=1.0).project(_w(A=1.5, B=-0.5))
    assert out.abs().sum() == pytest.approx(1.0)
    # Relative sizes preserved.
    assert out["A"] / out["B"] == pytest.approx(-3.0)


def test_project_scales_net():
    out = Constraints(max_net=0.1).project(_w(A=0.6, B=-0.2))
    assert abs(out.sum()) == pytest.approx(0.1)


def test_project_net_zero_demeans():
    out = Constraints(max_net=0.0).project(_w(A=0.6, B=-0.2))
    assert out.sum() == pytest.approx(0.0)


def test_project_fills_nan_with_zero():
    out = Constraints().project(pd.Series({"A": 0.5, "B": float("nan")}))
    assert out["B"] == 0.0


def test_is_satisfied():
    cons = Constraints(long_only=True, max_weight=0.5, max_gross=1.0)
    assert cons.is_satisfied(_w(A=0.5, B=0.5))
    assert not cons.is_satisfied(_w(A=0.6, B=0.3))  # max_weight
    assert not cons.is_satisfied(_w(A=0.5, B=-0.1))  # long_only
    assert not cons.is_satisfied(_w(A=0.5, B=0.5, C=0.5))  # gross


def test_scipy_bounds_respect_long_only_and_caps():
    assert Constraints(long_only=True, max_weight=0.3).scipy_bounds(2) == [
        (0.0, 0.3),
        (0.0, 0.3),
    ]
    assert Constraints(min_weight=-0.2).scipy_bounds(1) == [(-0.2, None)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_weight": 0.0},
        {"max_weight": 0.1, "min_weight": 0.2},
        {"long_only": True, "min_weight": -0.1},
        {"max_gross": -1.0},
        {"max_net": -0.5},
    ],
)
def test_constraints_validation(kwargs):
    with pytest.raises(OptimizationError):
        Constraints(**kwargs)


# ------------------------------------------------------------ RebalancePolicy


def test_no_policy_returns_target():
    policy = RebalancePolicy()
    cur, tgt = _w(A=0.5, B=0.5), _w(A=0.2, B=0.8)
    pd.testing.assert_series_equal(policy.apply(cur, tgt), tgt)


def test_band_suppresses_small_trades():
    policy = RebalancePolicy(band=0.05)
    out = policy.apply(_w(A=0.50, B=0.50), _w(A=0.53, B=0.30))
    assert out["A"] == pytest.approx(0.50)  # |0.03| within band: no trade
    assert out["B"] == pytest.approx(0.30)  # |0.20| trades in full


def test_turnover_cap_scales_trades_pro_rata():
    policy = RebalancePolicy(max_turnover=0.10)
    cur, tgt = _w(A=0.60, B=0.40), _w(A=0.20, B=0.80)
    # Desired one-way turnover: 0.5 * (0.4 + 0.4) = 0.4 -> scale by 0.25.
    out = policy.apply(cur, tgt)
    assert out["A"] == pytest.approx(0.60 - 0.40 * 0.25)
    assert out["B"] == pytest.approx(0.40 + 0.40 * 0.25)
    assert policy.turnover(cur, tgt) == pytest.approx(0.10)


def test_all_or_nothing_skips_when_over_cap():
    policy = RebalancePolicy(max_turnover=0.10, partial=False)
    cur = _w(A=0.60, B=0.40)
    pd.testing.assert_series_equal(policy.apply(cur, _w(A=0.20, B=0.80)), cur)


def test_cap_not_binding_trades_in_full():
    policy = RebalancePolicy(max_turnover=0.50)
    out = policy.apply(_w(A=0.60, B=0.40), _w(A=0.55, B=0.45))
    assert out["A"] == pytest.approx(0.55)


def test_apply_aligns_disjoint_indices():
    # A leaves the book, C enters.
    out = RebalancePolicy().apply(_w(A=0.5), _w(C=0.5))
    assert out["A"] == 0.0
    assert out["C"] == pytest.approx(0.5)


def test_band_and_cap_compose():
    policy = RebalancePolicy(band=0.05, max_turnover=0.05)
    cur = _w(A=0.50, B=0.30, C=0.20)
    tgt = _w(A=0.52, B=0.10, C=0.38)
    out = policy.apply(cur, tgt)
    # A's 0.02 move is banded away; B and C trades (0.2 + 0.18)/2 = 0.19
    # one-way, scaled to 0.05.
    assert out["A"] == pytest.approx(0.50)
    scale = 0.05 / 0.19
    assert out["B"] == pytest.approx(0.30 - 0.20 * scale)
    assert out["C"] == pytest.approx(0.20 + 0.18 * scale)


@pytest.mark.parametrize("kwargs", [{"band": -0.01}, {"max_turnover": -0.1}])
def test_rebalance_validation(kwargs):
    with pytest.raises(OptimizationError):
        RebalancePolicy(**kwargs)
