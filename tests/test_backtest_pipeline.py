"""Tests for the shared notebook/script backtest convention."""

from __future__ import annotations

import pytest

from quantlab.backtest import (
    PerShareCommission,
    VolumeImpactSlippage,
    ZeroCommission,
    ZeroSlippage,
)
from quantlab.backtest.pipeline import (
    COMMISSION_MINIMUM,
    COMMISSION_RATE,
    IMPACT_COEFF,
    MAX_PARTICIPATION,
    PRICE_FACTOR_BUILDERS,
    SPREAD_BPS,
    TICKERS,
    make_execution,
    normalization,
    run_ls_backtest,
)
from quantlab.exceptions import BacktestError
from quantlab.research import Factor


def test_universe_excludes_known_bad_tickers():
    assert len(TICKERS) == 148
    assert "MMC" not in TICKERS
    assert "EA" not in TICKERS


def test_price_catalog_builds_fresh_factors():
    assert len(PRICE_FACTOR_BUILDERS) == 11
    for build in PRICE_FACTOR_BUILDERS.values():
        first, second = build(), build()
        assert isinstance(first, Factor)
        assert first is not second


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_make_execution_scales_fees_not_impact(k):
    ex = make_execution(k)
    assert isinstance(ex.slippage, VolumeImpactSlippage)
    assert isinstance(ex.commission, PerShareCommission)
    assert ex.slippage.spread_bps == SPREAD_BPS * k
    assert ex.commission.rate == COMMISSION_RATE * k
    assert ex.commission.minimum == COMMISSION_MINIMUM * k
    assert ex.slippage.impact_coeff == IMPACT_COEFF
    assert ex.max_participation == MAX_PARTICIPATION


def test_make_execution_rejects_negative_multiplier():
    with pytest.raises(BacktestError, match="cost_multiplier"):
        make_execution(-1.0)


def test_higher_costs_cost_more(synth):
    signal = normalization().transform(PRICE_FACTOR_BUILDERS["st_reversal"]().compute(synth))
    base = run_ls_backtest(signal, synth)
    doubled = run_ls_backtest(signal, synth, cost_multiplier=2.0)
    assert base.n_trades > 0
    assert doubled.costs_paid.sum() > base.costs_paid.sum()
    assert doubled.final_equity < base.final_equity


def test_frictionless_execution_charges_nothing(synth):
    ex = make_execution(frictionless=True)
    assert isinstance(ex.slippage, ZeroSlippage)
    assert isinstance(ex.commission, ZeroCommission)
    assert ex.max_participation == MAX_PARTICIPATION
    signal = normalization().transform(PRICE_FACTOR_BUILDERS["st_reversal"]().compute(synth))
    gross = run_ls_backtest(signal, synth, frictionless=True)
    net = run_ls_backtest(signal, synth)
    assert gross.n_trades > 0
    assert gross.costs_paid.sum() == 0.0
    assert gross.final_equity > net.final_equity


def test_frictionless_rejects_cost_multiplier():
    with pytest.raises(BacktestError, match="not both"):
        make_execution(2.0, frictionless=True)
