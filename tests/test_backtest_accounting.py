"""Cost models and portfolio accounting: exact-math checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import (
    BpsCommission,
    FillEvent,
    FixedBpsSlippage,
    PerShareCommission,
    Portfolio,
    VolumeImpactSlippage,
    ZeroCommission,
    ZeroSlippage,
)
from quantlab.exceptions import BacktestError

TS = pd.Timestamp("2022-03-01")


def _fill(symbol="AAA", qty=100.0, price=50.0, commission=0.0, slip=0.0):
    return FillEvent(
        timestamp=TS, symbol=symbol, quantity=qty, price=price,
        commission=commission, slippage_cost=slip, order_id=1,
    )


# ------------------------------------------------------------------- costs
def test_per_share_commission():
    m = PerShareCommission(rate=0.005, minimum=1.0)
    assert m.commission(1000, 50.0) == pytest.approx(5.0)
    assert m.commission(-1000, 50.0) == pytest.approx(5.0)  # sells charged too
    assert m.commission(10, 50.0) == pytest.approx(1.0)  # minimum binds
    assert m.commission(0, 50.0) == 0.0


def test_bps_commission():
    m = BpsCommission(bps=2.0)
    assert m.commission(100, 50.0) == pytest.approx(100 * 50 * 2 / 1e4)
    assert ZeroCommission().commission(1e9, 100.0) == 0.0
    with pytest.raises(BacktestError):
        BpsCommission(bps=-1)


def test_fixed_bps_slippage_symmetric():
    m = FixedBpsSlippage(bps=10.0)
    assert m.slippage_fraction(100, 50.0, None, None) == pytest.approx(1e-3)
    assert m.slippage_fraction(-100, 50.0, None, None) == pytest.approx(1e-3)
    assert ZeroSlippage().slippage_fraction(1e6, 1.0, 1.0, 1.0) == 0.0


def test_volume_impact_sqrt_scaling():
    m = VolumeImpactSlippage(spread_bps=0.0, impact_coeff=0.1)
    base = m.slippage_fraction(1_000, 50.0, volume=100_000, sigma=0.02)
    quad = m.slippage_fraction(4_000, 50.0, volume=100_000, sigma=0.02)
    assert quad == pytest.approx(2.0 * base)  # 4x size -> 2x impact
    assert base == pytest.approx(0.1 * 0.02 * np.sqrt(0.01))


def test_volume_impact_fallbacks():
    m = VolumeImpactSlippage(spread_bps=4.0, impact_coeff=0.1, fallback_sigma=0.03)
    # No volume: only the half spread applies.
    assert m.slippage_fraction(1000, 50.0, None, 0.02) == pytest.approx(2e-4)
    # No sigma: fallback sigma is used.
    got = m.slippage_fraction(1000, 50.0, 100_000, None)
    assert got == pytest.approx(2e-4 + 0.1 * 0.03 * np.sqrt(0.01))


# --------------------------------------------------------------- portfolio
def test_buy_accounting_identity():
    p = Portfolio(100_000.0)
    p.apply_fill(_fill(qty=100, price=50.0, commission=5.0))
    assert p.cash == pytest.approx(100_000 - 5_000 - 5)
    assert p.position("AAA") == 100
    assert p.equity == pytest.approx(100_000 - 5)  # only commission lost
    p.mark_to_market(pd.Series({"AAA": 55.0}))
    assert p.equity == pytest.approx(100_000 - 5 + 100 * 5)


def test_short_position_marks_correctly():
    p = Portfolio(100_000.0)
    p.apply_fill(_fill(qty=-200, price=40.0))
    assert p.cash == pytest.approx(108_000)  # proceeds credited
    assert p.equity == pytest.approx(100_000)
    p.mark_to_market(pd.Series({"AAA": 45.0}))  # short moves against us
    assert p.equity == pytest.approx(100_000 - 200 * 5)
    w = p.weights()
    assert w["AAA"] == pytest.approx(-200 * 45 / p.equity)


def test_round_trip_flattens_position():
    p = Portfolio(50_000.0)
    p.apply_fill(_fill(qty=100, price=20.0))
    p.apply_fill(_fill(qty=-100, price=22.0))
    assert p.position("AAA") == 0.0
    assert "AAA" not in p.positions
    assert p.equity == pytest.approx(50_000 + 100 * 2)
    assert len(p.ledger) == 2


def test_nan_marks_keep_previous_price():
    p = Portfolio(10_000.0)
    p.apply_fill(_fill(qty=10, price=100.0))
    p.mark_to_market(pd.Series({"AAA": np.nan}))
    assert p.equity == pytest.approx(10_000)  # fill price mark retained


def test_ledger_frame_columns():
    p = Portfolio(10_000.0)
    p.apply_fill(_fill(qty=10, price=100.0, commission=1.0, slip=0.5))
    frame = p.ledger.to_frame()
    assert list(frame.columns) == [
        "timestamp", "symbol", "quantity", "price",
        "notional", "commission", "slippage_cost", "order_id",
    ]
    assert frame.iloc[0]["notional"] == pytest.approx(1000.0)
    empty = Portfolio(1.0).ledger.to_frame()
    assert empty.empty and list(empty.columns) == list(frame.columns)


def test_invalid_capital_and_weights_guard():
    with pytest.raises(BacktestError):
        Portfolio(0.0)
