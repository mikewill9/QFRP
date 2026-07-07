"""Validation: each check catches its injected defect; clean data passes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quantlab.data import DataValidator, MarketDataset, Severity
from quantlab.data.validation import (
    ExtremeReturnCheck,
    MissingDataCheck,
    NonPositivePriceCheck,
    OHLCConsistencyCheck,
    StalePriceCheck,
    ZeroVolumeRunCheck,
)
from quantlab.exceptions import DataValidationError


def _with_close(ds: MarketDataset, close) -> MarketDataset:
    return replace(ds, close=close)


def test_clean_synthetic_passes(synth):
    report = DataValidator().validate(synth)
    assert report.passed, report.to_frame()
    assert report.n_errors == 0


def test_missing_data_check(small_synth):
    close = small_synth.close.copy()
    sym = small_synth.symbols[0]
    close.iloc[:40, close.columns.get_loc(sym)] = np.nan
    issues = MissingDataCheck(max_missing_frac=0.10).run(_with_close(small_synth, close))
    assert any(i.symbol == sym and i.severity == Severity.WARNING for i in issues)

    close[:] = np.nan
    issues = MissingDataCheck().run(_with_close(small_synth, close))
    assert all(i.severity == Severity.ERROR for i in issues)
    assert len(issues) == small_synth.n_symbols


def test_non_positive_price_check(small_synth):
    close = small_synth.close.copy()
    close.iloc[10, 0] = -1.0
    issues = NonPositivePriceCheck().run(_with_close(small_synth, close))
    assert issues and issues[0].severity == Severity.ERROR
    assert issues[0].count == 1


def test_ohlc_consistency_check(small_synth):
    high = small_synth.high.copy()
    high.iloc[5, 1] = small_synth.low.iloc[5, 1] * 0.5  # high below low
    broken = replace(small_synth, high=high)
    issues = OHLCConsistencyCheck().run(broken)
    assert issues and issues[0].severity == Severity.ERROR


def test_stale_price_check(small_synth):
    close = small_synth.close.copy()
    close.iloc[20:35, 0] = close.iloc[20, 0]  # 15-bar flat run
    issues = StalePriceCheck(max_run=10).run(_with_close(small_synth, close))
    assert issues and issues[0].count >= 14
    assert not StalePriceCheck(max_run=10).run(small_synth)


def test_extreme_return_check(small_synth):
    close = small_synth.close.copy()
    close.iloc[30, 0] = close.iloc[29, 0] * 3.0  # +200% print
    issues = ExtremeReturnCheck(max_abs_return=0.5).run(_with_close(small_synth, close))
    assert issues and issues[0].severity == Severity.WARNING
    assert "200" in issues[0].message or "worst" in issues[0].message


def test_zero_volume_run_check(small_synth):
    volume = small_synth.volume.copy()
    volume.iloc[10:20, 2] = 0.0
    issues = ZeroVolumeRunCheck(max_run=5).run(replace(small_synth, volume=volume))
    assert issues and issues[0].count == 10


def test_report_aggregation_and_raise(small_synth):
    close = small_synth.close.copy()
    close.iloc[10, 0] = -5.0  # ERROR
    close.iloc[30, 1] = close.iloc[29, 1] * 3.0  # WARNING
    report = DataValidator().validate(_with_close(small_synth, close))
    assert not report.passed
    assert report.n_errors >= 1
    assert report.n_warnings >= 1
    frame = report.to_frame()
    assert list(frame.columns) == ["check", "severity", "symbol", "count", "message"]
    assert frame.iloc[0]["severity"] == "ERROR"  # sorted most severe first
    with pytest.raises(DataValidationError, match="non_positive_price"):
        report.raise_for_errors()


def test_custom_check_suite(small_synth):
    report = DataValidator(checks=[StalePriceCheck(max_run=3)]).validate(small_synth)
    assert all(i.check == "stale_price" for i in report.issues)
