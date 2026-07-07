"""Signal transforms: row-wise semantics, composition, no mutation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import DataError
from quantlab.research import (
    Clip,
    Decay,
    Demean,
    Lag,
    Neutralize,
    Rank,
    SignalPipeline,
    Winsorize,
    ZScore,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=50)
    cols = [f"S{i}" for i in range(8)]
    return pd.DataFrame(rng.normal(0, 1, (50, 8)), index=idx, columns=cols)


def test_zscore_rows_standardized(frame):
    z = ZScore().transform(frame)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-12)


def test_rank_centered_and_bounded(frame):
    r = Rank().transform(frame)
    assert r.max().max() <= 0.5
    assert r.min().min() >= -0.5 + 1e-9
    assert np.allclose(r.mean(axis=1), 0.0, atol=1e-2)
    raw = Rank(center=False).transform(frame)
    assert raw.min().min() > 0.0
    assert raw.max().max() <= 1.0


def test_winsorize_clips_outliers(frame):
    frame = frame.copy()
    frame.iloc[10, 0] = 100.0
    w = Winsorize(0.05, 0.95).transform(frame)
    assert w.iloc[10, 0] < 100.0
    assert w.iloc[10, 0] == pytest.approx(frame.iloc[10].quantile(0.95))
    with pytest.raises(DataError):
        Winsorize(0.6, 0.4)


def test_demean_and_clip(frame):
    d = Demean().transform(frame)
    assert np.allclose(d.mean(axis=1), 0.0, atol=1e-12)
    c = Clip(-0.5, 0.5).transform(frame)
    assert c.abs().max().max() <= 0.5


def test_neutralize_zeroes_group_means(frame):
    groups = {c: ("A" if i < 4 else "B") for i, c in enumerate(frame.columns)}
    n = Neutralize(groups).transform(frame)
    a_cols = [c for c in frame.columns if groups[c] == "A"]
    b_cols = [c for c in frame.columns if groups[c] == "B"]
    assert np.allclose(n[a_cols].mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(n[b_cols].mean(axis=1), 0.0, atol=1e-12)
    # Input frame untouched.
    assert not np.allclose(frame[a_cols].mean(axis=1), 0.0)


def test_neutralize_unknown_symbols_grouped_as_other(frame):
    groups = {frame.columns[0]: "A"}  # everything else falls into _other
    n = Neutralize(groups).transform(frame)
    others = list(frame.columns[1:])
    assert np.allclose(n[others].mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(n[frame.columns[0]], 0.0, atol=1e-12)  # singleton group


def test_lag_shifts_in_time(frame):
    lagged = Lag(3).transform(frame)
    assert lagged.iloc[:3].isna().all().all()
    pd.testing.assert_frame_equal(
        lagged.iloc[3:], frame.iloc[:-3].set_axis(frame.index[3:])
    )


def test_decay_smooths_less_with_short_halflife(frame):
    fast = Decay(halflife=1.0).transform(frame)
    slow = Decay(halflife=20.0).transform(frame)
    # Longer halflife = smoother = lower variance of day-over-day changes.
    assert slow.diff().std().mean() < fast.diff().std().mean()


def test_pipeline_composes_in_order(frame):
    pipe = SignalPipeline([Winsorize(0.05, 0.95), ZScore(), Clip(-2, 2)])
    out = pipe.transform(frame)
    assert out.abs().max().max() <= 2.0
    manual = Clip(-2, 2).transform(
        ZScore().transform(Winsorize(0.05, 0.95).transform(frame))
    )
    pd.testing.assert_frame_equal(out, manual)
    with pytest.raises(DataError):
        SignalPipeline([])
    with pytest.raises(DataError):
        SignalPipeline([ZScore(), "not a transform"])  # type: ignore[list-item]


def test_transforms_preserve_nan(frame):
    frame = frame.copy()
    frame.iloc[5, 2] = np.nan
    for t in (ZScore(), Rank(), Demean(), Winsorize()):
        out = t.transform(frame)
        assert np.isnan(out.iloc[5, 2])
        assert out.iloc[6].notna().all()
