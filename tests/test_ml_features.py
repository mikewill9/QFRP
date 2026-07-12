"""Feature assembly: naming, ordering, transforms, validation, mixed sources."""

from __future__ import annotations

import pytest

from quantlab.data import SyntheticFundamentalSource
from quantlab.exceptions import DataError
from quantlab.ml import FeatureSpec, build_features
from quantlab.research import (
    EarningsYield,
    Momentum,
    ShortTermReversal,
    SignalPipeline,
    ZScore,
)


def test_build_features_names_shapes_and_order(small_synth):
    specs = [
        FeatureSpec("mom", Momentum(lookback=63, skip=5)),
        FeatureSpec("rev", ShortTermReversal(lookback=5)),
    ]
    features = build_features(small_synth, specs)
    assert list(features.keys()) == ["mom", "rev"]
    for frame in features.values():
        assert frame.index.equals(small_synth.close.index)
        assert frame.columns.equals(small_synth.close.columns)


def test_build_features_applies_transform(small_synth):
    factor = ShortTermReversal(lookback=5)
    raw = factor.compute(small_synth)
    features = build_features(
        small_synth, [FeatureSpec("rev_z", factor, SignalPipeline([ZScore()]))]
    )
    zscored = features["rev_z"].dropna(how="any")
    assert len(zscored) > 0
    assert (zscored.mean(axis=1).abs() < 1e-9).all()
    raw_rows = raw.dropna(how="any")
    assert (raw_rows.mean(axis=1).abs() > 1e-9).any()


def test_build_features_validates(small_synth):
    with pytest.raises(DataError, match="empty"):
        build_features(small_synth, [])
    with pytest.raises(DataError, match="Duplicate.*mom"):
        build_features(
            small_synth,
            [
                FeatureSpec("mom", Momentum(lookback=63, skip=5)),
                FeatureSpec("mom", ShortTermReversal(lookback=5)),
            ],
        )


def test_build_features_mixed_price_and_fundamental(small_synth):
    # Same SYN000.. naming as the price source; fetch from well before the
    # market window so TTM values are published by the dataset's dates.
    source = SyntheticFundamentalSource(
        n_assets=5, seed=11, inception="2019-01-02"
    )
    fundamentals = source.fetch(source.symbols, "2019-01-02", "2021-06-30")
    specs = [
        FeatureSpec("rev", ShortTermReversal(lookback=5)),
        FeatureSpec("ey", EarningsYield(fundamentals)),
    ]
    features = build_features(small_synth, specs)
    assert list(features.keys()) == ["rev", "ey"]
    for frame in features.values():
        assert frame.index.equals(small_synth.close.index)
        assert frame.columns.equals(small_synth.close.columns)
    assert features["ey"].notna().any().any()
