"""Feature assembly for supervised panels.

Computes a set of factors on one dataset and applies per-feature
normalization, producing the ``{name: wide frame}`` mapping that
``PanelDataset.from_factors`` consumes. Cross-sectional normalization
(ZScore / Rank) is strongly recommended for every feature so that no
feature carries a market-level component or an incomparable scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.research.signals import Transform
from quantlab.types import WideFrame

__all__ = ["FeatureSpec", "build_features"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature: a factor plus optional post-compute normalization.

    ``transform=None`` emits raw scores; pass ``SignalPipeline([...])`` or
    any ``Transform`` for normalization. ``name`` becomes the panel column.
    """

    name: str
    factor: Factor
    transform: Transform | None = None


def build_features(
    dataset: MarketDataset, specs: Sequence[FeatureSpec]
) -> dict[str, WideFrame]:
    """Compute every spec on ``dataset`` -> ``{name: transformed frame}``.

    Raises DataError on empty specs or duplicate names. Order preserved.

    Fundamental factors carry their own ``FundamentalDataset`` from
    construction, so mixed price/fundamental feature sets need nothing
    extra here.
    """
    if not specs:
        raise DataError("specs is empty; provide at least one FeatureSpec")
    names = [spec.name for spec in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise DataError(f"Duplicate feature names: {duplicates}")

    features: dict[str, WideFrame] = {}
    for spec in specs:
        frame = spec.factor.compute(dataset)
        if spec.transform is not None:
            frame = spec.transform.transform(frame)
        features[spec.name] = frame
    return features
