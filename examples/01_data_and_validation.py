"""Example 01: data infrastructure end to end.

Builds a deterministic synthetic market, validates it, injects a defect to
show the validator catching it, filters the universe on liquidity,
resamples to weekly, and round-trips the dataset through the store.
Runs fully offline.

    python examples/01_data_and_validation.py
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from quantlab.data import (
    DatasetStore,
    DataValidator,
    filter_universe,
    make_synthetic_dataset,
    resample_dataset,
)
from quantlab.types import Frequency


def main() -> None:
    # 1. A deterministic 25-asset market with sector structure.
    ds = make_synthetic_dataset(
        n_assets=25, start="2019-01-01", end="2023-12-29", n_sectors=5, seed=7
    )
    print(f"dataset: {ds}")
    print(f"span: {ds.start.date()} .. {ds.end.date()}, bars: {ds.n_periods}")

    # 2. Validate. Clean synthetic data passes.
    report = DataValidator().validate(ds)
    print(f"\nvalidation: {report}")
    report.raise_for_errors()

    # 3. Inject a corrupt print and watch the validator flag it.
    close = ds.close.copy()
    close.iloc[100, 0] = -close.iloc[100, 0]
    broken_report = DataValidator().validate(replace(ds, close=close))
    print(f"\nafter injecting a negative price: {broken_report}")
    print(broken_report.to_frame().head(4).to_string(index=False))

    # 4. Universe filtering on liquidity.
    cutoff = float(ds.dollar_volume().median(axis=0).median())
    universe = filter_universe(ds, min_median_dollar_volume=cutoff)
    print(f"\nliquidity filter at median ADV {cutoff:,.0f}: "
          f"{ds.n_symbols} -> {universe.n_symbols} symbols")

    # 5. Returns and a couple of quick facts.
    rets = universe.returns(method="log")
    ann_vol = rets.std() * np.sqrt(252)
    print(f"annualized vol range: {ann_vol.min():.1%} .. {ann_vol.max():.1%}")

    # 6. Resample to weekly bars.
    weekly = resample_dataset(universe, Frequency.WEEKLY)
    print(f"weekly bars: {weekly.n_periods} (from {universe.n_periods} daily)")

    # 7. Persist and reload through the store.
    with tempfile.TemporaryDirectory() as tmp:
        store = DatasetStore(Path(tmp))
        store.save(universe, "universes/liquid_daily")
        loaded = store.load("universes/liquid_daily")
        assert loaded.equals(universe)
        print(f"\nstore round-trip OK: {store.list()}")


if __name__ == "__main__":
    main()
