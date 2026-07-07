"""Shared fixtures for the quantlab test suite.

All fixtures are deterministic (fixed seeds) and offline (synthetic source
only), so the suite runs identically anywhere.
"""

from __future__ import annotations

import pytest

from quantlab.data import MarketDataset, make_synthetic_dataset


@pytest.fixture(scope="session")
def synth() -> MarketDataset:
    """Medium synthetic panel shared (read-only) across the session."""
    return make_synthetic_dataset(
        n_assets=12, start="2020-01-01", end="2021-12-31", n_sectors=3, seed=42
    )


@pytest.fixture
def small_synth() -> MarketDataset:
    """Small, fast panel for tests that mutate or slice aggressively."""
    return make_synthetic_dataset(
        n_assets=5, start="2021-01-01", end="2021-06-30", n_sectors=2, seed=11
    )
