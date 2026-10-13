"""CLI config selection in scripts/run_permutations.py (no backtests run)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quantlab.backtest.pipeline import PRICE_FACTOR_BUILDERS

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_permutations.py"
_spec = importlib.util.spec_from_file_location("_test_run_permutations", _PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_defaults_unchanged() -> None:
    args = runner.parse_args([])
    assert args.n == runner.DEFAULT_N
    assert args.base_seed == runner.DEFAULT_BASE_SEED
    assert args.out_dir == runner.DEFAULT_OUT_DIR
    assert args.config is None
    assert len(runner.selected_configs(args)) == len(PRICE_FACTOR_BUILDERS) * len(runner.PERIODS)


def test_factor_and_period_filters_unchanged() -> None:
    args = runner.parse_args(["--factor", "momentum"])
    assert runner.selected_configs(args) == [("momentum", p) for p in runner.PERIODS]
    args = runner.parse_args(["--factor", "momentum", "--period", "2020_2024"])
    assert runner.selected_configs(args) == [("momentum", "2020_2024")]


def test_config_list_keeps_order_and_drops_repeats() -> None:
    args = runner.parse_args(
        [
            "--config", "st_reversal:2020_2024",
            "--config", "momentum:2005_2009",
            "--config", "st_reversal:2020_2024",
        ]
    )  # fmt: skip
    assert runner.selected_configs(args) == [
        ("st_reversal", "2020_2024"),
        ("momentum", "2005_2009"),
    ]


@pytest.mark.parametrize("bad", ["momentum", "nope:2020_2024", "momentum:1999_2000", ":"])
def test_config_rejects_bad_values(bad: str) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", bad])


def test_config_cannot_combine_with_factor_or_period() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", "momentum:2020_2024", "--factor", "momentum"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", "momentum:2020_2024", "--period", "2020_2024"])
