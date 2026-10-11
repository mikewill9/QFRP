"""Permutation test for every price factor x period.

For each (factor, period) config this script:

    1. Runs the real long/short backtest (quantlab.backtest.pipeline, the
       same code the notebook uses) with NO costs (gross), and at 1x, 2x
       and 3x costs.
    2. Runs N "shuffled" backtests, also with no costs. Each shuffle takes
       the real post-normalization z-scores and, on every rebalance date,
       randomly reassigns them across the stocks that have a score that day
       (permute_within_date). Everything else -- dates, universe, portfolio
       construction -- is identical to the real run.
    3. Computes a one-sided p-value from the GROSS Sharpes: how often a
       shuffled Sharpe is at least as good as the real one
       (permutation_pvalue).

Why gross? Independent shuffles on every rebalance date churn the whole
book each month -- roughly twice the real strategy's turnover -- so costed
nulls would pay about double the costs and sit below zero, making any real
signal look more significant than it is. Comparing gross against gross
tests the one question a permutation test can answer: does the factor's
ordering of stocks carry information? Whether that information survives
trading costs is answered separately by the 1x/2x/3x stress test (the
real strategy's own costs). The shuffles' average turnover is stored so the
gap can be reported.

Why shuffle within each date? The shuffle keeps each day's spread of
scores and which stocks are tradable, but breaks the link between a
stock's score and its future return. If the factor carries real
information, the real Sharpe should beat most shuffled Sharpes; if it
doesn't, the real Sharpe is just one more draw from the same distribution.

Data: prices must already be cached for DATA_START..2024-12-31 (see
quantlab.backtest.pipeline). Point QUANTLAB_CACHE_DIR at the fresh cache:

    QUANTLAB_CACHE_DIR=~/.quantlab/cache_2004_2024 python scripts/run_permutations.py --n 2000

The script refuses to run if the cache would need a download, so it can
never stitch new Yahoo prices onto an old cache (different dividend
adjustment bases create fake jumps at the seam).

To rerun only specific configs (e.g. borderline ones at a higher N, into a
separate directory with an independent seed), repeat --config:

    python scripts/run_permutations.py --n 20000 --base-seed 20261002 \
        --out-dir results/permutations_hp \
        --config st_reversal:2020_2024 --config price_to_ma:2015_2019

Output: one JSON per config in outputs/permutations/{factor}_{period}.json.
Configs whose JSON already exists are skipped, so an interrupted run can
simply be restarted (an existing JSON made with a different N or base seed
stops the run instead of being silently reused). Configs run in parallel,
leaving 2 cores free.

Seeds: each config gets its own SeedSequence built from (base_seed, a
stable hash of factor|period), so configs never share random draws and
the same command always reproduces the same nulls.

Every result carries the survivorship caveat: TICKERS is today's large
caps, survivors only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.analytics.significance import permutation_pvalue, permute_within_date
from quantlab.backtest.pipeline import (
    COMMISSION_MINIMUM,
    COMMISSION_RATE,
    DATA_START,
    GROSS,
    IMPACT_COEFF,
    LONG_FRAC,
    MAX_PARTICIPATION,
    PRICE_FACTOR_BUILDERS,
    REBALANCE_EVERY,
    SHORT_FRAC,
    SPREAD_BPS,
    TICKERS,
    normalization,
    run_ls_backtest,
)
from quantlab.backtest.strategy import signal_to_weights
from quantlab.config import get_settings
from quantlab.data import CachedSource, DataValidator, MarketDataset, YahooFinanceSource

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_END = "2024-12-31"
PERIODS: dict[str, tuple[str, str]] = {
    "2005_2024": ("2005-01-01", "2024-12-31"),
    "2005_2009": ("2005-01-01", "2009-12-31"),
    "2010_2014": ("2010-01-01", "2014-12-31"),
    "2015_2019": ("2015-01-01", "2019-12-31"),
    "2020_2024": ("2020-01-01", "2024-12-31"),
}
COST_MULTIPLIERS = (1.0, 2.0, 3.0)
DEFAULT_BASE_SEED = 20260927
DEFAULT_N = 1000
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "permutations"

# Filled once per worker process by _init_worker (loading prices per config
# would repeat the same 0.4s parquet read 55 times).
_WARM: MarketDataset | None = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def check_cache_complete() -> None:
    """Refuse to run if any ticker's cached range doesn't cover the data window.

    A partial cache would make CachedSource download the gap and stitch it
    onto older prices with a different adjustment basis.
    """
    source = CachedSource(YahooFinanceSource())
    start, end = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    short = []
    for sym in TICKERS:
        covered = source.cached_range(sym)
        if covered is None or covered[0] > start or covered[1] < end:
            short.append(sym)
    if short:
        sys.exit(
            f"price cache {source.cache_dir} does not cover {DATA_START}..{DATA_END} "
            f"for {len(short)} ticker(s) (e.g. {short[:5]}). Set QUANTLAB_CACHE_DIR to "
            "a cache fetched fresh over the full window; this script never downloads."
        )


def load_warm_dataset() -> MarketDataset:
    ds = CachedSource(YahooFinanceSource()).fetch(TICKERS, DATA_START, DATA_END)
    DataValidator().validate(ds).raise_for_errors()
    return ds


def _init_worker() -> None:
    global _WARM
    _WARM = load_warm_dataset()


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------
def rebalance_dates(zscores: pd.DataFrame) -> pd.DatetimeIndex:
    """The dates signal_to_weights trades on, read off its own output.

    Deriving the schedule from signal_to_weights (instead of re-coding its
    rule) guarantees the shuffle hits exactly the rows the portfolio uses.
    """
    weights = signal_to_weights(
        zscores,
        long_frac=LONG_FRAC,
        short_frac=SHORT_FRAC,
        gross=GROSS,
        rebalance_every=REBALANCE_EVERY,
    )
    return pd.DatetimeIndex(weights.index[weights.notna().any(axis=1)])


def shuffle_zscores(
    zscores: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Copy of ``zscores`` with each rebalance date's row permuted across stocks."""
    shuffled = zscores.copy()
    for date in dates:
        shuffled.loc[date] = permute_within_date(zscores.loc[date], rng)
    return shuffled


def config_entropy(base_seed: int, factor: str, period: str) -> list[int]:
    """Seed entropy unique to this config and stable across runs and machines.

    Python's built-in hash() is randomized per process, so a SHA-256 of the
    config name is used instead.
    """
    digest = hashlib.sha256(f"{factor}|{period}".encode()).digest()
    return [base_seed, int.from_bytes(digest[:8], "little")]


def _finite_or_none(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_config(
    factor: str, period: str, n: int, base_seed: int, out_dir: Path
) -> tuple[str, float]:
    """Run one (factor, period) config and write its JSON. Returns (path, seconds)."""
    t0 = time.perf_counter()
    warm = _WARM if _WARM is not None else load_warm_dataset()
    start, end = PERIODS[period]
    ds = warm.between(start, end)

    # Factor computed on the full warm history, then trimmed to the window,
    # so every period sees identical scores on shared dates.
    full = normalization().transform(PRICE_FACTOR_BUILDERS[factor]().compute(warm))
    zscores = full.loc[ds.index]

    # --- real backtest: gross (for the p-value) and 1x/2x/3x (cost stress test) ---
    real_gross = run_ls_backtest(zscores, ds, frictionless=True).summary()
    sharpe_gross = float(real_gross["sharpe"])
    real = {k: run_ls_backtest(zscores, ds, cost_multiplier=k) for k in COST_MULTIPLIERS}
    real_1x = real[1.0]
    summary = real_1x.summary()

    # --- null distribution (gross, like the real Sharpe it is compared with) ---
    dates = rebalance_dates(zscores)
    entropy = config_entropy(base_seed, factor, period)
    child_seeds = np.random.SeedSequence(entropy).spawn(n)
    null_sharpes = np.full(n, np.nan)
    null_turnover = np.full(n, np.nan)
    for i, seed in enumerate(child_seeds):
        shuffled = shuffle_zscores(zscores, dates, np.random.default_rng(seed))
        null = run_ls_backtest(shuffled, ds, frictionless=True).summary()
        null_sharpes[i] = null["sharpe"]
        null_turnover[i] = null["annualized_turnover"]
    p_value, n_dropped = permutation_pvalue(sharpe_gross, null_sharpes)

    # Names with a usable score on each rebalance date, averaged per year.
    counts = zscores.loc[dates].notna().sum(axis=1)
    names_per_year = {
        str(year): round(float(v), 1) for year, v in counts.groupby(counts.index.year).mean().items()
    }

    equity = real_1x.equity_curve
    record = {
        "factor": factor,
        "period": period,
        "start": start,
        "end": end,
        "base_seed": base_seed,
        "seed_entropy": entropy,
        "n_shuffles": n,
        "p_value": float(p_value),
        "p_value_basis": "sharpe_gross vs null_sharpes_gross (both frictionless)",
        "n_dropped": int(n_dropped),
        "sharpe_gross": _finite_or_none(sharpe_gross),
        "null_sharpes_gross": [_finite_or_none(s) for s in null_sharpes],
        "sharpe_1x": _finite_or_none(summary["sharpe"]),
        "sharpe_2x": _finite_or_none(real[2.0].summary()["sharpe"]),
        "sharpe_3x": _finite_or_none(real[3.0].summary()["sharpe"]),
        "total_return_1x": _finite_or_none(summary["total_return"]),
        "max_drawdown_1x": _finite_or_none(summary["max_drawdown"]),
        "hit_rate_1x": _finite_or_none(summary["hit_rate"]),
        "turnover_annualized_1x": _finite_or_none(summary["annualized_turnover"]),
        "turnover_annualized_gross": _finite_or_none(real_gross["annualized_turnover"]),
        "null_turnover_annualized_mean": _finite_or_none(np.nanmean(null_turnover)),
        "equity_curve_1x": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "values": [float(v) for v in equity.to_numpy()],
        },
        "names_per_year": names_per_year,
        "n_rebalance_dates": len(dates),
        "settings": {
            "data_start": DATA_START,
            "n_tickers": len(TICKERS),
            "long_frac": LONG_FRAC,
            "short_frac": SHORT_FRAC,
            "gross": GROSS,
            "rebalance_every": REBALANCE_EVERY,
            "spread_bps_1x": SPREAD_BPS,
            "commission_rate_1x": COMMISSION_RATE,
            "commission_minimum_1x": COMMISSION_MINIMUM,
            "impact_coeff": IMPACT_COEFF,
            "max_participation": MAX_PARTICIPATION,
            "cost_scaling": "1x/2x/3x scale spread_bps and commission rate+minimum; "
            "impact_coeff unchanged. gross = no spread, commission or impact "
            "(participation cap kept); the real gross run and all shuffles are gross",
        },
        "provenance": {
            "git_commit": _git_commit(),
            "cache_dir": str(get_settings().cache_dir),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - t0, 1),
        },
        "caveat": "Survivors-only universe (today's large caps); results overstate "
        "what was achievable historically.",
    }

    # Atomic write: a killed run never leaves a half-written JSON that the
    # resume logic would mistake for a finished config.
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{factor}_{period}.json"
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=1)
        os.chmod(tmp, 0o644)  # mkstemp creates owner-only files
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return str(path), time.perf_counter() - t0


def _run_config_star(args: tuple[str, str, int, int, Path]) -> tuple[str, float]:
    return run_config(*args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def config_arg(text: str) -> tuple[str, str]:
    """argparse type for --config: "factor:period" -> (factor, period)."""
    factor, sep, period = text.partition(":")
    if not sep or factor not in PRICE_FACTOR_BUILDERS or period not in PERIODS:
        raise argparse.ArgumentTypeError(
            f"expected FACTOR:PERIOD with FACTOR in {sorted(PRICE_FACTOR_BUILDERS)} "
            f"and PERIOD in {list(PERIODS)}, got {text!r}"
        )
    return factor, period


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="shuffles per config")
    parser.add_argument("--factor", choices=sorted(PRICE_FACTOR_BUILDERS), help="run only this factor")
    parser.add_argument("--period", choices=list(PERIODS), help="run only this period")
    parser.add_argument(
        "--config",
        type=config_arg,
        action="append",
        metavar="FACTOR:PERIOD",
        help="run only this config (repeatable; replaces --factor/--period)",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="parallel processes (default: all cores but 2)",
    )
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be >= 1")
    if args.config and (args.factor or args.period):
        parser.error("--config cannot be combined with --factor or --period")
    return args


def selected_configs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The (factor, period) pairs to run: --config list, else the factor x period grid."""
    if args.config:
        return list(dict.fromkeys(args.config))  # drop repeats, keep order
    factors = [args.factor] if args.factor else list(PRICE_FACTOR_BUILDERS)
    periods = [args.period] if args.period else list(PERIODS)
    return [(f, p) for f in factors for p in periods]


def main() -> None:
    args = parse_args()
    check_cache_complete()

    configs = selected_configs(args)
    todo, mismatched = [], []
    for f, p in configs:
        path = args.out_dir / f"{f}_{p}.json"
        if not path.exists():
            todo.append((f, p))
            continue
        done = json.loads(path.read_text())
        if (done.get("n_shuffles"), done.get("base_seed")) != (args.n, args.base_seed):
            mismatched.append(f"{path.name} (n={done.get('n_shuffles')}, seed={done.get('base_seed')})")
    if mismatched:
        sys.exit(
            f"existing results were made with different settings than n={args.n}, "
            f"base_seed={args.base_seed}: {mismatched}. Move or delete them, or use --out-dir."
        )
    print(
        f"{len(configs)} config(s), {len(configs) - len(todo)} already done, "
        f"{len(todo)} to run | N={args.n} base_seed={args.base_seed} "
        f"cache={get_settings().cache_dir}"
    )
    if not todo:
        return

    # Longest windows first, so a 20-year config isn't left running alone at the end.
    todo.sort(key=lambda fp: pd.Timestamp(PERIODS[fp[1]][0]) - pd.Timestamp(PERIODS[fp[1]][1]))
    tasks = [(f, p, args.n, args.base_seed, args.out_dir) for f, p in todo]
    workers = min(args.workers, len(tasks))
    t0 = time.perf_counter()
    if workers == 1:
        _init_worker()
        results = map(_run_config_star, tasks)
        for i, (path, secs) in enumerate(results, 1):
            print(f"[{i}/{len(tasks)}] {path} ({secs:.1f}s)", flush=True)
    else:
        with mp.Pool(processes=workers, initializer=_init_worker) as pool:
            done = pool.imap_unordered(_run_config_star, tasks)
            for i, (path, secs) in enumerate(done, 1):
                print(f"[{i}/{len(tasks)}] {path} ({secs:.1f}s)", flush=True)
    print(f"finished in {time.perf_counter() - t0:.1f}s with {workers} worker(s)")


if __name__ == "__main__":
    main()
