"""Real implementation of the QFRP app data contract (see CONTRACT.md).

Reads the precomputed permutation-test results written by
``scripts/run_permutations.py`` (one JSON per factor x period) and turns
them into the shapes the web app displays. Nothing here runs a backtest:
every call is a file read plus a few lines of arithmetic, so it is instant.

Where things come from, so nothing is defined twice:

- the factor menu is ``quantlab.backtest.pipeline.PRICE_FACTOR_BUILDERS``
  (the catalog the permutation runner iterates over);
- the period menu is ``PERIODS`` in ``scripts/run_permutations.py``;
- the multiple-testing adjustment is
  ``quantlab.analytics.significance.bonferroni_adjust``.

The results directory defaults to ``results/permutations`` at the repo root
(tracked in git: copies of the runner's ``outputs/permutations`` JSONs for
the configurations the app offers) and can be overridden with the
``QFRP_RESULTS_DIR`` environment variable (used by the tests).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from quantlab.analytics.performance import annualized_vol
from quantlab.analytics.significance import bonferroni_adjust, permutation_pvalue
from quantlab.backtest.pipeline import PRICE_FACTOR_BUILDERS

__all__ = [
    "EXCLUDED_FACTORS",
    "FAIL_ABOVE",
    "PASS_BELOW",
    "SURVIVORSHIP_NOTE",
    "AnalysisResult",
    "CostPoint",
    "FactorInfo",
    "PeriodInfo",
    "ResultNotComputedError",
    "Verdict",
    "get_analysis",
    "list_factors",
    "list_periods",
    "verdict_for",
]

# --------------------------------------------------------------------------
# Verdict cutoffs (applied to the multiple-testing-ADJUSTED p-value)
# --------------------------------------------------------------------------
# Why adjust at all: Harvey, Liu & Zhu (2016), "...and the Cross-Section of
# Expected Returns" (Review of Financial Studies 29(1), 5-68), show that
# once hundreds of factors have been tried, the usual p < 0.05 bar lets
# through mostly false discoveries; a p-value only means what it claims if
# it accounts for how many things were tested. Here that count is the number
# of configurations the user has viewed, and the adjustment is Bonferroni.
#
#   adjusted p <  PASS_BELOW               -> PASS
#   PASS_BELOW <= adjusted p <= FAIL_ABOVE -> INCONCLUSIVE
#   adjusted p >  FAIL_ABOVE               -> FAIL
PASS_BELOW = 0.05
FAIL_ABOVE = 0.20

# Factors computed by the permutation runner but NOT offered in the app.
#
# roll_spread: Roll's estimator clamps the spread at zero, so on a typical
# rebalance date ~54 of 148 stocks tie at the best score while the long leg
# holds only 30. Which tied stocks get bought is then decided by how the sort
# breaks ties (effectively by column/sector order), not by the factor -- in
# the real portfolio AND in every shuffle. The shuffled long legs are skewed
# (e.g. Energy 1% vs 6% of the universe), so the null distribution is biased
# (null means 5-21 standard errors from zero) and the p-value is meaningless.
# It stays in PRICE_FACTOR_BUILDERS; it is only hidden here.
EXCLUDED_FACTORS: tuple[str, ...] = ("roll_spread",)

# Appended to every PASS explanation when non-empty (text by Michael).
SURVIVORSHIP_NOTE = ""

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results" / "permutations"


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorInfo:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class PeriodInfo:
    key: str
    label: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class CostPoint:
    multiplier: float  # 1.0, 2.0, 3.0 x the base cost assumptions
    sharpe: float | None  # None if the backtest's Sharpe was undefined


@dataclass(frozen=True)
class AnalysisResult:
    # identity
    factor_key: str
    factor_label: str
    period_label: str
    start_date: str
    end_date: str
    # headline performance: the real strategy, NET of 1x trading costs
    sharpe_1x: float | None
    total_return: float
    annualized_vol: float
    max_drawdown: float  # negative by convention
    hit_rate: float  # fraction of days with a positive return
    turnover: float  # annualized; 1.0 = the whole book traded once a year
    # curve: DatetimeIndex, one column "strategy", 1.0 at the start (1x costs)
    equity_curve: pd.DataFrame
    # significance: GROSS (no costs) real Sharpe vs GROSS shuffled Sharpes
    sharpe_gross: float | None
    null_sharpes_gross: list[float]  # raw and unbinned; NaN draws removed
    n_permutations: int  # shuffles run (including any dropped as NaN)
    n_dropped: int  # shuffles whose Sharpe was NaN and so were excluded
    p_value: float  # raw, one-sided
    adjusted_p_value: float  # bonferroni_adjust(p_value, k)
    k: int  # number of configurations the adjustment accounts for
    # "worse than random" test: same shuffles, opposite direction
    reverse_p_value: float  # permutation_pvalue(-sharpe_gross, -nulls)
    adjusted_reverse_p_value: float  # bonferroni_adjust(reverse_p_value, k)
    null_turnover: float  # annualized turnover of the shuffled portfolios
    # cost sensitivity of the real strategy
    cost_stress: list[CostPoint]
    # universe breadth: average names with a score on each year's rebalance dates
    names_per_year: dict[str, float]
    # scope
    limitations: list[str]


@dataclass(frozen=True)
class Verdict:
    label: str  # "PASS" | "FAIL" | "INCONCLUSIVE"
    headline: str
    raw_p_value: float
    adjusted_p_value: float
    n_configs_viewed: int
    explanation: str
    # True when the strategy did significantly WORSE than random (label stays FAIL)
    reversed: bool
    raw_reverse_p_value: float
    adjusted_reverse_p_value: float  # bonferroni_adjust(raw_reverse_p_value, n)


class ResultNotComputedError(KeyError):
    """The (factor, period) is valid but its results file doesn't exist yet.

    Subclasses KeyError so callers that already map KeyError to "not found"
    keep working; catch this first to distinguish "not computed yet" from
    "no such factor/period".
    """

    def __str__(self) -> str:  # KeyError would wrap the message in quotes
        return str(self.args[0]) if self.args else ""


# --------------------------------------------------------------------------
# Menus
# --------------------------------------------------------------------------

# Display text only. The keys and their order come from
# PRICE_FACTOR_BUILDERS minus EXCLUDED_FACTORS; the check below fails loudly
# if they drift.
_FACTOR_TEXT: dict[str, tuple[str, str]] = {
    "momentum": (
        "Momentum (12-1)",
        "Buys stocks that rose most over the past year, skipping the last month; "
        "sells the biggest losers.",
    ),
    "ts_momentum": (
        "Time-Series Momentum",
        "Ranks stocks by the strength of their own six-month trend, scaled by "
        "their volatility.",
    ),
    "st_reversal": (
        "Short-Term Reversal",
        "Buys last week's biggest losers and sells last week's biggest winners.",
    ),
    "price_to_ma": (
        "Price to Moving Average",
        "Buys stocks trading far below their one-month average price; sells "
        "those far above it.",
    ),
    "realized_vol": (
        "Low Volatility",
        "Buys the calmest stocks (lowest three-month volatility) and sells the "
        "most volatile.",
    ),
    "ewma_vol": (
        "Low Volatility (EWMA)",
        "Like Low Volatility, but weights recent days more so it reacts faster.",
    ),
    "market_beta": (
        "Low Beta",
        "Buys stocks that move least with the market and sells those that move "
        "most.",
    ),
    "pca_reversal": (
        "Residual Reversal",
        "Buys stocks that fell for reasons the broad market moves don't "
        "explain, betting they bounce back.",
    ),
    "amihud_illiquidity": (
        "Amihud Illiquidity",
        "Buys stocks whose prices move most per dollar traded (least liquid) "
        "and sells the most liquid.",
    ),
    "dollar_volume": (
        "Dollar Volume",
        "Buys the most heavily traded stocks and sells the least traded.",
    ),
}
if not set(EXCLUDED_FACTORS) <= set(PRICE_FACTOR_BUILDERS):
    raise RuntimeError(f"EXCLUDED_FACTORS not in the catalog: {EXCLUDED_FACTORS}")
_MENU_FACTORS: list[str] = [k for k in PRICE_FACTOR_BUILDERS if k not in EXCLUDED_FACTORS]
if list(_FACTOR_TEXT) != _MENU_FACTORS:
    raise RuntimeError(
        "app/api.py factor text is out of sync with PRICE_FACTOR_BUILDERS minus "
        f"EXCLUDED_FACTORS: {sorted(set(_FACTOR_TEXT) ^ set(_MENU_FACTORS))}"
    )


def _load_runner() -> ModuleType:
    """Load scripts/run_permutations.py (not a package) to read its PERIODS.

    Importing it has no side effects: the script only runs under
    ``if __name__ == "__main__"``.
    """
    path = _REPO_ROOT / "scripts" / "run_permutations.py"
    spec = importlib.util.spec_from_file_location("_qfrp_run_permutations", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PERIOD_DATES: dict[str, tuple[str, str]] = dict(_load_runner().PERIODS)


def _period_info(key: str) -> PeriodInfo:
    start, end = _PERIOD_DATES[key]
    return PeriodInfo(key, f"{start[:4]}-{end[:4]}", start, end)


def list_factors() -> list[FactorInfo]:
    """The factor menu (10 price factors: the catalog minus EXCLUDED_FACTORS)."""
    return [FactorInfo(key, *_FACTOR_TEXT[key]) for key in _MENU_FACTORS]


def list_periods() -> list[PeriodInfo]:
    """The date-range menu (the 5 periods), in display order."""
    return [_period_info(key) for key in _PERIOD_DATES]


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _results_dir() -> Path:
    override = os.environ.get("QFRP_RESULTS_DIR")
    return Path(override) if override else _DEFAULT_RESULTS_DIR


def _check_k(k: Any, name: str) -> int:
    # bool is a subclass of int; True would silently mean k=1.
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{name} must be an integer >= 1, got {k!r}")
    if k < 1:
        raise ValueError(f"{name} must be >= 1, got {k}")
    return k


def _validate_keys(factor_key: str, period_key: str) -> None:
    if factor_key in EXCLUDED_FACTORS:
        raise KeyError(
            f"factor_key {factor_key!r} is excluded from the app (see EXCLUDED_FACTORS "
            f"in app/api.py); valid: {_MENU_FACTORS}"
        )
    if factor_key not in _MENU_FACTORS:
        raise KeyError(f"unknown factor_key {factor_key!r}; valid: {_MENU_FACTORS}")
    if period_key not in _PERIOD_DATES:
        raise KeyError(f"unknown period_key {period_key!r}; valid: {list(_PERIOD_DATES)}")


def _load_record(factor_key: str, period_key: str) -> dict[str, Any]:
    path = _results_dir() / f"{factor_key}_{period_key}.json"
    if not path.is_file():
        raise ResultNotComputedError(
            f"results for {factor_key} / {period_key} are not computed yet "
            f"(expected {path}); run scripts/run_permutations.py and copy the "
            "JSON into results/permutations/"
        )
    record: dict[str, Any] = json.loads(path.read_text())
    if (record.get("factor"), record.get("period")) != (factor_key, period_key):
        raise ValueError(
            f"{path.name} contains {record.get('factor')}/{record.get('period')}, "
            f"not {factor_key}/{period_key}"
        )
    return record


def _limitations(record: dict[str, Any]) -> list[str]:
    s = record["settings"]
    names = list(record["names_per_year"].values())
    n = record["n_shuffles"]
    lo, hi = min(names), max(names)
    breadth = (
        f"{lo:.0f} stocks had a score throughout this period."
        if round(lo) == round(hi)
        else f"Stocks join only once they started trading: between {lo:.0f} and "
        f"{hi:.0f} stocks had a score in this period."
    )
    return [
        f"Universe: {s['n_tickers']} US large-cap stocks picked from today's (2026) "
        "market. Companies that were delisted, acquired or went bankrupt earlier "
        "(e.g. Lehman Brothers in 2008) are missing, so these historical results "
        "are biased by hindsight (survivorship bias).",
        breadth,
        f"The p-value compares the strategy's Sharpe ratio BEFORE trading costs with "
        f"{n:,} random reshufflings of the factor's stock rankings, also before costs. "
        f"Reshuffled portfolios trade far more ({record['null_turnover_annualized_mean']:.1f}x "
        f"a year vs {record['turnover_annualized_gross']:.1f}x), so comparing after "
        "costs would unfairly handicap them. Whether any edge survives costs is shown "
        "separately in the cost stress test.",
        f"Trading costs are modelled, not measured: a {s['spread_bps_1x']:g} bp "
        f"bid-ask spread, ${s['commission_rate_1x']:g} per share commission "
        f"(${s['commission_minimum_1x']:g} minimum), square-root market impact, at most "
        f"{s['max_participation']:.0%} of a day's volume, filled at the next day's open.",
        "Prices are Yahoo Finance daily bars adjusted for splits and dividends; Yahoo "
        "occasionally mis-adjusts (e.g. around Danaher's 2016 spin-off).",
        "The reshuffle test treats each month's ranking as independent; it does not "
        "model how slowly a factor's rankings change over time.",
        f"The adjusted p-value multiplies the raw p-value by the number of "
        f"configurations viewed (Bonferroni), which is deliberately cautious. With "
        f"{n:,} reshuffles the smallest possible raw p-value is 1/{n + 1:,}.",
        "Research output for learning purposes. Not investment advice.",
    ]


def _finite(x: Any) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


def get_analysis(factor_key: str, period_key: str, k: int = 1) -> AnalysisResult:
    """Precomputed analysis for one (factor, period) configuration.

    ``k`` is the number of configurations to adjust the p-value for (the
    number the user has viewed); ``adjusted_p_value`` is
    ``bonferroni_adjust(p_value, k)``.

    Raises KeyError for an unknown factor or period, ResultNotComputedError
    (a KeyError) if the results file doesn't exist yet, TypeError/ValueError
    if ``k`` is not an integer >= 1.
    """
    _validate_keys(factor_key, period_key)
    k = _check_k(k, "k")
    record = _load_record(factor_key, period_key)
    period = _period_info(period_key)

    curve = record["equity_curve_1x"]
    equity = pd.Series(curve["values"], index=pd.DatetimeIndex(curve["dates"]), dtype=float)
    # The engine's first-day return is 0 (equity starts at initial capital),
    # so fillna(0) reproduces the backtest's own annualized_vol exactly.
    returns = equity.pct_change().fillna(0.0)

    nulls = [float(x) for x in record["null_sharpes_gross"] if x is not None]
    p_value = float(record["p_value"])
    sharpe_gross = _finite(record["sharpe_gross"])
    if sharpe_gross is None:  # the runner can't produce this; permutation_pvalue raises on NaN
        raise ValueError(f"{factor_key}_{period_key}.json has no finite sharpe_gross")
    # Worse-than-random test: flip the sign of every Sharpe, so "at least as
    # large" becomes "at least as small". Same hand-written p-value function.
    reverse_p_value, _ = permutation_pvalue(-sharpe_gross, -np.asarray(nulls, dtype=float))

    return AnalysisResult(
        factor_key=factor_key,
        factor_label=_FACTOR_TEXT[factor_key][0],
        period_label=period.label,
        start_date=period.start_date,
        end_date=period.end_date,
        sharpe_1x=_finite(record["sharpe_1x"]),
        total_return=float(record["total_return_1x"]),
        annualized_vol=annualized_vol(returns),
        max_drawdown=float(record["max_drawdown_1x"]),
        hit_rate=float(record["hit_rate_1x"]),
        turnover=float(record["turnover_annualized_1x"]),
        equity_curve=(equity / equity.iloc[0]).to_frame("strategy"),
        sharpe_gross=sharpe_gross,
        null_sharpes_gross=nulls,
        n_permutations=int(record["n_shuffles"]),
        n_dropped=int(record["n_dropped"]),
        p_value=p_value,
        adjusted_p_value=bonferroni_adjust(p_value, k),
        k=k,
        reverse_p_value=float(reverse_p_value),
        adjusted_reverse_p_value=bonferroni_adjust(float(reverse_p_value), k),
        null_turnover=float(record["null_turnover_annualized_mean"]),
        cost_stress=[
            CostPoint(multiplier=m, sharpe=_finite(record[f"sharpe_{m:g}x"]))
            for m in (1.0, 2.0, 3.0)
        ],
        names_per_year={str(y): float(v) for y, v in record["names_per_year"].items()},
        limitations=_limitations(record),
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def _band(adjusted_p: float) -> tuple[str, str]:
    """(label, headline) for an adjusted p-value, using the cutoffs above."""
    if adjusted_p < PASS_BELOW:
        return "PASS", "Signal beats random rankings"
    if adjusted_p <= FAIL_ABOVE:
        return "INCONCLUSIVE", "Not enough evidence either way"
    return "FAIL", "Indistinguishable from luck"


def _share_below(result: AnalysisResult) -> str:
    """Percent of null Sharpes strictly below the real gross Sharpe, e.g. "97.3".

    Rounded DOWN to one decimal so the text never overstates: "100" only
    when every shuffle did worse.
    """
    nulls = result.null_sharpes_gross
    if result.sharpe_gross is None or not nulls:
        return "0"
    below = sum(1 for x in nulls if x < result.sharpe_gross)
    share = math.floor(1000 * below / len(nulls)) / 10
    return f"{share:.1f}".removesuffix(".0")


def _is_reversed(label: str, adjusted_reverse_p: float) -> bool:
    """Significantly WORSE than random: a FAIL whose reverse p clears PASS_BELOW."""
    return label == "FAIL" and adjusted_reverse_p < PASS_BELOW


def verdict_for(result: AnalysisResult, n_configs_viewed: int) -> Verdict:
    """Adjust ``result.p_value`` for ``n_configs_viewed`` tests and band it.

    The verdict is about the SIGNAL (gross of costs): does the factor's
    ranking of stocks beat random rankings? Whether it makes money after
    costs is a separate question, noted in the explanation when it doesn't.

    A FAIL is also checked in the other direction: if the strategy did
    significantly WORSE than random (adjusted reverse p < PASS_BELOW) the
    label stays FAIL but ``reversed`` is True and the text says so.
    Explanation wording by Michael.
    """
    n = _check_k(n_configs_viewed, "n_configs_viewed")
    raw_p = float(result.p_value)
    adjusted_p = bonferroni_adjust(raw_p, n)
    label, headline = _band(adjusted_p)
    raw_rev = float(result.reverse_p_value)
    adjusted_rev = bonferroni_adjust(raw_rev, n)
    is_reversed = _is_reversed(label, adjusted_rev)
    if is_reversed:
        headline = "Consistently worse than random"
    tests = f"{n} test{'s' if n != 1 else ''}"

    if label == "PASS":
        explanation = (
            f"This strategy beat {_share_below(result)}% of "
            f"{result.n_permutations:,} random versions of itself. Even counting "
            f"the {tests} you've run this session, luck alone would rarely produce "
            "a result this strong. That's evidence the signal is real, before "
            "trading costs."
        )
    elif label == "INCONCLUSIVE":
        explanation = (
            "This strategy did better than most random versions, but not by "
            f"enough to rule out luck given the {tests} you've run. It might be "
            "real. This data can't tell."
        )
    elif is_reversed:
        explanation = (
            "This strategy did consistently worse than random, beating only "
            f"{_share_below(result)}% of {result.n_permutations:,} random versions. "
            "The opposite strategy would have looked strong on this data."
        )
    else:
        explanation = (
            "Random versions of this strategy did about as well or better. "
            "There's no evidence the signal is anything more than luck."
        )

    if n > 1:
        # Same logic applied to the unadjusted p-values (a single test).
        single = _band(raw_p)[0]
        if _is_reversed(single, raw_rev):
            single += " (consistently worse than random)"
        explanation += f" If this were your only test, the verdict would be {single}."
    if result.sharpe_1x is not None and result.sharpe_1x <= 0:
        explanation += " After realistic trading costs, this strategy lost money."
    if label == "PASS" and SURVIVORSHIP_NOTE:
        explanation += f" {SURVIVORSHIP_NOTE}"

    return Verdict(
        label=label,
        headline=headline,
        raw_p_value=raw_p,
        adjusted_p_value=adjusted_p,
        n_configs_viewed=n,
        explanation=explanation,
        reversed=is_reversed,
        raw_reverse_p_value=raw_rev,
        adjusted_reverse_p_value=adjusted_rev,
    )
