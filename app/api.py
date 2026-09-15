"""Placeholder implementation of the QFRP app data contract.

Returns made-up data in exactly the shape described in CONTRACT.md, so the
frontend can be built before the real research and statistics layer exists.

    EVERY NUMBER PRODUCED BY THIS MODULE IS FABRICATED.

Nothing here touches quantlab. Nothing here is research. This file is replaced
wholesale in week 2 by an implementation that reads precomputed real results.
The function names, argument names, and returned field names are the contract
and will NOT change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "AnalysisResult",
    "CostPoint",
    "FactorInfo",
    "PeriodInfo",
    "Verdict",
    "get_analysis",
    "list_factors",
    "list_periods",
    "verdict_for",
]


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
    multiplier: float
    sharpe: float
    total_return: float


@dataclass(frozen=True)
class AnalysisResult:
    # identity
    factor_key: str
    factor_label: str
    period_label: str
    start_date: str
    end_date: str
    # headline performance
    sharpe: float
    total_return: float
    annualized_vol: float
    max_drawdown: float  # negative by convention
    hit_rate: float
    n_trades: int
    # curves: DatetimeIndex, columns "strategy" and "benchmark", both start at 1.0
    equity_curve: pd.DataFrame
    # significance
    p_value: float
    null_sharpes: list[float]
    n_permutations: int
    # cost sensitivity
    cost_stress: list[CostPoint]
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


# --------------------------------------------------------------------------
# Menus
# --------------------------------------------------------------------------

_FACTORS: tuple[FactorInfo, ...] = (
    FactorInfo(
        "st_reversal",
        "Short-Term Reversal",
        "Buys recent losers and sells recent winners over a one-week window.",
    ),
    FactorInfo(
        "amihud_illiquidity",
        "Amihud Illiquidity",
        "Favours stocks whose prices move a lot per dollar traded.",
    ),
    FactorInfo(
        "price_to_ma",
        "Price to Moving Average",
        "Compares price to its own recent average.",
    ),
    FactorInfo(
        "momentum",
        "Momentum (12-1)",
        "Buys stocks that rose over the past year, skipping the last month.",
    ),
    FactorInfo(
        "realized_vol",
        "Realized Volatility",
        "Ranks stocks by how much their returns have varied recently.",
    ),
    FactorInfo(
        "ts_momentum",
        "Time-Series Momentum",
        "Each stock's own trend rather than its rank against others.",
    ),
)

_PERIODS: tuple[PeriodInfo, ...] = (
    PeriodInfo("2015_2020", "2015-2020", "2015-01-01", "2019-12-31"),
    PeriodInfo("2020_2024", "2020-2024", "2020-01-01", "2024-12-31"),
    PeriodInfo("2015_2024", "Full sample", "2015-01-01", "2024-12-31"),
)

# Placeholder Sharpe values, roughly in the range real results fall in so the
# layout is designed against realistic numbers. NOT real results.
_PLACEHOLDER_SHARPE: dict[tuple[str, str], float] = {
    ("st_reversal", "2015_2020"): 0.50,
    ("st_reversal", "2020_2024"): 0.39,
    ("st_reversal", "2015_2024"): 0.44,
    ("amihud_illiquidity", "2015_2020"): 0.97,
    ("amihud_illiquidity", "2020_2024"): 0.30,
    ("amihud_illiquidity", "2015_2024"): 0.61,
    ("price_to_ma", "2015_2020"): 0.73,
    ("price_to_ma", "2020_2024"): 0.07,
    ("price_to_ma", "2015_2024"): 0.38,
    ("momentum", "2015_2020"): 0.06,
    ("momentum", "2020_2024"): -0.02,
    ("momentum", "2015_2024"): 0.01,
    ("realized_vol", "2015_2020"): -0.21,
    ("realized_vol", "2020_2024"): -0.60,
    ("realized_vol", "2015_2024"): -0.39,
    ("ts_momentum", "2015_2020"): -0.11,
    ("ts_momentum", "2020_2024"): -0.34,
    ("ts_momentum", "2015_2024"): -0.22,
}

_LIMITATIONS: tuple[str, ...] = (
    "PLACEHOLDER TEXT. Michael writes the real limitations in week 2.",
    "Daily price data only; no intraday execution modelling.",
    "149 US large-cap names; smaller and delisted companies are excluded.",
    "Fixed commission and slippage assumptions rather than live quotes.",
    "Results are precomputed for a fixed menu of configurations.",
    "Research output. Not investment advice.",
)

_N_PERMUTATIONS = 1000

_FACTORS_BY_KEY = {f.key: f for f in _FACTORS}
_PERIODS_BY_KEY = {p.key: p for p in _PERIODS}


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------


def list_factors() -> list[FactorInfo]:
    """The factor menu, in display order."""
    return list(_FACTORS)


def list_periods() -> list[PeriodInfo]:
    """The date-range menu, in display order."""
    return list(_PERIODS)


def get_analysis(factor_key: str, period_key: str) -> AnalysisResult:
    """Fabricated analysis for one (factor, period) pair.

    Deterministic: the same keys always give the same numbers, so the UI does
    not flicker between reruns.
    """
    if factor_key not in _FACTORS_BY_KEY:
        raise KeyError(f"unknown factor_key: {factor_key!r}")
    if period_key not in _PERIODS_BY_KEY:
        raise KeyError(f"unknown period_key: {period_key!r}")

    factor = _FACTORS_BY_KEY[factor_key]
    period = _PERIODS_BY_KEY[period_key]

    # Seed off the keys so results are stable across reruns.
    rng = np.random.default_rng(abs(hash((factor_key, period_key))) % (2**32))

    sharpe = _PLACEHOLDER_SHARPE[(factor_key, period_key)]

    dates = pd.bdate_range(period.start_date, period.end_date)
    n = len(dates)

    ann_vol = 0.10 + 0.04 * rng.random()
    daily_vol = ann_vol / np.sqrt(252.0)
    daily_mean = sharpe * daily_vol

    strat_returns = rng.normal(daily_mean, daily_vol, n)
    strat_curve = np.cumprod(1.0 + strat_returns)

    bench_returns = rng.normal(0.09 / 252.0, 0.18 / np.sqrt(252.0), n)
    bench_curve = np.cumprod(1.0 + bench_returns)

    equity_curve = pd.DataFrame(
        {"strategy": strat_curve, "benchmark": bench_curve}, index=dates
    )

    running_peak = np.maximum.accumulate(strat_curve)
    max_drawdown = float((strat_curve / running_peak - 1.0).min())

    null_sharpes = rng.normal(0.0, 0.32, _N_PERMUTATIONS)
    p_value = float((null_sharpes >= sharpe).mean())

    total_return = float(strat_curve[-1] - 1.0)
    cost_stress = [
        CostPoint(
            multiplier=m,
            sharpe=round(sharpe - 0.18 * (m - 1.0), 3),
            total_return=round(total_return - 0.09 * (m - 1.0), 4),
        )
        for m in (1.0, 2.0, 3.0)
    ]

    return AnalysisResult(
        factor_key=factor.key,
        factor_label=factor.label,
        period_label=period.label,
        start_date=period.start_date,
        end_date=period.end_date,
        sharpe=round(sharpe, 3),
        total_return=round(total_return, 4),
        annualized_vol=round(float(ann_vol), 4),
        max_drawdown=round(max_drawdown, 4),
        hit_rate=round(float((strat_returns > 0).mean()), 4),
        n_trades=int(rng.integers(40, 400)),
        equity_curve=equity_curve,
        p_value=round(p_value, 4),
        null_sharpes=[float(x) for x in null_sharpes],
        n_permutations=_N_PERMUTATIONS,
        cost_stress=cost_stress,
        limitations=list(_LIMITATIONS),
    )


def verdict_for(result: AnalysisResult, n_configs_viewed: int) -> Verdict:
    """Apply a multiple-testing penalty and render a verdict.

    Placeholder rule (Sidak-style adjustment). The real thresholds and method
    are Michael's decision in week 2; only the returned shape is fixed.
    """
    n = max(1, int(n_configs_viewed))
    raw_p = float(result.p_value)
    adjusted_p = float(1.0 - (1.0 - raw_p) ** n)

    survives_2x = any(
        c.multiplier == 2.0 and c.sharpe > 0.0 for c in result.cost_stress
    )

    if adjusted_p > 0.10:
        label = "FAIL"
        headline = "Indistinguishable from luck"
    elif adjusted_p < 0.05 and survives_2x:
        label = "PASS"
        headline = "Edge survives testing"
    else:
        label = "INCONCLUSIVE"
        headline = "Not enough evidence either way"

    explanation = (
        f"Across {_N_PERMUTATIONS} random reshuffles, {raw_p:.0%} did as well "
        f"or better. You have looked at {n} configuration"
        f"{'s' if n != 1 else ''} this session, which raises the chance that "
        f"one looks good by accident; adjusted for that, the figure is "
        f"{adjusted_p:.0%}."
    )

    return Verdict(
        label=label,
        headline=headline,
        raw_p_value=round(raw_p, 4),
        adjusted_p_value=round(adjusted_p, 4),
        n_configs_viewed=n,
        explanation=explanation,
    )
