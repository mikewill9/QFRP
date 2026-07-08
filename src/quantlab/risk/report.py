"""Consolidated risk report.

:meth:`RiskReport.from_backtest` distills a
:class:`~quantlab.backtest.results.BacktestResult` into one object: a
headline metric table, per-bar exposure frames, the drawdown episode
table, and any limit breaches at the end of the run. ``print(report)``
gives the text summary; the frames stay available for plotting or
deeper inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from quantlab.risk.drawdown import drawdown_episodes, drawdown_series, ulcer_index
from quantlab.risk.exposures import gross_net_exposure, sector_exposures
from quantlab.risk.limits import LimitBreach, PortfolioState, RiskLimit, RiskMonitor
from quantlab.risk.metrics import annualized_vol, beta, expected_shortfall, value_at_risk
from quantlab.types import WideFrame

if TYPE_CHECKING:  # risk may import backtest, but only the type is needed
    from collections.abc import Mapping

    from quantlab.backtest.results import BacktestResult

__all__ = ["RiskReport"]


@dataclass(frozen=True, slots=True)
class RiskReport:
    """Risk summary of one strategy run.

    Attributes
    ----------
    summary:
        Headline risk numbers (vol, VaR/ES at ``var_level``, drawdown
        stats, average exposures, optional benchmark beta).
    exposures:
        Per-bar gross and net exposure.
    sectors:
        Per-bar net exposure by sector (``None`` when no mapping given).
    drawdowns:
        Episode table from :func:`~quantlab.risk.drawdown.drawdown_episodes`.
    breaches:
        Limit breaches evaluated on the final state (empty without limits).
    var_level:
        Confidence level used for the VaR/ES rows.
    """

    summary: pd.Series
    exposures: pd.DataFrame
    sectors: WideFrame | None
    drawdowns: pd.DataFrame
    breaches: list[LimitBreach] = field(default_factory=list)
    var_level: float = 0.95

    @classmethod
    def from_backtest(
        cls,
        result: BacktestResult,
        benchmark: pd.Series | None = None,
        sectors: Mapping[str, str] | pd.Series | None = None,
        limits: list[RiskLimit] | None = None,
        var_level: float = 0.95,
    ) -> RiskReport:
        """Build the report from a backtest result.

        ``benchmark`` (a return series) adds a beta row; ``sectors`` adds
        the sector exposure frame; ``limits`` are checked against the
        final weights and full return history.
        """
        returns = result.returns
        weights = result.weights
        exposures = gross_net_exposure(weights)
        dd = drawdown_series(returns)
        episodes = drawdown_episodes(returns)

        summary: dict[str, float] = {
            "annualized_vol": annualized_vol(returns),
            f"var_{var_level:.0%}_hist": value_at_risk(returns, var_level, "historical"),
            f"var_{var_level:.0%}_normal": value_at_risk(returns, var_level, "normal"),
            f"var_{var_level:.0%}_cf": value_at_risk(returns, var_level, "cornish_fisher"),
            f"es_{var_level:.0%}_hist": expected_shortfall(returns, var_level, "historical"),
            "max_drawdown": float(dd.min()),
            "current_drawdown": float(dd.iloc[-1]),
            "ulcer_index": ulcer_index(returns),
            "n_drawdown_episodes": float(len(episodes)),
            "avg_gross_exposure": float(exposures["gross"].mean()),
            "avg_net_exposure": float(exposures["net"].mean()),
            "max_position_weight": float(weights.fillna(0.0).abs().max().max()),
        }
        if benchmark is not None:
            summary["beta"] = beta(returns, benchmark)

        sector_frame = sector_exposures(weights, sectors) if sectors is not None else None

        breaches: list[LimitBreach] = []
        if limits:
            state = PortfolioState(
                weights=weights.iloc[-1].fillna(0.0),
                returns=returns,
                timestamp=returns.index[-1],
            )
            breaches = RiskMonitor(limits).evaluate(state)
        summary["n_limit_breaches"] = float(len(breaches))

        return cls(
            summary=pd.Series(summary, name="risk"),
            exposures=exposures,
            sectors=sector_frame,
            drawdowns=episodes,
            breaches=breaches,
            var_level=var_level,
        )

    def to_text(self) -> str:
        """Multi-line text rendering of the report."""
        lines = ["Risk report", "=" * 11, ""]
        for key, value in self.summary.items():
            lines.append(f"{key:<24} {value:>10.4f}")
        if not self.drawdowns.empty:
            worst = self.drawdowns.sort_values("depth").head(3)
            lines += ["", "Worst drawdowns:"]
            for _, row in worst.iterrows():
                recovery = (
                    row["recovery"].date() if pd.notna(row["recovery"]) else "ongoing"
                )
                lines.append(
                    f"  {row['depth']:>7.1%}  {row['start'].date()} -> "
                    f"{row['trough'].date()} -> {recovery} ({row['length']} bars)"
                )
        if self.breaches:
            lines += ["", "Limit breaches:"]
            lines += [f"  {b}" for b in self.breaches]
        if self.sectors is not None:
            lines += ["", "Final sector exposures:"]
            final = self.sectors.iloc[-1]
            lines += [f"  {sector:<12} {value:>8.3f}" for sector, value in final.items()]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - delegates to to_text
        return self.to_text()
