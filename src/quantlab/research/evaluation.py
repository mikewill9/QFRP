"""Signal evaluation.

Alignment convention used throughout: a signal value at date ``t`` is scored
against the *forward* return over ``(t, t+h]`` produced by
:func:`forward_returns`. As long as the signal only uses information through
the close of ``t`` (the Factor contract), this alignment is free of
lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = [
    "SignalEvaluation",
    "evaluate_signal",
    "forward_returns",
    "ic_decay",
    "ic_series",
    "ic_summary",
    "newey_west_tstat",
    "quantile_returns",
    "quantile_spread",
    "signal_turnover",
]


def forward_returns(
    prices: WideFrame, horizon: int = 1, method: str = "log"
) -> WideFrame:
    """Return over ``(t, t+horizon]``, indexed at ``t``.

    The final ``horizon`` rows are NaN by construction; evaluation code must
    keep them NaN rather than dropping the dates, so indexes stay aligned.
    """
    if horizon < 1:
        raise DataError("horizon must be >= 1")
    if method == "log":
        logp = np.log(prices)
        return logp.shift(-horizon) - logp
    if method == "simple":
        return prices.shift(-horizon) / prices - 1.0
    raise DataError(f"Unknown return method '{method}'")


def _rowwise_corr(a: np.ndarray, b: np.ndarray, min_valid: int = 3) -> np.ndarray:
    """Pearson correlation per row with pairwise NaN handling."""
    valid = np.isfinite(a) & np.isfinite(b)
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)
    n = valid.sum(axis=1).astype(float)
    n_safe = np.where(n > 0, n, np.nan)
    ma = a.sum(axis=1) / n_safe
    mb = b.sum(axis=1) / n_safe
    da = np.where(valid, a - ma[:, None], 0.0)
    db = np.where(valid, b - mb[:, None], 0.0)
    cov = (da * db).sum(axis=1)
    denom = np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    corr[(n < min_valid) | ~np.isfinite(corr)] = np.nan
    return corr


def ic_series(
    signal: WideFrame,
    fwd_returns: WideFrame,
    method: str = "spearman",
) -> pd.Series:
    """Per-date information coefficient between signal and forward returns.

    ``spearman`` (default) is the rank IC, robust to outliers and monotone
    transforms of the signal; ``pearson`` is the linear IC.
    """
    if method not in ("spearman", "pearson"):
        raise DataError("method must be 'spearman' or 'pearson'")
    idx = signal.index.intersection(fwd_returns.index)
    cols = signal.columns.intersection(fwd_returns.columns)
    if len(idx) == 0 or len(cols) < 3:
        raise DataError("signal and forward returns share too little overlap")
    s = signal.loc[idx, cols]
    f = fwd_returns.loc[idx, cols]
    if method == "spearman":
        s = s.rank(axis=1)
        f = f.rank(axis=1)
    corr = _rowwise_corr(s.to_numpy(dtype=float), f.to_numpy(dtype=float))
    return pd.Series(corr, index=idx, name=f"ic_{method}")


def newey_west_tstat(series: pd.Series, lags: int = 0) -> float:
    """t-statistic of the series mean with HAC (Newey-West) standard errors.

    ``lags=0`` reduces to the ordinary t-stat. For an IC series built on
    ``h``-bar forward returns, ``lags=h-1`` corrects for the mechanical
    overlap-induced autocorrelation.
    """
    if lags < 0:
        raise DataError("lags must be >= 0")
    x = series.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    mean = x.mean()
    d = x - mean
    gamma0 = float(d @ d) / n
    lrv = gamma0
    for j in range(1, min(lags, n - 1) + 1):
        gamma_j = float(d[j:] @ d[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * gamma_j
    if lrv <= 0:
        return float("nan")
    return mean / np.sqrt(lrv / n)


def ic_summary(ic: pd.Series, lags: int = 0) -> pd.Series:
    """Aggregate an IC series: mean, std, IR, HAC t-stat, hit rate, count."""
    clean = ic.dropna()
    if clean.empty:
        raise DataError("IC series is entirely NaN")
    return pd.Series(
        {
            "ic_mean": clean.mean(),
            "ic_std": clean.std(),
            "ic_ir": clean.mean() / clean.std() if clean.std() > 0 else np.nan,
            "t_stat_nw": newey_west_tstat(clean, lags=lags),
            "pct_positive": (clean > 0).mean(),
            "n_periods": float(clean.size),
        }
    )


def _quantile_buckets(signal_row: pd.Series, n: int) -> pd.Series:
    ranks = signal_row.rank(pct=True)
    return np.ceil(ranks * n).clip(1, n)


def quantile_returns(
    signal: WideFrame,
    fwd_returns: WideFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return per signal quantile, per date.

    Columns are ``1`` (lowest scores) through ``n_quantiles`` (highest).
    A monotonically increasing column mean across quantiles is the classic
    picture of a predictive signal.
    """
    if n_quantiles < 2:
        raise DataError("n_quantiles must be >= 2")
    idx = signal.index.intersection(fwd_returns.index)
    cols = signal.columns.intersection(fwd_returns.columns)
    s = signal.loc[idx, cols]
    f = fwd_returns.loc[idx, cols]
    out = np.full((len(idx), n_quantiles), np.nan)
    s_arr = s.to_numpy(dtype=float)
    f_arr = f.to_numpy(dtype=float)
    for i in range(len(idx)):
        row_s = pd.Series(s_arr[i], index=cols)
        row_f = f_arr[i]
        valid = np.isfinite(row_s.to_numpy()) & np.isfinite(row_f)
        if valid.sum() < n_quantiles:
            continue
        buckets = _quantile_buckets(row_s[valid], n_quantiles).to_numpy()
        rf = row_f[valid]
        for q in range(1, n_quantiles + 1):
            in_q = buckets == q
            if in_q.any():
                out[i, q - 1] = rf[in_q].mean()
    return pd.DataFrame(out, index=idx, columns=range(1, n_quantiles + 1))


def quantile_spread(quantile_rets: pd.DataFrame) -> pd.Series:
    """Top-minus-bottom quantile return per date (long/short proxy)."""
    top = quantile_rets.columns.max()
    bottom = quantile_rets.columns.min()
    return (quantile_rets[top] - quantile_rets[bottom]).rename("quantile_spread")


def signal_turnover(signal: WideFrame) -> pd.Series:
    """Mean absolute day-over-day change in cross-sectional percentile rank.

    0 means a frozen signal; values near 0.33 mean ranks are reshuffled
    almost independently each bar (the expected |difference| of two iid
    uniforms). High turnover signals need strong ICs to survive costs.
    """
    ranks = signal.rank(axis=1, pct=True)
    return ranks.diff().abs().mean(axis=1).rename("rank_turnover")


def ic_decay(
    signal: WideFrame,
    prices: WideFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
    method: str = "spearman",
) -> pd.Series:
    """Mean IC at multiple forward horizons (how fast the edge fades)."""
    if not horizons:
        raise DataError("horizons must be non-empty")
    values = {
        h: ic_series(signal, forward_returns(prices, h), method=method).mean()
        for h in sorted(horizons)
    }
    return pd.Series(values, name="ic_by_horizon")


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """Full evaluation bundle for one signal."""

    name: str
    horizon: int
    ic: pd.Series
    summary: pd.Series
    quantile_rets: pd.DataFrame
    spread: pd.Series
    turnover: pd.Series
    decay: pd.Series
    params: dict = field(default_factory=dict)

    def headline(self) -> pd.Series:
        """One row of the numbers that matter for a quick comparison."""
        return pd.Series(
            {
                "ic_mean": self.summary["ic_mean"],
                "ic_ir": self.summary["ic_ir"],
                "t_stat_nw": self.summary["t_stat_nw"],
                "spread_mean": self.spread.mean(),
                "rank_turnover": self.turnover.mean(),
            },
            name=self.name,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SignalEvaluation({self.name!r}, h={self.horizon}, "
            f"ic_mean={self.summary['ic_mean']:.4f}, "
            f"t_nw={self.summary['t_stat_nw']:.2f})"
        )


def evaluate_signal(
    signal: WideFrame,
    prices: WideFrame,
    *,
    name: str = "signal",
    horizon: int = 1,
    n_quantiles: int = 5,
    method: str = "spearman",
    decay_horizons: tuple[int, ...] = (1, 5, 10, 21),
    params: dict | None = None,
) -> SignalEvaluation:
    """Run the standard evaluation battery on a signal frame.

    Parameters mirror the individual functions; the HAC lag for the IC
    t-statistic is set to ``horizon - 1`` automatically.
    """
    fwd = forward_returns(prices, horizon)
    ic = ic_series(signal, fwd, method=method)
    q = quantile_returns(signal, fwd, n_quantiles)
    return SignalEvaluation(
        name=name,
        horizon=horizon,
        ic=ic,
        summary=ic_summary(ic, lags=horizon - 1),
        quantile_rets=q,
        spread=quantile_spread(q),
        turnover=signal_turnover(signal),
        decay=ic_decay(signal, prices, decay_horizons, method=method),
        params=dict(params or {}),
    )
