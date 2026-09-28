import numpy as np
import pandas as pd


def permute_within_date(zscores: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Shuffle one date's z-scores across stocks, leaving NaN entries in place."""
    out = zscores.copy()
    mask = zscores.notna()
    values = zscores[mask].to_numpy()
    if len(values)<=1:
        return out
    out[mask] = rng.permutation(values)
    return out


def permutation_pvalue(real_sharpe: float, null_sharpes: np.ndarray) -> tuple[float, int]:
    """One-sided p-value: share of null Sharpes at least as large as the real one.

    Uses (count+1)/(n+1) so the p-value is never exactly zero.
    NaN null Sharpes are dropped, and the number dropped is returned.
    """
    if np.isnan(real_sharpe):
        raise ValueError("real_sharpe is NaN")
    null_sharpes = np.asarray(null_sharpes, dtype=float)
    valid = null_sharpes[~np.isnan(null_sharpes)]
    n_dropped = len(null_sharpes) - len(valid)
    if len(valid) == 0:
        raise ValueError("no valid null Sharpes to compare against")
    count = (valid >= real_sharpe).sum()
    p_value = (count + 1) / (len(valid) + 1)
    return p_value, n_dropped


def bonferroni_adjust(p: float, k: int) -> float:
    """Adjust a p-value for k tests viewed in a session.

    Returns min(1.0, p * k). Raises ValueError on invalid inputs.
    """
    if np.isnan(p) or p < 0 or p > 1:
        raise ValueError("p must be a real probability between 0 and 1")

    if k<1:
        raise ValueError("k must be a positive integer")

    return min(1.0, p * k)
