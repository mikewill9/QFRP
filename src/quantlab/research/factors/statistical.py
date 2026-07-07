"""Statistical factors: cross-sectional structure extracted from returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.types import WideFrame

__all__ = ["MarketBeta", "PCAResidualReversal"]


class MarketBeta(Factor):
    """Rolling beta to the equal-weight market of the dataset itself.

    With the default ``invert=True`` the score is ``-beta``, expressing the
    low-beta anomaly (defensive names rank high). ``invert=False`` returns
    raw beta, useful as a hedging input or neutralization group.
    """

    name = "market_beta"

    def __init__(self, window: int = 63, invert: bool = True) -> None:
        if window < 10:
            raise DataError("window must be >= 10")
        self.window = window
        self.invert = invert
        self.min_history = window + 1

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        rets = dataset.returns(method="log")
        market = rets.mean(axis=1)
        cov = rets.rolling(self.window).cov(market)
        var = market.rolling(self.window).var()
        beta = cov.div(var, axis=0)
        return -beta if self.invert else beta


class PCAResidualReversal(Factor):
    """Reversal in returns orthogonal to the top principal components.

    At each refit date (every ``stride`` bars) a PCA is fitted on the
    trailing ``window`` of standardized returns; between refits the loadings
    are held fixed. Each day's standardized cross-sectional return vector is
    projected off the top ``n_components`` components, leaving an
    idiosyncratic residual. The score at ``t`` is the *negative* sum of the
    last ``resid_lookback`` residuals: names that fell for reasons the
    common factors do not explain score high, the classic statistical
    arbitrage reversion setup.

    Deterministic given the data (SVD-based, no randomized solver).
    """

    name = "pca_residual_reversal"

    def __init__(
        self,
        window: int = 126,
        n_components: int = 3,
        stride: int = 21,
        resid_lookback: int = 5,
    ) -> None:
        if window < 40:
            raise DataError("window must be >= 40")
        if not 1 <= n_components < window:
            raise DataError("n_components must be in [1, window)")
        if stride < 1:
            raise DataError("stride must be >= 1")
        if resid_lookback < 1:
            raise DataError("resid_lookback must be >= 1")
        self.window = window
        self.n_components = n_components
        self.stride = stride
        self.resid_lookback = resid_lookback
        self.min_history = window + resid_lookback

    def compute(self, dataset: MarketDataset) -> WideFrame:
        self._require_history(dataset)
        rets = dataset.returns(method="log")
        r = rets.to_numpy()
        n_t, n_sym = r.shape
        k = min(self.n_components, n_sym - 1)
        residuals = np.full((n_t, n_sym), np.nan)
        loadings: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        first = self.window  # first row with a full trailing window behind it
        for t in range(first, n_t):
            if loadings is None or (t - first) % self.stride == 0:
                x = r[t - self.window + 1 : t + 1]
                mu = np.nanmean(x, axis=0)
                sd = np.nanstd(x, axis=0, ddof=1)
                sd[~np.isfinite(sd) | (sd <= 0)] = np.nan
                z = (x - mu) / sd
                z = np.where(np.isfinite(z), z, 0.0)
                # Economical SVD; right singular vectors are the components.
                _, _, vt = np.linalg.svd(z, full_matrices=False)
                loadings = (mu, sd, vt[:k].T)  # V: (n_sym, k)
            mu, sd, v = loadings
            z_t = (r[t] - mu) / sd
            valid = np.isfinite(z_t)
            z_filled = np.where(valid, z_t, 0.0)
            resid = z_filled - v @ (v.T @ z_filled)
            resid[~valid] = np.nan
            residuals[t] = resid

        lb = self.resid_lookback
        scores = np.full((n_t, n_sym), np.nan)
        for t in range(first + lb - 1, n_t):
            block = residuals[t - lb + 1 : t + 1]
            with np.errstate(invalid="ignore"):
                s = -np.nansum(block, axis=0)
            s[np.isnan(block).all(axis=0)] = np.nan
            scores[t] = s
        return pd.DataFrame(scores, index=rets.index, columns=rets.columns)
