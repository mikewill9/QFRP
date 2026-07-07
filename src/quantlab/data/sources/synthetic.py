"""Deterministic synthetic market data source.

Purpose
-------
Every subsystem in quantlab (tests, examples, notebooks) must run offline and
reproducibly. This source simulates a cross-section of equities driven by a
linear factor model, which gives the data the structure real research code
cares about:

- a common **market factor** (assets co-move; beta dispersion is real);
- **sector factors** (block correlation, so PCA / HRP / sector
  neutralization behave meaningfully);
- **idiosyncratic noise** with per-asset volatility levels;
- internally consistent OHLC bars (``low <= min(open, close)`` and
  ``high >= max(open, close)`` by construction, all prices positive);
- volume with AR(1) persistence and a volume/|return| link, so liquidity
  factors such as Amihud have signal.

Determinism
-----------
Daily shocks are drawn from a generator seeded by ``(seed, date ordinal)``,
and per-asset parameters from ``(seed, asset index)``. Consequently, for a
fixed ``(seed, n_assets, n_sectors, inception)``, any two fetches agree
exactly on their overlapping dates, regardless of the requested window or
symbol subset.

Model
-----
Daily log-return of asset ``i`` in sector ``k(i)``::

    r_it = mu_i / A  +  beta_i * f_t  +  g_i * s_{k(i), t}  +  e_it

with ``A = 252``, ``f_t ~ N(mu_m/A, (sig_m/sqrt(A))^2)``, sector shocks
``s_kt ~ N(0, (sig_s/sqrt(A))^2)`` and ``e_it ~ N(0, sig_i^2)``. Prices are
``p0_i * exp(cumsum(r))`` from ``inception``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.data.sources.base import DataSource, normalize_date, validate_request
from quantlab.exceptions import DataError
from quantlab.types import Frequency

__all__ = ["SyntheticSource", "make_synthetic_dataset"]

_ANNUALIZATION = 252.0
_ASSET_STREAM = 1_000
_DATE_STREAM = 2_000_000_000


@dataclass(frozen=True, slots=True)
class _AssetParams:
    """Per-asset structural parameters (drawn once from the seed)."""

    beta: np.ndarray  # market loading
    sector_loading: np.ndarray  # loading on own-sector factor
    sector_id: np.ndarray  # int sector index per asset
    mu_annual: np.ndarray  # annual log drift
    idio_vol_daily: np.ndarray  # daily idiosyncratic vol
    p0: np.ndarray  # initial price
    adv_shares: np.ndarray  # typical daily share volume
    total_vol_daily: np.ndarray  # total daily vol (for bar-range scaling)


class SyntheticSource(DataSource):
    """Factor-model synthetic equity universe.

    Parameters
    ----------
    n_assets:
        Universe size; symbols are ``SYN000 .. SYN{n_assets-1}``.
    n_sectors:
        Number of sector factors; assets are assigned round-robin.
    seed:
        Master seed; fully determines every path.
    inception:
        First simulated date. ``fetch`` requests must start on or after it.
    market_mu, market_vol:
        Annualized drift and vol of the market factor.
    sector_vol:
        Annualized vol of each sector factor.
    """

    name = "synthetic"

    def __init__(
        self,
        n_assets: int = 50,
        n_sectors: int = 5,
        seed: int = 7,
        inception: pd.Timestamp | str = "2015-01-02",
        market_mu: float = 0.07,
        market_vol: float = 0.16,
        sector_vol: float = 0.08,
    ) -> None:
        if n_assets < 1:
            raise DataError("n_assets must be >= 1")
        if not 1 <= n_sectors <= n_assets:
            raise DataError("n_sectors must satisfy 1 <= n_sectors <= n_assets")
        self.n_assets = int(n_assets)
        self.n_sectors = int(n_sectors)
        self.seed = int(seed)
        self.inception = normalize_date(inception)
        self.market_mu = float(market_mu)
        self.market_vol = float(market_vol)
        self.sector_vol = float(sector_vol)
        self._params = self._draw_asset_params()

    # ------------------------------------------------------------- universe
    @property
    def symbols(self) -> list[str]:
        """All symbols in the universe."""
        return [f"SYN{i:03d}" for i in range(self.n_assets)]

    def sector_map(self) -> dict[str, str]:
        """Mapping symbol -> sector label (``'S0'``, ``'S1'``, ...)."""
        return {
            sym: f"S{int(k)}"
            for sym, k in zip(self.symbols, self._params.sector_id, strict=True)
        }

    def beta_map(self) -> dict[str, float]:
        """True market betas by symbol (useful for validating risk models)."""
        return {
            sym: float(b) for sym, b in zip(self.symbols, self._params.beta, strict=True)
        }

    # ---------------------------------------------------------------- fetch
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        if start_ts < self.inception:
            raise DataError(
                f"start {start_ts.date()} precedes inception {self.inception.date()}"
            )
        idx_map = {sym: i for i, sym in enumerate(self.symbols)}
        unknown = [s for s in symbols_list if s not in idx_map]
        if unknown:
            raise DataError(f"Unknown synthetic symbols: {unknown[:10]}")
        cols = np.array([idx_map[s] for s in symbols_list], dtype=int)

        dates = pd.bdate_range(self.inception, end_ts)
        if len(dates) == 0:
            raise DataError("No business days in requested range")
        open_a, high_a, low_a, close_a, vol_a = self._simulate(dates)

        def frame(arr: np.ndarray) -> pd.DataFrame:
            df = pd.DataFrame(arr[:, cols], index=dates, columns=symbols_list)
            return df.loc[start_ts:end_ts]

        sectors = self.sector_map()
        betas = self.beta_map()
        ds = MarketDataset(
            open=frame(open_a),
            high=frame(high_a),
            low=frame(low_a),
            close=frame(close_a),
            volume=frame(vol_a),
            frequency=Frequency.DAILY,
            metadata={
                "source": self.name,
                "seed": self.seed,
                "inception": str(self.inception.date()),
                "sectors": {s: sectors[s] for s in symbols_list},
                "betas": {s: betas[s] for s in symbols_list},
            },
        )
        if frequency is not Frequency.DAILY:
            from quantlab.data.preprocessing import resample_dataset

            ds = resample_dataset(ds, frequency)
        return ds

    # ----------------------------------------------------------- simulation
    def _draw_asset_params(self) -> _AssetParams:
        n = self.n_assets
        rng = np.random.default_rng([self.seed, _ASSET_STREAM])
        beta = rng.normal(1.0, 0.25, size=n)
        sector_loading = rng.uniform(0.5, 1.5, size=n)
        sector_id = np.arange(n) % self.n_sectors
        mu_annual = rng.normal(0.06, 0.08, size=n)
        idio_vol_daily = rng.uniform(0.10, 0.40, size=n) / np.sqrt(_ANNUALIZATION)
        p0 = rng.uniform(15.0, 250.0, size=n)
        adv_shares = np.exp(rng.normal(13.0, 0.9, size=n))  # ~4e5 median shares
        mvd = self.market_vol / np.sqrt(_ANNUALIZATION)
        svd = self.sector_vol / np.sqrt(_ANNUALIZATION)
        total_vol_daily = np.sqrt(
            (beta * mvd) ** 2 + (sector_loading * svd) ** 2 + idio_vol_daily**2
        )
        return _AssetParams(
            beta=beta,
            sector_loading=sector_loading,
            sector_id=sector_id,
            mu_annual=mu_annual,
            idio_vol_daily=idio_vol_daily,
            p0=p0,
            adv_shares=adv_shares,
            total_vol_daily=total_vol_daily,
        )

    def _daily_shocks(
        self, date: pd.Timestamp
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw all shocks for one date in a fixed, documented order."""
        rng = np.random.default_rng([self.seed, _DATE_STREAM + date.toordinal()])
        z_market = float(rng.standard_normal())
        z_sector = rng.standard_normal(self.n_sectors)
        z_idio = rng.standard_normal(self.n_assets)
        z_gap = rng.standard_normal(self.n_assets)
        z_up = np.abs(rng.standard_normal(self.n_assets))
        z_down = np.abs(rng.standard_normal(self.n_assets))
        z_volume = rng.standard_normal(self.n_assets)
        return z_market, z_sector, z_idio, z_gap, z_up, z_down, z_volume

    def _simulate(
        self, dates: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self._params
        n_t, n = len(dates), self.n_assets
        mvd = self.market_vol / np.sqrt(_ANNUALIZATION)
        svd = self.sector_vol / np.sqrt(_ANNUALIZATION)
        mu_daily = p.mu_annual / _ANNUALIZATION
        market_mu_daily = self.market_mu / _ANNUALIZATION

        log_ret = np.empty((n_t, n))
        gap = np.empty((n_t, n))
        up = np.empty((n_t, n))
        down = np.empty((n_t, n))
        vol_state = np.zeros(n)
        volume = np.empty((n_t, n))
        phi, state_sd = 0.85, 0.55  # AR(1) persistence and stationary sd

        for t, date in enumerate(dates):
            z_m, z_s, z_i, z_g, z_u, z_d, z_v = self._daily_shocks(date)
            f_t = market_mu_daily + mvd * z_m
            s_t = svd * z_s
            log_ret[t] = (
                mu_daily
                + p.beta * f_t
                + p.sector_loading * s_t[p.sector_id]
                + p.idio_vol_daily * z_i
            )
            gap[t] = 0.25 * p.total_vol_daily * z_g
            up[t] = np.minimum(0.6 * p.total_vol_daily * z_u, 3.0 * p.total_vol_daily)
            down[t] = np.minimum(0.6 * p.total_vol_daily * z_d, 3.0 * p.total_vol_daily)
            vol_state = phi * vol_state + state_sd * np.sqrt(1.0 - phi**2) * z_v
            volume[t] = vol_state  # temporarily store state; scaled below

        close = p.p0 * np.exp(np.cumsum(log_ret, axis=0))
        prev_close = np.vstack([p.p0[None, :], close[:-1]])
        open_ = prev_close * np.exp(gap)
        high = np.maximum(open_, close) * np.exp(up)
        low = np.minimum(open_, close) * np.exp(-down)

        abs_ret_scaled = np.abs(log_ret) / p.total_vol_daily
        volume = p.adv_shares * np.exp(volume + 0.35 * abs_ret_scaled - 0.5 * state_sd**2)
        volume = np.round(volume)

        return open_, high, low, close, volume


def make_synthetic_dataset(
    n_assets: int = 20,
    start: pd.Timestamp | str = "2020-01-01",
    end: pd.Timestamp | str = "2023-12-29",
    *,
    n_sectors: int = 4,
    seed: int = 7,
    frequency: Frequency = Frequency.DAILY,
) -> MarketDataset:
    """One-call synthetic dataset for tests, examples, and notebooks.

    Inception is pinned to ``start``, so the whole panel is generated for the
    requested window only (fast). Determinism follows from
    :class:`SyntheticSource`.
    """
    source = SyntheticSource(
        n_assets=n_assets, n_sectors=n_sectors, seed=seed, inception=start
    )
    return source.fetch(source.symbols, start=start, end=end, frequency=frequency)
