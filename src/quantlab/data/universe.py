"""Point-in-time universe construction.

Membership at date t is decided with data through t only (trailing
dollar-volume ranks as of the most recent rebalance bar), so membership
timing is lookahead-free. Membership COMPOSITION still inherits
survivorship bias from the symbol list itself: a list of today's large
caps contains only survivors, and free price sources do not carry
delisted names. Every backtest on such a list overstates returns; say so
when reporting results (see docs/research_roadmap.md, Appendix A).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = [
    "US_LARGE_CAP_150",
    "US_LARGE_CAP_SECTORS",
    "apply_universe",
    "rolling_universe",
]


def rolling_universe(
    dataset: MarketDataset,
    *,
    top_n: int = 100,
    window: int = 63,
    min_price: float | None = 5.0,
    rebalance_every: int = 21,
) -> WideFrame:
    """Boolean membership mask (dates x symbols).

    Every ``rebalance_every`` bars, starting at the first bar with a full
    ``window`` of history (bar index ``window - 1``), rank symbols by
    trailing ``window``-bar mean dollar volume computed through that bar
    (inclusive); the top ``top_n`` with trailing median close >=
    ``min_price`` are members until the next rebalance. Rows before the
    first rebalance are all False. ``top_n`` larger than the number of
    eligible symbols keeps them all (no error). Symbols with insufficient
    data at a rebalance (NaN trailing volume) are not members. Exact
    dollar-volume ties are resolved by pandas ``nlargest`` order
    (first-come in column order), which is acceptable for a liquidity
    screen.

    Because every rank uses data through the rebalance bar only,
    membership TIMING is point-in-time by construction. Membership
    COMPOSITION is only as honest as the symbol list itself — see the
    module docstring for the survivorship caveat.
    """
    if top_n < 1:
        raise DataError("top_n must be >= 1")
    if window < 1:
        raise DataError("window must be >= 1")
    if rebalance_every < 1:
        raise DataError("rebalance_every must be >= 1")

    # Trailing stats through each bar (inclusive). The default
    # min_periods == window means a symbol without a full window of data
    # has NaN trailing volume and is excluded at that rebalance.
    adv = dataset.dollar_volume().rolling(window).mean()
    med_price = dataset.close.rolling(window).median()

    mask = np.zeros((dataset.n_periods, dataset.n_symbols), dtype=bool)
    col_of = {sym: j for j, sym in enumerate(dataset.symbols)}
    for pos in range(window - 1, dataset.n_periods, rebalance_every):
        row = adv.iloc[pos].dropna()
        if min_price is not None:
            floor = med_price.iloc[pos].reindex(row.index)
            row = row[floor >= min_price]  # NaN median compares False -> excluded
        members = row.nlargest(top_n).index
        end = min(pos + rebalance_every, dataset.n_periods)
        for sym in members:
            mask[pos:end, col_of[sym]] = True
    return pd.DataFrame(mask, index=dataset.index, columns=dataset.symbols)


def apply_universe(frame: WideFrame, universe: WideFrame) -> WideFrame:
    """NaN out ``frame`` cells outside the universe.

    ``universe`` is reindexed to ``frame``'s index/columns with missing
    entries treated as False, so a signal for a symbol the universe never
    admitted is fully masked. Values inside the universe pass through
    unchanged (this masks, it does not renormalize).
    """
    aligned = universe.reindex(
        index=frame.index, columns=frame.columns, fill_value=False
    ).astype(bool)
    return frame.where(aligned)


# --------------------------------------------------------------------------
# Static seed universe.
#
# SURVIVORSHIP WARNING: this list reflects large, liquid US names drawn from
# well-known S&P 500 constituents as of mid-2026 (the authoring date). It
# contains SURVIVORS ONLY — everything delisted, acquired, or bankrupted
# before then is absent, and free price sources cannot backfill those names.
# Historical backtests on this list systematically overstate returns; state
# that caveat on every result produced from it (docs/research_roadmap.md,
# Appendix A). The correctness of the universe code never depends on the
# specific names here.
# --------------------------------------------------------------------------
US_LARGE_CAP_SECTORS: dict[str, str] = {
    # Information Technology (30)
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "ACN": "Information Technology",
    "CSCO": "Information Technology",
    "IBM": "Information Technology",
    "INTC": "Information Technology",
    "INTU": "Information Technology",
    "NOW": "Information Technology",
    "QCOM": "Information Technology",
    "TXN": "Information Technology",
    "AMAT": "Information Technology",
    "MU": "Information Technology",
    "LRCX": "Information Technology",
    "KLAC": "Information Technology",
    "ADI": "Information Technology",
    "SNPS": "Information Technology",
    "CDNS": "Information Technology",
    "PANW": "Information Technology",
    "ANET": "Information Technology",
    "MSI": "Information Technology",
    "ADSK": "Information Technology",
    "FTNT": "Information Technology",
    "NXPI": "Information Technology",
    "MCHP": "Information Technology",
    # Communication Services (10)
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "TMUS": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "EA": "Communication Services",
    "WBD": "Communication Services",
    # Consumer Discretionary (15)
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    # Consumer Staples (12)
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "CL": "Consumer Staples",
    "KMB": "Consumer Staples",
    "GIS": "Consumer Staples",
    "KHC": "Consumer Staples",
    # Health Care (20)
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "PFE": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "DHR": "Health Care",
    "BMY": "Health Care",
    "AMGN": "Health Care",
    "GILD": "Health Care",
    "CVS": "Health Care",
    "MDT": "Health Care",
    "ISRG": "Health Care",
    "VRTX": "Health Care",
    "REGN": "Health Care",
    "ZTS": "Health Care",
    "SYK": "Health Care",
    "BSX": "Health Care",
    # Financials (22) — V/MA/PYPL sit here per the 2023 GICS reshuffle
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",
    "SPGI": "Financials",
    "MCO": "Financials",
    "PYPL": "Financials",
    "PNC": "Financials",
    "USB": "Financials",
    "COF": "Financials",
    "CB": "Financials",
    "MMC": "Financials",
    "ICE": "Financials",
    "CME": "Financials",
    # Industrials (19)
    "CAT": "Industrials",
    "DE": "Industrials",
    "UNP": "Industrials",
    "UPS": "Industrials",
    "HON": "Industrials",
    "GE": "Industrials",
    "BA": "Industrials",
    "RTX": "Industrials",
    "LMT": "Industrials",
    "NOC": "Industrials",
    "GD": "Industrials",
    "MMM": "Industrials",
    "ETN": "Industrials",
    "EMR": "Industrials",
    "ITW": "Industrials",
    "CSX": "Industrials",
    "NSC": "Industrials",
    "FDX": "Industrials",
    "WM": "Industrials",
    # Energy (9)
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "OXY": "Energy",
    # Utilities (6)
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "SRE": "Utilities",
    # Real Estate (4)
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    # Materials (3)
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
}

US_LARGE_CAP_150: tuple[str, ...] = tuple(US_LARGE_CAP_SECTORS)
