"""Fundamental factors: hand-computed kernels, publication-straddle lookahead
tests, coverage semantics, and the estimate-revision interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import (
    FundamentalDataset,
    MarketDataset,
    SyntheticFundamentalSource,
    make_synthetic_dataset,
)
from quantlab.exceptions import DataError
from quantlab.research import (
    Accruals,
    BookToMarket,
    EarningsGrowth,
    EarningsYield,
    EstimateRevision,
    FCFYield,
    GrossMarginStability,
    ReturnOnEquity,
    RevenueGrowth,
)

# Five calendar quarters; a flat 45-day lag publishes them on 2019-05-15,
# 2019-08-14, 2019-11-14, 2020-02-14 and 2020-05-15 (all business days).
QP = pd.to_datetime(
    ["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"]
)
PUBS = QP + pd.Timedelta(days=45)
TTM_PUB = pd.Timestamp("2020-02-14")  # first date a full 2019 TTM is public
Q1_PUB = pd.Timestamp("2020-05-15")  # 2020Q1 statement becomes public

_DEFAULTS = {
    "revenue": 100.0,
    "gross_profit": 40.0,
    "net_income": 10.0,
    "total_equity": 400.0,
    "total_liabilities": 600.0,
    "shares_outstanding": 100.0,
    "operating_cash_flow": 10.0,
    "capital_expenditure": -10.0,
}


def build_fundamentals(
    symbol: str = "A",
    periods: pd.DatetimeIndex = QP,
    lag_days: int = 45,
    **overrides: float | list[float],
) -> FundamentalDataset:
    """Internally consistent hand-built statements with scalar/list overrides."""
    rows = []
    for i, pe in enumerate(periods):
        row: dict = {
            "symbol": symbol,
            "period_end": pe,
            "publication_date": pe + pd.Timedelta(days=lag_days),
        }
        for item, default in _DEFAULTS.items():
            value = overrides.get(item, default)
            row[item] = float(value[i]) if isinstance(value, (list, tuple)) else float(value)
        row["total_assets"] = row["total_equity"] + row["total_liabilities"]
        rows.append(row)
    return FundamentalDataset(pd.DataFrame(rows))


def price_dataset(
    symbols: list[str],
    start: str = "2019-01-01",
    end: str = "2020-06-30",
    price: float = 8.0,
) -> MarketDataset:
    """Constant-price market so market cap is exact in hand computations."""
    idx = pd.bdate_range(start, end)
    close = pd.DataFrame(price, index=idx, columns=symbols)
    return MarketDataset(
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=close * 0 + 1e6,
    )


# ------------------------------------------------------------ value kernels
def test_earnings_yield_hand_computed_with_publication_timing():
    funda = build_fundamentals(net_income=[10.0, 20.0, 30.0, 40.0, 50.0])
    ds = price_dataset(["A"])  # market cap = 8 * 100 = 800 every day
    scores = EarningsYield(funda).compute(ds)["A"]
    # No complete TTM is public before the 2019Q4 statement lands.
    assert scores.loc[: TTM_PUB - pd.Timedelta(days=1)].isna().all()
    # 2019 TTM = 10+20+30+40 = 100 -> 100 / 800.
    assert scores.loc[TTM_PUB] == pytest.approx(100.0 / 800.0)
    assert scores.loc[Q1_PUB - pd.Timedelta(days=1)] == pytest.approx(100.0 / 800.0)
    # On 2020Q1 publication the window rolls to 20+30+40+50 = 140.
    assert scores.loc[Q1_PUB] == pytest.approx(140.0 / 800.0)
    assert scores.iloc[-1] == pytest.approx(140.0 / 800.0)


def test_factor_ignores_statement_until_its_publication_date():
    """The lookahead pin: a statement whose publication lag straddles the
    evaluation date must be invisible until the publication date, so the
    factor at t changes only when information published on or before t
    changes."""
    base_funda = build_fundamentals(net_income=[10.0, 20.0, 30.0, 40.0, 50.0])
    spiked_funda = build_fundamentals(net_income=[10.0, 20.0, 30.0, 40.0, 500.0])
    ds = price_dataset(["A"])
    base = EarningsYield(base_funda).compute(ds)
    spiked = EarningsYield(spiked_funda).compute(ds)
    cutoff = Q1_PUB - pd.Timedelta(days=1)
    # 2020Q1 ended 2020-03-31 but is published 2020-05-15: every score
    # through 2020-05-14 must be bit-identical despite the 10x earnings move.
    pd.testing.assert_frame_equal(base.loc[:cutoff], spiked.loc[:cutoff])
    assert spiked["A"].loc[Q1_PUB] == pytest.approx(590.0 / 800.0)
    assert base["A"].loc[Q1_PUB] == pytest.approx(140.0 / 800.0)


def test_book_to_market_hand_computed_and_negative_equity_scores_low():
    funda = build_fundamentals(
        symbol="A", total_equity=400.0
    ).statements
    neg = build_fundamentals(symbol="B", total_equity=-100.0).statements
    both = FundamentalDataset(pd.concat([funda, neg], ignore_index=True))
    ds = price_dataset(["A", "B"])
    scores = BookToMarket(both).compute(ds)
    first_pub = PUBS[0]
    assert scores["A"].loc[: first_pub - pd.Timedelta(days=1)].isna().all()
    assert scores["A"].loc[first_pub] == pytest.approx(400.0 / 800.0)
    # Negative book equity is a *low* score under the value hypothesis,
    # deliberately not NaN.
    assert scores["B"].loc[first_pub] == pytest.approx(-100.0 / 800.0)


def test_fcf_yield_hand_computed():
    funda = build_fundamentals(
        operating_cash_flow=30.0, capital_expenditure=-10.0
    )
    ds = price_dataset(["A"])
    scores = FCFYield(funda).compute(ds)["A"]
    # TTM FCF = 4 * (30 - 10) = 80 over market cap 800.
    assert scores.loc[: TTM_PUB - pd.Timedelta(days=1)].isna().all()
    assert scores.loc[TTM_PUB] == pytest.approx(80.0 / 800.0)


# ---------------------------------------------------------- quality kernels
def test_roe_hand_computed_and_negative_equity_is_nan():
    ok = build_fundamentals(symbol="A", net_income=25.0, total_equity=400.0).statements
    broken = build_fundamentals(symbol="B", net_income=25.0, total_equity=-50.0).statements
    both = FundamentalDataset(pd.concat([ok, broken], ignore_index=True))
    ds = price_dataset(["A", "B"])
    scores = ReturnOnEquity(both).compute(ds)
    # TTM earnings 100 over book equity 400.
    assert scores["A"].loc[TTM_PUB] == pytest.approx(100.0 / 400.0)
    # Earnings on negative equity is a nonsense ratio -> undefined, not a number.
    assert scores["B"].loc[TTM_PUB:].isna().all()


def test_gross_margin_stability_hand_computed():
    wobbly = build_fundamentals(
        symbol="A", revenue=100.0, gross_profit=[40.0, 60.0, 40.0, 60.0, 40.0]
    ).statements
    steady = build_fundamentals(symbol="B", revenue=100.0, gross_profit=50.0).statements
    both = FundamentalDataset(pd.concat([wobbly, steady], ignore_index=True))
    ds = price_dataset(["A", "B"])
    scores = GrossMarginStability(both, quarters=4).compute(ds)
    expected_std = np.std([0.4, 0.6, 0.4, 0.6], ddof=1)
    assert scores["A"].loc[TTM_PUB] == pytest.approx(-expected_std)
    assert scores["B"].loc[TTM_PUB] == pytest.approx(0.0)
    assert scores["B"].loc[TTM_PUB] > scores["A"].loc[TTM_PUB]  # stable ranks higher
    with pytest.raises(DataError, match="quarters"):
        GrossMarginStability(both, quarters=2)


def test_accruals_hand_computed_low_accruals_rank_high():
    aggressive = build_fundamentals(
        symbol="A", net_income=40.0, operating_cash_flow=30.0
    ).statements
    cash_backed = build_fundamentals(
        symbol="B", net_income=40.0, operating_cash_flow=45.0
    ).statements
    both = FundamentalDataset(pd.concat([aggressive, cash_backed], ignore_index=True))
    ds = price_dataset(["A", "B"])
    scores = Accruals(both).compute(ds)
    # A: TTM accruals = 4 * (40 - 30) = 40 on assets 1000 -> score -0.04.
    assert scores["A"].loc[TTM_PUB] == pytest.approx(-40.0 / 1000.0)
    # B books less income than cash: negative accruals, positive score.
    assert scores["B"].loc[TTM_PUB] == pytest.approx(20.0 / 1000.0)
    assert scores["B"].loc[TTM_PUB] > scores["A"].loc[TTM_PUB]


# ----------------------------------------------------------- growth kernels
EIGHT_Q = pd.date_range("2018-03-31", periods=8, freq="QE")


def test_revenue_growth_quarterly_and_ttm_modes_hand_computed():
    revenue = [100.0] * 4 + [110.0] * 4
    funda = build_fundamentals(periods=EIGHT_Q, revenue=revenue)
    ds = price_dataset(["A"], start="2018-01-01", end="2020-06-30")
    q_pub = EIGHT_Q[4] + pd.Timedelta(days=45)  # 2019Q1 published
    ttm_pub = EIGHT_Q[7] + pd.Timedelta(days=45)  # 2019Q4 published

    quarterly = RevenueGrowth(funda, ttm=False).compute(ds)["A"]
    assert quarterly.loc[: q_pub - pd.Timedelta(days=1)].isna().all()
    assert quarterly.loc[q_pub] == pytest.approx(0.10)

    smooth = RevenueGrowth(funda, ttm=True).compute(ds)["A"]
    # TTM growth needs all eight quarters: 440 vs 400 = +10%, one year later.
    assert smooth.loc[: ttm_pub - pd.Timedelta(days=1)].isna().all()
    assert smooth.loc[ttm_pub] == pytest.approx(440.0 / 400.0 - 1.0)


def test_earnings_growth_recovery_from_loss_is_positive():
    net_income = [-10.0] * 4 + [10.0] * 4
    funda = build_fundamentals(periods=EIGHT_Q, net_income=net_income)
    ds = price_dataset(["A"], start="2018-01-01", end="2020-06-30")
    scores = EarningsGrowth(funda, ttm=False).compute(ds)["A"]
    pub = EIGHT_Q[4] + pd.Timedelta(days=45)
    # (10 - (-10)) / |-10| = +2: recovering from a loss scores as improvement.
    assert scores.loc[pub] == pytest.approx(2.0)


# ------------------------------------------------------ structural behavior
def test_scores_are_step_functions_between_publications():
    funda = build_fundamentals(total_equity=[400.0, 410.0, 420.0, 430.0, 440.0])
    ds = price_dataset(["A"])
    series = BookToMarket(funda).compute(ds)["A"]
    changes = series.diff().fillna(0.0).ne(0.0)
    # With constant prices the score may move only on publication dates.
    assert set(series.index[changes]) <= set(PUBS)
    assert changes.sum() == 4  # equity changed at every publication after the first


def test_uncovered_symbol_is_nan_and_zero_overlap_raises():
    funda = build_fundamentals(symbol="A")
    ds = price_dataset(["A", "ZZZ"])
    scores = EarningsYield(funda).compute(ds)
    assert list(scores.columns) == ["A", "ZZZ"]
    assert scores.index.equals(ds.index)
    assert scores["ZZZ"].isna().all()
    disjoint = price_dataset(["X", "Y"])
    with pytest.raises(DataError, match="no overlap"):
        EarningsYield(funda).compute(disjoint)
    with pytest.raises(DataError, match="FundamentalDataset"):
        EarningsYield("not-a-dataset")  # type: ignore[arg-type]


def test_params_echo_config_but_not_the_data_object():
    funda = build_fundamentals()
    assert EarningsYield(funda).params == {}
    assert GrossMarginStability(funda, quarters=6).params == {"quarters": 6}
    assert RevenueGrowth(funda, ttm=False).params == {"ttm": False}
    assert EarningsYield(funda).fundamentals is funda


# --------------------------------------------------------- estimate revision
def test_estimate_revision_requires_point_in_time_estimates():
    ds = price_dataset(["A"])
    with pytest.raises(DataError, match="point-in-time"):
        EstimateRevision().compute(ds)
    wrong_item = build_fundamentals()  # statements, not estimates
    with pytest.raises(DataError, match="point-in-time"):
        EstimateRevision(estimates=wrong_item).compute(ds)


def test_estimate_revision_computes_on_a_pit_estimates_panel():
    def snapshots(symbol: str, values: dict[str, float]) -> list[dict]:
        return [
            {
                "symbol": symbol,
                "period_end": pd.Timestamp(date),
                "publication_date": pd.Timestamp(date),
                "eps_estimate_mean": value,
            }
            for date, value in values.items()
        ]

    estimates = FundamentalDataset(
        pd.DataFrame(
            snapshots("A", {"2019-03-01": 10.0, "2019-09-02": 12.0})
            + snapshots("B", {"2019-03-01": -10.0, "2019-09-02": -5.0})
        )
    )
    ds = price_dataset(["A", "B"])
    scores = EstimateRevision(estimates=estimates, window=63).compute(ds)
    # Well after the second snapshot: consensus rose 10 -> 12 over the window.
    assert scores["A"].loc["2019-12-02"] == pytest.approx(0.0)  # 12 vs 12 by then
    assert scores["A"].loc["2019-09-16"] == pytest.approx(0.2)  # 12 vs 10
    # Improving from a loss estimate scores positive via the |base| scaling.
    assert scores["B"].loc["2019-09-16"] == pytest.approx(0.5)
    # Before any snapshot is 63 bars old the factor is NaN, never backfilled.
    assert scores["A"].loc[:"2019-03-01"].isna().all()


# ------------------------------------------------- synthetic integration set
@pytest.fixture(scope="module")
def synth_funda() -> FundamentalDataset:
    src = SyntheticFundamentalSource(n_assets=6, seed=42, inception="2018-01-02")
    return src.fetch(src.symbols, "2018-01-02", "2021-12-31")


@pytest.fixture(scope="module")
def synth_prices() -> MarketDataset:
    return make_synthetic_dataset(
        n_assets=6, start="2020-01-01", end="2021-12-31", n_sectors=2, seed=42
    )


ALL_FUNDAMENTAL_FACTORS = [
    EarningsYield,
    BookToMarket,
    FCFYield,
    ReturnOnEquity,
    GrossMarginStability,
    Accruals,
    RevenueGrowth,
    EarningsGrowth,
]


@pytest.mark.parametrize("factor_cls", ALL_FUNDAMENTAL_FACTORS, ids=lambda c: c.__name__)
def test_shape_coverage_and_determinism_on_synthetic(
    factor_cls, synth_funda, synth_prices
):
    factor = factor_cls(synth_funda)
    scores = factor.compute(synth_prices)
    assert scores.index.equals(synth_prices.index)
    assert list(scores.columns) == synth_prices.symbols
    values = scores.to_numpy()
    assert not np.isinf(values[np.isfinite(values) | np.isnan(values)]).any()
    # With two years of pre-history every factor should be live by the
    # back half of the sample for every symbol.
    back_half = scores.iloc[len(scores) // 2 :]
    assert back_half.notna().all().all(), f"{factor.name} has NaN in the back half"
    again = factor_cls(synth_funda).compute(synth_prices)
    pd.testing.assert_frame_equal(scores, again)
