"""Fundamentals: PIT alignment hand-computed, container invariants, sources,
caching, and fundamental validation checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from quantlab.data import (
    FundamentalDataset,
    FundamentalDataValidator,
    SyntheticFundamentalSource,
    SyntheticSource,
    combine_events,
    pit_align,
    trailing_change,
    trailing_stat,
)
from quantlab.data.fundamentals import (
    CachedFundamentalSource,
    FundamentalSource,
    YahooFundamentalSource,
)
from quantlab.data.sources.base import validate_request
from quantlab.data.validation import (
    AccountingIdentityCheck,
    MissingStatementCheck,
    NonPositiveFundamentalCheck,
    RestatementCheck,
    Severity,
    StaleFundamentalValueCheck,
)
from quantlab.exceptions import DataError

# Calendar 2019 quarters published with a flat 45-day lag land on clean
# business days: Q1 -> May 15, Q2 -> Aug 14, Q3 -> Nov 14, Q4 -> Feb 14.
Q = pd.to_datetime(["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"])
PUB = Q + pd.Timedelta(days=45)


def make_dataset(rows: list[dict], **metadata) -> FundamentalDataset:
    return FundamentalDataset(pd.DataFrame(rows), metadata=metadata)


def simple_rows(
    symbol: str = "A",
    values: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
    item: str = "net_income",
    periods: pd.DatetimeIndex | None = None,
    pubs: pd.DatetimeIndex | None = None,
) -> list[dict]:
    periods = Q if periods is None else periods
    pubs = periods + pd.Timedelta(days=45) if pubs is None else pubs
    return [
        {"symbol": symbol, "period_end": pe, "publication_date": pub, item: v}
        for pe, pub, v in zip(periods, pubs, values, strict=True)
    ]


# ----------------------------------------------------------------- container
def test_constructor_normalizes_and_sorts():
    rows = simple_rows()[::-1]  # deliberately reversed
    ds = make_dataset(rows)
    assert ds.statements["period_end"].is_monotonic_increasing
    assert ds.symbols == ["A"]
    assert ds.items == ["net_income"]
    assert ds.n_statements == 4
    assert ds.start == Q[0]
    assert ds.end == Q[-1]


def test_constructor_rejects_structural_defects():
    with pytest.raises(DataError, match="required columns"):
        FundamentalDataset(pd.DataFrame({"symbol": ["A"], "period_end": [Q[0]]}))
    with pytest.raises(DataError, match="no item columns"):
        FundamentalDataset(
            pd.DataFrame(
                {"symbol": ["A"], "period_end": [Q[0]], "publication_date": [PUB[0]]}
            )
        )
    # A statement cannot be public before its quarter ends.
    early = simple_rows(pubs=pd.DatetimeIndex([Q[0] - pd.Timedelta(days=1)] * 4))
    with pytest.raises(DataError, match="precedes period_end"):
        make_dataset(early)
    dupes = simple_rows() + simple_rows()
    with pytest.raises(DataError, match="Duplicate"):
        make_dataset(dupes)
    bad_item = simple_rows()
    bad_item[0]["net_income"] = "not-a-number"
    with pytest.raises(DataError, match="not numeric"):
        make_dataset(bad_item)


def test_constructor_rejects_tz_aware_dates():
    rows = pd.DataFrame(simple_rows())
    rows["publication_date"] = rows["publication_date"].dt.tz_localize("UTC")
    with pytest.raises(DataError, match="tz-naive"):
        FundamentalDataset(rows)


def test_empty_dataset_is_allowed():
    empty = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "publication_date": pd.Series(dtype="datetime64[ns]"),
            "net_income": pd.Series(dtype=float),
        }
    )
    ds = FundamentalDataset(empty)
    assert ds.symbols == []
    assert ds.n_statements == 0
    panel = ds.pit_panel("net_income", pd.bdate_range("2020-01-01", periods=5), ["A"])
    assert panel.isna().all().all()


def test_select_and_unknown_symbol():
    ds = make_dataset(simple_rows("A") + simple_rows("B", values=(9, 9, 9, 9)))
    only_a = ds.select(["A"])
    assert only_a.symbols == ["A"]
    with pytest.raises(DataError, match="not in fundamental dataset"):
        ds.select(["A", "Z"])


def test_as_of_boundary_is_inclusive():
    ds = make_dataset(simple_rows())
    at_pub = ds.as_of(PUB[1])
    assert at_pub.n_statements == 2  # Q1 and Q2 are public, Q3/Q4 are not
    before_pub = ds.as_of(PUB[1] - pd.Timedelta(days=1))
    assert before_pub.n_statements == 1
    assert at_pub.metadata["as_of"] == str(PUB[1].date())


def test_quarterly_keeps_restatement_versions():
    rows = simple_rows(values=(1.0, 2.0, 3.0, 4.0))
    # Restate Q2: same period_end, later publication, different value.
    rows.append(
        {
            "symbol": "A",
            "period_end": Q[1],
            "publication_date": PUB[1] + pd.Timedelta(days=30),
            "net_income": 20.0,
        }
    )
    events = make_dataset(rows).quarterly("net_income")
    assert len(events) == 5  # both versions of Q2 survive for PIT alignment
    q2 = events[events["period_end"] == Q[1]].sort_values("publication_date")
    assert q2["value"].tolist() == [2.0, 20.0]
    with pytest.raises(DataError, match="Unknown fundamental item"):
        make_dataset(rows).quarterly("revenue")


# ------------------------------------------------------------ pit alignment
def test_pit_align_steps_exactly_at_publication():
    ds = make_dataset(simple_rows(values=(1.0, 2.0, 3.0, 4.0)))
    index = pd.bdate_range("2019-01-01", "2020-03-31")
    panel = ds.pit_panel("net_income", index, ["A"])
    series = panel["A"]
    assert series.loc[: PUB[0] - pd.Timedelta(days=1)].isna().all()
    assert series.loc[PUB[0]] == 1.0  # public on its publication date
    assert series.loc[PUB[1] - pd.Timedelta(days=1)] == 1.0
    assert series.loc[PUB[1]] == 2.0
    assert series.iloc[-1] == 4.0


def test_pit_align_restatement_switches_at_second_publication():
    v2_pub = pd.Timestamp("2019-07-01")
    rows = simple_rows(values=(1.0,), periods=Q[:1], pubs=PUB[:1])
    rows.append(
        {"symbol": "A", "period_end": Q[0], "publication_date": v2_pub, "net_income": 99.0}
    )
    panel = pit_align(
        make_dataset(rows).quarterly("net_income"),
        pd.bdate_range("2019-05-01", "2019-08-01"),
        ["A"],
    )
    # The original print is the honest answer between the two publications;
    # the restated value takes over exactly at its own publication date.
    assert panel["A"].loc[: "2019-05-14"].isna().all()
    assert (panel["A"].loc["2019-05-15":"2019-06-28"] == 1.0).all()
    assert (panel["A"].loc["2019-07-01":] == 99.0).all()


def test_pit_align_old_quarter_restated_late_never_displaces_newer():
    rows = [
        {"symbol": "A", "period_end": Q[0], "publication_date": PUB[0], "x": 1.0},
        {"symbol": "A", "period_end": Q[1], "publication_date": PUB[1], "x": 2.0},
        # Q1 restated AFTER Q2 was already published.
        {
            "symbol": "A",
            "period_end": Q[0],
            "publication_date": pd.Timestamp("2019-09-02"),
            "x": 99.0,
        },
    ]
    events = pd.DataFrame(rows).rename(columns={"x": "value"})
    panel = pit_align(events, pd.bdate_range("2019-05-01", "2019-10-01"), ["A"])
    # Between the two original publications the original Q1 value rules.
    assert panel["A"].loc["2019-05-15"] == 1.0
    assert panel["A"].loc["2019-08-13"] == 1.0
    # After Q2 is out it stays the answer, even after Q1's late restatement.
    assert panel["A"].loc["2019-08-14"] == 2.0
    assert panel["A"].loc["2019-09-03"] == 2.0


def test_pit_align_weekend_publication_flows_to_next_bar_only():
    saturday = pd.Timestamp("2019-05-18")
    events = pd.DataFrame(
        [{"symbol": "A", "period_end": Q[0], "publication_date": saturday, "value": 7.0}]
    )
    index = pd.bdate_range("2019-05-13", "2019-05-24")
    panel = pit_align(events, index, ["A"])
    assert panel["A"].loc["2019-05-17"] != panel["A"].loc["2019-05-17"]  # NaN Friday
    assert panel["A"].loc["2019-05-20"] == 7.0  # first bar on/after publication


def test_pit_align_uncovered_symbol_is_nan_and_index_validated():
    ds = make_dataset(simple_rows())
    index = pd.bdate_range("2019-01-01", "2019-12-31")
    panel = ds.pit_panel("net_income", index, ["A", "ZZZ"])
    assert panel["ZZZ"].isna().all()
    assert panel.columns.tolist() == ["A", "ZZZ"]
    with pytest.raises(DataError, match="DatetimeIndex"):
        pit_align(ds.quarterly("net_income"), pd.Index([1, 2, 3]), ["A"])


# --------------------------------------------------------- trailing kernels
def test_trailing_sum_hand_computed_and_gated_by_slowest_publication():
    late_pub = pd.Timestamp("2020-03-02")  # Q2 filed absurdly late...
    pubs = pd.DatetimeIndex([PUB[0], late_pub, PUB[2], PUB[3]])
    ds = make_dataset(simple_rows(values=(10.0, 20.0, 30.0, 40.0), pubs=pubs))
    ttm = ds.ttm("net_income")
    assert len(ttm) == 1
    row = ttm.iloc[0]
    assert row["value"] == 100.0
    assert row["period_end"] == Q[3]
    # ...so the TTM is usable only when its slowest constituent is public,
    # NOT at the Q4 publication in February.
    assert row["publication_date"] == late_pub

    panel = pit_align(ttm, pd.bdate_range("2020-02-01", "2020-03-31"), ["A"])
    assert panel["A"].loc["2020-02-14"] != panel["A"].loc["2020-02-14"]  # NaN
    assert panel["A"].loc[late_pub] == 100.0


def test_trailing_sum_refuses_windows_spanning_missing_quarters():
    periods = pd.to_datetime(["2019-03-31", "2019-06-30", "2019-12-31", "2020-03-31"])
    ds = make_dataset(simple_rows(values=(1.0, 1.0, 1.0, 1.0), periods=periods))
    # Any 4-quarter window here spans a hole (2019Q3 missing) -> no output.
    assert ds.ttm("net_income").empty


def test_trailing_std_hand_computed():
    values = (0.4, 0.6, 0.4, 0.6)
    events = make_dataset(simple_rows(values=values)).quarterly("net_income")
    out = trailing_stat(events, quarters=4, stat="std")
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(DataError, match="quarters"):
        trailing_stat(events, quarters=1)
    with pytest.raises(DataError, match="stat"):
        trailing_stat(events, stat="median")


def test_trailing_change_hand_computed_signs_and_zero_base():
    periods = pd.date_range("2018-03-31", periods=8, freq="QE")
    values = (10.0, -5.0, 0.0, 4.0, 12.0, 10.0, 3.0, 4.0)
    events = make_dataset(simple_rows(values=values, periods=periods)).quarterly(
        "net_income"
    )
    out = trailing_change(events, quarters=4)
    by_period = out.set_index("period_end")["value"]
    assert by_period[periods[4]] == pytest.approx((12.0 - 10.0) / 10.0)  # +20%
    # Negative base: -5 -> 10 is an improvement of 3x the base's magnitude.
    assert by_period[periods[5]] == pytest.approx((10.0 - (-5.0)) / 5.0)
    assert periods[6] not in by_period.index  # zero base -> dropped
    assert by_period[periods[7]] == pytest.approx(0.0)
    # Usability is the later of the two constituent publications (the newer
    # quarter here, since the lag is uniform).
    assert (out["publication_date"] == out["period_end"] + pd.Timedelta(days=45)).all()


def test_combine_events_inner_join_and_max_publication():
    a = make_dataset(simple_rows(values=(1.0, 2.0, 3.0, 4.0), item="x")).quarterly("x")
    b_pubs = PUB + pd.Timedelta(days=10)  # b is always published later
    b = make_dataset(
        simple_rows(values=(10.0, 20.0, 30.0, 40.0), item="y", pubs=b_pubs)
    ).quarterly("y")
    combined = combine_events(a, b.iloc[:3], lambda x, y: y - x)
    assert len(combined) == 3  # inner join drops b's missing Q4
    assert combined["value"].tolist() == [9.0, 18.0, 27.0]
    assert (combined["publication_date"] == b_pubs[:3]).all()


# ------------------------------------------------------------------- sources
def test_synthetic_fundamentals_deterministic_and_window_independent():
    src = SyntheticFundamentalSource(n_assets=6, seed=11, inception="2018-01-02")
    a = src.fetch(src.symbols, "2018-01-02", "2021-12-31")
    b = SyntheticFundamentalSource(n_assets=6, seed=11, inception="2018-01-02").fetch(
        src.symbols, "2018-01-02", "2021-12-31"
    )
    assert a.equals(b)
    # A narrower window and a symbol subset agree exactly on the overlap.
    sub = src.fetch(src.symbols[:3], "2019-01-01", "2020-12-31")
    merged = sub.statements.merge(
        a.statements, on=["symbol", "period_end"], suffixes=("_sub", "_full")
    )
    assert len(merged) == len(sub.statements)
    for item in ("revenue", "net_income", "total_equity", "publication_date"):
        col = item if item != "publication_date" else item
        assert (merged[f"{col}_sub"] == merged[f"{col}_full"]).all()


def test_synthetic_fundamentals_internally_consistent():
    src = SyntheticFundamentalSource(n_assets=5, seed=3, inception="2018-01-02")
    ds = src.fetch(src.symbols, "2018-01-02", "2022-12-31")
    s = ds.statements
    # Accounting identity holds exactly (up to float noise).
    gap = (s["total_assets"] - s["total_liabilities"] - s["total_equity"]).abs()
    assert (gap <= 1e-6 * s["total_assets"]).all()
    # Gross profit is a (5%..95%) margin on revenue.
    margin = s["gross_profit"] / s["revenue"]
    assert margin.between(0.05, 0.95).all()
    # Equity evolves by a constant retained fraction of net income: the
    # implied retention ratio is constant per symbol.
    for _, g in s.groupby("symbol"):
        g = g.sort_values("period_end")
        retention = (g["total_equity"].diff() / g["net_income"]).dropna()
        assert retention.std() == pytest.approx(0.0, abs=1e-9)
        assert 0.5 <= retention.iloc[0] <= 1.0
    # Publication lag respects lag + jitter bounds.
    lag = (s["publication_date"] - s["period_end"]).dt.days
    assert lag.between(45, 55).all()


def test_synthetic_fundamentals_passes_default_validation():
    src = SyntheticFundamentalSource(n_assets=6, seed=7, inception="2018-01-02")
    ds = src.fetch(src.symbols, "2018-01-02", "2021-12-31")
    report = FundamentalDataValidator().validate(ds)
    assert report.passed, report.to_frame()
    assert len(report.issues) == 0


def test_synthetic_fundamentals_matches_price_universe_naming():
    prices = SyntheticSource(n_assets=4, n_sectors=2, seed=7)
    funda = SyntheticFundamentalSource(n_assets=4, seed=7)
    assert prices.symbols == funda.symbols


def test_synthetic_fundamentals_request_validation():
    src = SyntheticFundamentalSource(n_assets=3, seed=7, inception="2020-01-01")
    with pytest.raises(DataError, match="precedes inception"):
        src.fetch(src.symbols, "2019-01-01", "2020-12-31")
    with pytest.raises(DataError, match="Unknown synthetic symbols"):
        src.fetch(["NOPE"], "2020-01-01", "2020-12-31")
    with pytest.raises(DataError, match="No completed fiscal quarters"):
        src.fetch(src.symbols, "2020-01-01", "2020-02-01")


def test_yahoo_mapper_offline_with_vendor_shaped_frames():
    periods = pd.to_datetime(["2024-03-31", "2023-12-31"])
    income = pd.DataFrame(
        [[100.0, 90.0], [40.0, 35.0], [10.0, 8.0]],
        index=["Total Revenue", "Gross Profit", "Net Income Common Stockholders"],
        columns=periods,
    )
    balance = pd.DataFrame(
        [[1000.0, 950.0], [600.0, 580.0], [400.0, 370.0], [50.0, 50.0]],
        index=[
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Stockholders Equity",
            "Ordinary Shares Number",
        ],
        columns=periods,
    )
    cashflow = pd.DataFrame(
        [[12.0, 11.0], [-3.0, -2.5]],
        index=["Operating Cash Flow", "Capital Expenditure"],
        columns=periods,
    )
    mapped = YahooFundamentalSource._map_statements(income, balance, cashflow)
    assert list(mapped.index) == sorted(periods)
    assert mapped.loc[periods[0], "revenue"] == 100.0
    # Fallback candidate was used for net income.
    assert mapped.loc[periods[0], "net_income"] == 10.0
    assert mapped.loc[periods[1], "capital_expenditure"] == -2.5
    assert mapped.loc[periods[0], "shares_outstanding"] == 50.0
    # Missing statements produce NaN items, empty inputs an empty frame.
    partial = YahooFundamentalSource._map_statements(income, None, pd.DataFrame())
    assert partial["total_assets"].isna().all()
    assert YahooFundamentalSource._map_statements(None, None, None).empty


# --------------------------------------------------------------------- cache
class RecordingFundamentalSource(FundamentalSource):
    """Deterministic in-memory source that records every fetch call.

    ``trailing_quarters`` mimics snapshot vendors (Yahoo) that ignore the
    requested start and serve only the last k quarters before ``end``.
    """

    name = "recording_fund"

    def __init__(self, trailing_quarters: int | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]] = []
        self.trailing_quarters = trailing_quarters

    def fetch(self, symbols, start, end) -> FundamentalDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        self.calls.append((tuple(symbols_list), start_ts, end_ts))
        quarters = pd.date_range("2018-03-31", end_ts, freq="QE")
        if self.trailing_quarters is not None:
            quarters = quarters[-self.trailing_quarters :]
        quarters = quarters[(quarters >= start_ts) & (quarters <= end_ts)]
        if len(quarters) == 0:
            raise DataError("no statements in range")
        rows = []
        for sym in symbols_list:
            base = float(sum(ord(c) for c in sym))
            for i, pe in enumerate(quarters):
                rows.append(
                    {
                        "symbol": sym,
                        "period_end": pe,
                        "publication_date": pe + pd.Timedelta(days=45),
                        "revenue": base + float(i),
                    }
                )
        return FundamentalDataset(pd.DataFrame(rows), metadata={"source": self.name})


@pytest.fixture
def cached(tmp_path):
    src = RecordingFundamentalSource()
    return src, CachedFundamentalSource(src, cache_dir=tmp_path)


def test_fundamental_cache_first_fetch_hits_source_second_is_pure(cached):
    src, cache = cached
    a = cache.fetch(["AAA", "BBB"], "2018-01-01", "2019-12-31")
    assert len(src.calls) == 1
    b = cache.fetch(["AAA", "BBB"], "2018-01-01", "2019-12-31")
    assert len(src.calls) == 1
    assert a.equals(b)
    assert b.metadata["cached"] is True


def test_fundamental_cache_tail_extension_fetches_only_gap(cached):
    src, cache = cached
    cache.fetch(["AAA"], "2018-01-01", "2019-12-31")
    cache.fetch(["AAA"], "2018-01-01", "2020-12-31")
    assert len(src.calls) == 2
    _, gap_start, gap_end = src.calls[1]
    assert gap_start == pd.Timestamp("2020-01-01")
    assert gap_end == pd.Timestamp("2020-12-31")


def test_fundamental_cache_gap_without_statements_not_rethrashed(cached):
    src, cache = cached
    cache.fetch(["AAA"], "2018-01-01", "2019-12-31")
    # Extending the end into a window with no quarter end: the source raises
    # internally, the range is remembered as covered, and repeat calls do
    # not hit the source again.
    cache.fetch(["AAA"], "2018-01-01", "2020-01-15")
    n = len(src.calls)
    out = cache.fetch(["AAA"], "2018-01-01", "2020-01-15")
    assert len(src.calls) == n
    assert out.n_statements == 8  # 2018Q1..2019Q4


def test_fundamental_cache_accumulates_snapshot_vendor_history(tmp_path):
    src = RecordingFundamentalSource(trailing_quarters=4)
    cache = CachedFundamentalSource(src, cache_dir=tmp_path)
    first = cache.fetch(["AAA"], "2018-01-01", "2019-12-31")
    assert first.n_statements == 4  # vendor only serves 2019Q1..Q4
    later = cache.fetch(["AAA"], "2018-01-01", "2020-12-31")
    # A year later the vendor serves 2020Q1..Q4 only, but the cache kept the
    # 2019 quarters: history accumulates instead of being overwritten.
    assert later.n_statements == 8
    fresh = src.fetch(["AAA"], "2018-01-01", "2020-12-31")
    assert fresh.n_statements == 4


def test_fundamental_cache_persists_across_instances(tmp_path):
    src1 = RecordingFundamentalSource()
    CachedFundamentalSource(src1, cache_dir=tmp_path).fetch(
        ["AAA"], "2018-01-01", "2019-12-31"
    )
    src2 = RecordingFundamentalSource()
    out = CachedFundamentalSource(src2, cache_dir=tmp_path).fetch(
        ["AAA"], "2018-01-01", "2019-12-31"
    )
    assert src2.calls == []
    assert out.n_statements == 8


def test_fundamental_cache_corruption_refetches(cached, tmp_path):
    src, cache = cached
    cache.fetch(["AAA"], "2018-01-01", "2019-12-31")
    parquet = next(tmp_path.rglob("AAA.parquet"))
    parquet.write_bytes(b"not a parquet file")
    out = cache.fetch(["AAA"], "2018-01-01", "2019-12-31")
    assert len(src.calls) == 2
    assert out.n_statements == 8


def test_fundamental_cache_clear_and_range(cached):
    src, cache = cached
    cache.fetch(["AAA", "BBB"], "2018-01-01", "2019-12-31")
    assert cache.cached_range("AAA") is not None
    assert cache.clear(["AAA"]) == 1
    assert cache.cached_range("AAA") is None
    assert cache.clear() == 1  # remaining BBB


# ---------------------------------------------------------------- validation
def _consistent_rows(symbol: str, periods: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for i, pe in enumerate(periods):
        equity, liabilities = 400.0 + i, 600.0
        rows.append(
            {
                "symbol": symbol,
                "period_end": pe,
                "publication_date": pe + pd.Timedelta(days=45),
                "revenue": 100.0 + i,
                "net_income": 10.0 + i,
                "operating_cash_flow": 9.0 + i,
                "total_assets": equity + liabilities,
                "total_liabilities": liabilities,
                "total_equity": equity,
                "shares_outstanding": 50.0,
            }
        )
    return rows


def test_missing_statement_check_flags_gap_and_stale_coverage():
    periods_a = pd.date_range("2018-03-31", periods=8, freq="QE").delete(3)
    periods_b = pd.date_range("2018-03-31", periods=4, freq="QE")  # ends early
    ds = make_dataset(_consistent_rows("A", periods_a) + _consistent_rows("B", periods_b))
    issues = MissingStatementCheck().run(ds)
    assert any(i.symbol == "A" and "gap" in i.message for i in issues)
    assert any(i.symbol == "B" and "trails" in i.message for i in issues)
    clean = make_dataset(_consistent_rows("A", pd.date_range("2018-03-31", periods=8, freq="QE")))
    assert not MissingStatementCheck().run(clean)


def test_non_positive_fundamental_check():
    rows = _consistent_rows("A", Q)
    rows[0]["shares_outstanding"] = 0.0
    rows[1]["total_assets"] = -5.0
    rows[2]["revenue"] = -1.0
    issues = NonPositiveFundamentalCheck().run(make_dataset(rows))
    assert len(issues) == 3
    assert all(i.severity == Severity.ERROR for i in issues)


def test_accounting_identity_check():
    rows = _consistent_rows("A", Q)
    rows[2]["total_assets"] = rows[2]["total_assets"] * 1.5  # break A = L + E
    issues = AccountingIdentityCheck(rel_tol=0.05).run(make_dataset(rows))
    assert issues and issues[0].severity == Severity.WARNING
    assert not AccountingIdentityCheck().run(make_dataset(_consistent_rows("A", Q)))


def test_stale_fundamental_value_check():
    rows = _consistent_rows("A", pd.date_range("2018-03-31", periods=6, freq="QE"))
    for r in rows[:4]:
        r["revenue"] = 123.45  # identical revenue four quarters running
    issues = StaleFundamentalValueCheck(max_run=3).run(make_dataset(rows))
    assert issues and issues[0].count == 4
    assert "revenue" in issues[0].message


def test_restatement_check_info_and_warning():
    rows = _consistent_rows("A", Q) + _consistent_rows("B", Q)
    rows.append(
        {**rows[0], "publication_date": PUB[0] + pd.Timedelta(days=20)}
    )  # A: identical re-print -> INFO
    restated = {**rows[4], "publication_date": PUB[0] + pd.Timedelta(days=20)}
    restated["net_income"] = rows[4]["net_income"] * 2.0  # B: material change
    rows.append(restated)
    issues = RestatementCheck().run(make_dataset(rows))
    by_symbol = {i.symbol: i for i in issues}
    assert by_symbol["A"].severity == Severity.INFO
    assert by_symbol["B"].severity == Severity.WARNING


def test_fundamental_validator_aggregates_and_raises():
    rows = _consistent_rows("A", Q)
    rows[0]["shares_outstanding"] = -1.0
    report = FundamentalDataValidator().validate(make_dataset(rows))
    assert not report.passed
    assert report.n_errors >= 1
    with pytest.raises(Exception, match="non_positive_fundamental"):
        report.raise_for_errors()


# ---------------------------------------------------------------- coverage


def test_coverage_table_hand_computed():
    # A's Q1 publishes May 15; B's Q1 publishes Aug 14. Before either
    # publication nothing is covered, in between only A, afterwards both.
    rows = simple_rows("A", values=(1.0,), periods=Q[:1]) + simple_rows(
        "B", values=(2.0,), periods=Q[:1], pubs=pd.DatetimeIndex([PUB[1]])
    )
    ds = make_dataset(rows)
    index = pd.DatetimeIndex(["2019-05-01", "2019-06-01", "2019-09-01"])
    table = ds.coverage_table("net_income", index)
    assert list(table.columns) == ["n_covered", "n_universe", "pct_covered"]
    assert table["n_covered"].tolist() == [0, 1, 2]
    assert table["n_universe"].tolist() == [2, 2, 2]
    assert table["pct_covered"].tolist() == [0.0, 50.0, 100.0]


def test_coverage_table_counts_uncovered_universe_symbols():
    ds = make_dataset(simple_rows("A", values=(1.0,), periods=Q[:1]))
    index = pd.DatetimeIndex(["2019-06-01"])
    table = ds.coverage_table("net_income", index, symbols=["A", "YYY", "ZZZ"])
    assert table.loc[index[0], "n_covered"] == 1
    assert table.loc[index[0], "n_universe"] == 3
    assert table.loc[index[0], "pct_covered"] == pytest.approx(100.0 / 3.0)


def test_coverage_table_empty_universe_raises():
    ds = make_dataset(simple_rows("A"))
    with pytest.raises(DataError, match="at least one symbol"):
        ds.coverage_table("net_income", pd.DatetimeIndex(["2019-06-01"]), symbols=[])
