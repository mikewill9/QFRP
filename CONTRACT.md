# QFRP App — Data Contract

**Status:** v1, frozen as of week 1
**Owners:** Michael (everything behind the contract), BP (everything in front of it)

Changes to this file require both of us to agree. Everything else in our
respective directories, we change freely.

---

## 1. What this is

BP builds the Streamlit interface. Michael builds the research and statistics.
This document defines the single boundary between them.

BP calls three functions and reads named fields off what they return. He never
imports anything from `quantlab` directly. Michael can restructure anything
behind these functions without breaking the app, as long as the names and types
below stay the same.

Until the real implementation lands (week 2), `app/api.py` returns hardcoded
fake data in exactly this shape. BP builds the entire interface against the
fake. When the real version replaces it, nothing on his side changes.

---

## 2. Design decisions

**Fixed menu, precomputed results.** The app offers 6 factors × 3 date ranges =
18 configurations. All statistics are computed offline by
`scripts/precompute_results.py` and saved to `app/data/results.parquet`. The app
reads that file. It never runs a backtest.

Rationale: a single backtest takes ~0.3s, so 1,000 permutations takes ~5
minutes. Precomputing makes the app instant and removes any long-running
computation from the user's path. The tradeoff is that users cannot test
arbitrary configurations, which is acceptable — 18 options is plenty for someone
to hunt for a good-looking result, which is the point of the demo.

**Verdict is computed at display time, not stored.** The multiple-testing
adjustment depends on how many configurations the user has viewed this session,
which is session state BP owns. So `get_analysis()` returns raw statistics, and
`verdict_for()` applies the session context. See §5.

---

## 3. Functions

### `list_factors() -> list[FactorInfo]`

Returns the factor menu, for populating a dropdown.

```
FactorInfo
├── key: str          # "momentum", stable identifier, use this in calls
├── label: str        # "Momentum (12-1)", show this to users
└── description: str  # one sentence explaining what it measures
```

### `list_periods() -> list[PeriodInfo]`

Returns the date range menu.

```
PeriodInfo
├── key: str          # "2020_2024"
├── label: str        # "2020-2024"
├── start_date: str   # "2020-01-01"
└── end_date: str     # "2024-12-31"
```

### `get_analysis(factor_key: str, period_key: str) -> AnalysisResult`

The main call. Always returns instantly (reads from the precomputed file).
Raises `KeyError` if either key is not in the menus above.

---

## 4. AnalysisResult

```
AnalysisResult
│
├── Identity
│   ├── factor_key: str
│   ├── factor_label: str
│   ├── period_label: str          # "2020-2024"
│   ├── start_date: str
│   └── end_date: str
│
├── Headline performance
│   ├── sharpe: float              # annualized, e.g. 0.39
│   ├── total_return: float        # 0.34 means +34%
│   ├── annualized_vol: float      # 0.12 means 12%
│   ├── max_drawdown: float        # NEGATIVE, e.g. -0.28 means -28%
│   ├── hit_rate: float            # 0.52 means 52% of periods positive
│   └── n_trades: int
│
├── Curves
│   └── equity_curve: pd.DataFrame
│       ├── index: DatetimeIndex
│       ├── "strategy": float      # normalized to 1.0 at start
│       └── "benchmark": float     # normalized to 1.0 at start
│
├── Significance
│   ├── p_value: float             # 0.0 to 1.0
│   ├── null_sharpes: list[float]  # RAW, unbinned, length n_permutations
│   └── n_permutations: int        # 1000
│
├── Cost sensitivity
│   └── cost_stress: list[CostPoint]
│       └── CostPoint
│           ├── multiplier: float  # 1.0, 2.0, 3.0
│           ├── sharpe: float
│           └── total_return: float
│
└── Scope
    └── limitations: list[str]     # display verbatim, do not edit
```

### Notes for BP

- `max_drawdown` is **negative by convention**. Display as-is or take
  `abs()` — but don't show `--28%`.
- `null_sharpes` is deliberately raw and unbinned so you can tune histogram bin
  width in Plotly without Michael regenerating anything.
- `equity_curve` benchmark is an **equal-weight long-only basket of the same
  universe**. The strategy is long-short and roughly market-neutral, so this is
  context, not a like-for-like comparison. Label it as such — this is a point of
  honesty the judges will notice.
- Percentages arrive as decimals (0.34, not 34). Format at display time.

---

## 5. The verdict

### `verdict_for(result: AnalysisResult, n_configs_viewed: int) -> Verdict`

```
Verdict
├── label: str           # "PASS" | "FAIL" | "INCONCLUSIVE"
├── headline: str        # one line, e.g. "Indistinguishable from luck"
├── raw_p_value: float   # same as result.p_value
├── adjusted_p_value: float
├── n_configs_viewed: int
└── explanation: str     # 1-2 sentences for display under the verdict
```

BP tracks `n_configs_viewed` in `st.session_state` — increment it each time a
new (factor, period) pair is viewed for the first time this session. Pass it in.

The adjustment is what makes the app's argument: the more configurations you
look at, the more likely one of them looks good by chance, and the raw p-value
doesn't account for that.

---

## 6. Directory split

```
src/quantlab/         # Michael only — the research library
scripts/              # Michael only — precompute script
app/
├── api.py            # Michael only — the contract implementation
├── data/             # Michael only — precomputed results
└── ui/               # BP only — everything Streamlit
CONTRACT.md           # both, by agreement
```

Michael does not edit `app/ui/`. BP does not edit anything else.

---

## 7. Build order

**Week 1**
- Michael: write `app/api.py` returning hardcoded fake data in this shape.
  Commit it day one so BP is unblocked.
- BP: build the interface against the fake. Layout, dropdowns, charts, verdict
  panel.
- Michael: build the real permutation test in the library, with tests.

**Week 2**
- Michael: factor registry, parameterized backtest runner, precompute script.
  Replace the fake implementation with the real one.
- BP: real charts against real numbers.

If the shape in this document holds, week 2 integration is a one-line swap.

---

## 8. Open items

- [ ] Confirm which 6 factors make the menu
- [ ] Confirm which 3 date ranges
- [ ] Decide PASS/FAIL/INCONCLUSIVE thresholds
- [ ] Write the `limitations` strings
