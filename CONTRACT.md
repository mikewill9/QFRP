# QFRP App — Data Contract

**Status:** v2 (week 2, real implementation). Supersedes the frozen v1.
**Owners:** Michael (everything behind the contract), BP (everything in front of it)

Changes to this file require both of us to agree. Everything else in our
respective directories, we change freely.

---

## 1. What this is

BP builds the web interface (React frontend + FastAPI bridge in `app/ui/`).
Michael builds the research and statistics. This document defines the single
boundary between them.

BP calls four functions in `app/api.py` and reads named fields off what they
return. He never imports anything from `quantlab` directly. Michael can
restructure anything behind these functions without breaking the app, as long
as the names and types below stay the same.

`app/api.py` is now the real implementation: every number it returns comes from
the precomputed permutation results. Nothing in it is fabricated.

---

## 2. Design decisions

**Fixed menu, precomputed results.** The app offers **11 price factors × 5
date ranges = 55 configurations**. Fundamental factors (value, quality, growth)
are not offered: free data has no real fundamental history before ~2025.

All statistics are computed offline by `scripts/run_permutations.py`, one JSON
file per configuration in `outputs/permutations/{factor}_{period}.json`. The app
reads those files. It never runs a backtest. The directory can be overridden
with the `QFRP_RESULTS_DIR` environment variable (e.g. for a deployment that
ships the files somewhere else; `outputs/` is not in git).

Rationale: one 20-year backtest takes ~1s, and each configuration runs 2,000
shuffled backtests, so the full set is about an hour of compute. Precomputing
makes the app instant. The tradeoff is that users cannot test arbitrary
configurations, which is acceptable — 55 options is plenty for someone to hunt
for a good-looking result, which is the point of the demo.

**Verdict is computed at display time, not stored.** The multiple-testing
adjustment depends on how many configurations the user has viewed this session,
which is session state BP owns. So `get_analysis()` returns raw statistics, and
`verdict_for()` applies the session context. See §5.

**Significance is tested before costs; costs are tested separately.** See §4
"Gross vs net". This is the single most important thing to label correctly.

---

## 3. Functions

### `list_factors() -> list[FactorInfo]`

The factor menu (the 11 price factors), for populating a dropdown. Comes from
the same catalog the permutation runner uses
(`quantlab.backtest.pipeline.PRICE_FACTOR_BUILDERS`).

```
FactorInfo
├── key: str          # "momentum", stable identifier, use this in calls
├── label: str        # "Momentum (12-1)", show this to users
└── description: str  # one sentence: what the portfolio buys and sells
```

Keys: `momentum`, `ts_momentum`, `st_reversal`, `price_to_ma`,
`realized_vol`, `ewma_vol`, `market_beta`, `pca_reversal`,
`amihud_illiquidity`, `dollar_volume`, `roll_spread`.

### `list_periods() -> list[PeriodInfo]`

The date range menu (the 5 periods), from `PERIODS` in
`scripts/run_permutations.py`.

```
PeriodInfo
├── key: str          # "2020_2024"
├── label: str        # "2020-2024"
├── start_date: str   # "2020-01-01"
└── end_date: str     # "2024-12-31"
```

| key | label | dates |
|---|---|---|
| `2005_2024` | 2005-2024 | 2005-01-01 .. 2024-12-31 (full sample) |
| `2005_2009` | 2005-2009 | includes the 2008 crisis |
| `2010_2014` | 2010-2014 | |
| `2015_2019` | 2015-2019 | |
| `2020_2024` | 2020-2024 | |

The full sample overlaps the other four; they are not independent tests.

### `get_analysis(factor_key: str, period_key: str, k: int = 1) -> AnalysisResult`

The main call. Always returns instantly (reads one JSON file).

- `k`: the number of configurations to adjust the p-value for — pass the
  number the user has viewed this session. `adjusted_p_value` is
  `bonferroni_adjust(p_value, k)`. Defaults to 1 (no adjustment).

Errors:

| Situation | Exception | Suggested HTTP |
|---|---|---|
| unknown `factor_key` or `period_key` | `KeyError` | 404 |
| valid keys, results file not written yet | `ResultNotComputedError` (a `KeyError` subclass — catch it first) | 503 |
| `k` < 1 | `ValueError` | 422 |
| `k` not an integer (incl. `True`) | `TypeError` | 422 |

### `verdict_for(result: AnalysisResult, n_configs_viewed: int) -> Verdict`

See §5.

---

## 4. AnalysisResult

```
AnalysisResult
│
├── Identity
│   ├── factor_key: str
│   ├── factor_label: str
│   ├── period_label: str             # "2020-2024"
│   ├── start_date: str
│   └── end_date: str
│
├── Headline performance (real strategy, NET of 1x trading costs)
│   ├── sharpe_1x: float | None       # annualized, e.g. 0.39; None if undefined
│   ├── total_return: float           # 0.34 means +34%
│   ├── annualized_vol: float         # 0.12 means 12%
│   ├── max_drawdown: float           # NEGATIVE, e.g. -0.28 means -28%
│   ├── hit_rate: float               # 0.52 means 52% of DAYS positive
│   └── turnover: float               # annualized; 5.5 = book traded 5.5x a year
│
├── Curve
│   └── equity_curve: pd.DataFrame    # net of 1x costs
│       ├── index: DatetimeIndex
│       └── "strategy": float         # normalized to 1.0 at start
│
├── Significance (GROSS: no costs on either side)
│   ├── sharpe_gross: float | None    # the Sharpe the p-value tests
│   ├── null_sharpes_gross: list[float]  # RAW, unbinned; NaN draws removed
│   ├── n_permutations: int           # 2000 shuffles run
│   ├── n_dropped: int                # shuffles excluded because Sharpe was NaN
│   ├── p_value: float                # raw, one-sided, 0.0 to 1.0
│   ├── adjusted_p_value: float       # bonferroni_adjust(p_value, k)
│   ├── k: int                        # what adjusted_p_value accounts for
│   └── null_turnover: float          # shuffled portfolios' annualized turnover
│
├── Cost sensitivity (real strategy)
│   └── cost_stress: list[CostPoint]
│       └── CostPoint
│           ├── multiplier: float     # 1.0, 2.0, 3.0
│           └── sharpe: float | None
│
├── Breadth
│   └── names_per_year: dict[str, float]  # {"2005": 124.0, ...} stocks with a score
│
└── Scope
    └── limitations: list[str]        # display verbatim, do not edit
```

### Gross vs net — read this

There are two Sharpe ratios and they answer different questions:

- **`sharpe_gross` → significance.** The p-value compares the real strategy's
  gross Sharpe against `null_sharpes_gross`: the same strategy run 2,000 times
  on randomly reshuffled factor rankings, also gross. Reshuffled portfolios
  trade about twice as much (`null_turnover` vs `turnover`), so comparing
  after costs would handicap them and make the real factor look more
  significant than it is. **On the null histogram, draw the observed line at
  `sharpe_gross`, not `sharpe_1x`.**
- **`sharpe_1x` / `cost_stress` → is it worth trading?** The real strategy
  with modelled costs at 1x, 2x and 3x (spread and commission scaled; market
  impact kept at 1x).

A factor can be significant gross and still lose money net of costs. The
verdict explanation says so when that happens.

### Changes from v1

| v1 field | v2 | why |
|---|---|---|
| `sharpe` | `sharpe_1x` + `sharpe_gross` | two different questions (above) |
| `null_sharpes` | `null_sharpes_gross` | label which Sharpe it is comparable with |
| `n_trades` | `turnover` | trade counts depend on rebalance tolerance; turnover is the standard measure |
| `equity_curve["benchmark"]` | removed | not in the precomputed results (open item) |
| `CostPoint.total_return` | removed | only Sharpe is stored at 2x/3x |
| — | `adjusted_p_value`, `k`, `n_dropped`, `null_turnover`, `names_per_year` | new |

### Notes for BP

- `max_drawdown` is **negative by convention**. Display as-is or take
  `abs()` — but don't show `--28%`.
- `null_sharpes_gross` is deliberately raw and unbinned so you can tune
  histogram bin width without Michael regenerating anything.
- Sharpe fields can be `None` (serialized as `null`) if a backtest's Sharpe
  was undefined; handle it rather than printing "None".
- Percentages arrive as decimals (0.34, not 34). Format at display time.
- Every result is on a survivors-only universe (today's large caps). The first
  `limitations` string says so; keep it visible.

---

## 5. The verdict

### `verdict_for(result: AnalysisResult, n_configs_viewed: int) -> Verdict`

```
Verdict
├── label: str           # "PASS" | "FAIL" | "INCONCLUSIVE"
├── headline: str        # one line, e.g. "Indistinguishable from luck"
├── raw_p_value: float   # same as result.p_value
├── adjusted_p_value: float  # bonferroni_adjust(raw_p_value, n_configs_viewed)
├── n_configs_viewed: int
└── explanation: str     # plain English, 2-5 sentences, display under the verdict
```

The adjusted p-value is banded (cutoffs are named constants `PASS_BELOW` and
`FAIL_ABOVE` at the top of `app/api.py`):

| adjusted p | label |
|---|---|
| < 0.05 | PASS |
| 0.05 to 0.20 (both ends included) | INCONCLUSIVE |
| > 0.20 | FAIL |

The verdict judges the **signal** (gross significance). Profitability after
costs is shown by `cost_stress`, and the explanation adds a note when the 1x
Sharpe is zero or negative. When `n_configs_viewed` > 1 it also states what the
verdict would be for a single test (the same bands applied to the raw p-value).

`n_configs_viewed` must be an integer >= 1 (same errors as `k` in §3). Track it
in frontend session state — increment it each time a new (factor, period) pair
is viewed for the first time this session — and pass it in. Pass the same
number as `k` to `get_analysis` so the two adjusted p-values agree.

Why adjust: Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected
Returns" (Review of Financial Studies 29(1)), show that after hundreds of
factors have been tried, p < 0.05 mostly admits false discoveries. The more
configurations you look at, the more likely one of them looks good by chance,
and the raw p-value doesn't account for that. Bonferroni (`p × k`, capped at
1) is the simplest, most conservative correction. With 2,000 shuffles the
smallest raw p-value is 1/2001 ≈ 0.0005, so a PASS is still reachable after
viewing all 55 configurations only if no shuffle beat the real Sharpe.

---

## 6. Directory split

```
src/quantlab/         # Michael only — the research library
scripts/              # Michael only — run_permutations.py (the precompute)
outputs/permutations/ # Michael only — precomputed results (not in git)
app/
├── api.py            # Michael only — the contract implementation
└── ui/               # BP only — frontend and FastAPI server
CONTRACT.md           # both, by agreement
```

Michael does not edit `app/ui/`. BP does not edit anything else.

---

## 7. Deployment requirements

- The server environment needs `quantlab` installed (`pip install -e .` from
  the repo root): `app/api.py` imports it.
- The 55 result JSONs must be present on the server, at
  `outputs/permutations/` or wherever `QFRP_RESULTS_DIR` points.

---

## 8. Open items

- [x] Which factors make the menu — the 11 price factors
- [x] Which date ranges — the 5 periods above
- [x] PASS/FAIL/INCONCLUSIVE thresholds — §5
- [x] `limitations` strings — drafted in `app/api.py`, Michael to review
- [ ] Benchmark curve: an equal-weight long-only basket was in v1 but is not
      precomputed. Add a separate precompute step, or drop it from the UI.
- [ ] BP: update `app/ui/server.py` and the Analyzer page to the v2 field names
      (§4 "Changes from v1").
- [ ] Decide how the result JSONs reach the deployed server.
