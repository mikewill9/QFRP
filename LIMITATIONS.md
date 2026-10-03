# Limitations

## Data
- **Survivorship bias.** The 148-stock list was chosen in 2026, so it includes future winners (TSLA, META) and excludes companies that failed (Lehman, Bear Stearns). Results for 2005–2009 are the least reliable. EA was dropped mid-project after being delisted, a live example of the problem.
- **Universe grows over time.** Late IPOs (META 2012, ABBV 2013, etc.) mean 126 names in 2005 vs 148 by 2016.
- **Price factors only.** Free fundamental data only goes back to about 2025, so value, quality and growth factors can't be tested.
- **Data errors.** DHR shows a false +61% jump on 2016-07-05 (Fortive spin-off, mis-adjusted by Yahoo). Yahoo also occasionally returns unadjusted prices; the dataset was verified against independent downloads, but future downloads could be affected.

## Method
- **Significance tested before costs.** The p-value uses zero-cost backtests, because shuffled portfolios trade about twice as much as real ones and would be unfairly penalized. Costs are tested separately at 1x/2x/3x.
- **Shuffle method.** Stocks are reshuffled every rebalance. A "hold" shuffle that keeps random picks fixed could produce a wider null. We checked this and found no difference at 100 shuffles, but that sample can't rule out a small effect.
- **Bonferroni is conservative.** It assumes tests are independent, but periods overlap (2005–2024 contains the other four), so the adjustment is stricter than necessary.
- **Minimum p-value.** With 2,000 shuffles, p can't go below about 0.0005.
- - **Not every borderline config was rerun.** market_beta 2005_2024 sits near a verdict cutoff at higher k (2 to 5) and was not rerun at 20,000 shuffles (about 11 hours), so its verdict uses 2,000 shuffles.

## App
- **Counter resets on page refresh.** A user could avoid the multiple-testing penalty by refreshing.
- **Cost model is simplified.** Commission, spread and market impact are modeled, but not borrow costs for short positions.