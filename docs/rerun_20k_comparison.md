## N=2,000 vs N=20,000 (verdicts at k=1)

| factor | period | p_2k | p_20k | verdict_2k | verdict_20k | reverse_p_2k | reverse_p_20k | flipped | z | selected for |
|---|---|---|---|---|---|---|---|---|---|---|
| amihud_illiquidity | 2005_2009 | 0.0005 | 0.0001 | PASS | PASS | 1.0000 | 0.9999 | no | -0.7 | k=1: PASS |
| amihud_illiquidity | 2010_2014 | 0.0015 | 0.0003 | PASS | PASS | 0.9990 | 0.9998 | no | -1.4 | k=1: PASS |
| amihud_illiquidity | 2015_2019 | 0.0050 | 0.0080 | PASS | PASS | 0.9955 | 0.9920 | no | +1.8 | k=1: PASS |
| dollar_volume | 2005_2009 | 0.9930 | 0.9942 | FAIL (reversed) | FAIL (reversed) | 0.0075 | 0.0059 | no | +0.6 | k=4: reverse p near 0.05/4; k=5: reverse p near 0.05/5 |
| dollar_volume | 2010_2014 | 0.9950 | 0.9958 | FAIL (reversed) | FAIL (reversed) | 0.0055 | 0.0043 | no | +0.5 | k=5: reverse p near 0.05/5 |
| ewma_vol | 2005_2009 | 0.9580 | 0.9590 | FAIL (reversed) | FAIL (reversed) | 0.0425 | 0.0410 | no | +0.2 | k=1: reverse p near 0.05 |
| ewma_vol | 2010_2014 | 0.9385 | 0.9275 | FAIL | FAIL | 0.0620 | 0.0726 | no | -2.0 | k=1: reverse p near 0.05 |
| price_to_ma | 2015_2019 | 0.0035 | 0.0056 | PASS | PASS | 0.9970 | 0.9944 | no | +1.5 | k=1: PASS |
| price_to_ma | 2020_2024 | 0.1029 | 0.0986 | INCONCLUSIVE | INCONCLUSIVE | 0.8976 | 0.9015 | no | -0.6 | k=2: p near 0.20/2 |
| realized_vol | 2005_2009 | 0.9720 | 0.9649 | FAIL (reversed) | FAIL (reversed) | 0.0285 | 0.0351 | no | -1.8 | k=2: reverse p near 0.05/2 |
| realized_vol | 2015_2019 | 0.9570 | 0.9554 | FAIL (reversed) | FAIL (reversed) | 0.0435 | 0.0446 | no | -0.3 | k=1: reverse p near 0.05 |
| st_reversal | 2005_2009 | 0.2104 | 0.2147 | FAIL | FAIL | 0.7901 | 0.7854 | no | +0.4 | k=1: p near 0.20 |
| st_reversal | 2015_2019 | 0.1999 | 0.1762 | INCONCLUSIVE | INCONCLUSIVE | 0.8006 | 0.8238 | no | -2.5 | k=1: p near 0.20 |
| st_reversal | 2020_2024 | 0.0580 | 0.0645 | INCONCLUSIVE | INCONCLUSIVE | 0.9425 | 0.9355 | no | +1.2 | k=1: p near 0.05 |

z = (p_20k - p_2k) / combined Monte Carlo SE; |z| > 2 is unusual for two independent estimates of the same p.

## Verdict changes at k=2..5 only (not at k=1)

| factor | period | k | adj_p_2k | adj_p_20k | adj_rev_p_2k | adj_rev_p_20k | verdict_2k | verdict_20k |
|---|---|---|---|---|---|---|---|---|
| price_to_ma | 2020_2024 | 2 | 0.2059 | 0.1972 | 1.0000 | 1.0000 | FAIL | INCONCLUSIVE |
