# AI Usage

## How we used AI
Michael Williams:
Michael wrote the core statistical functions by hand: the within-date shuffle, the permutation p-value, and the Bonferroni adjustment, plus their tests. Claude Code wrote larger infrastructure: the backtest pipeline module, the permutation runner, and the API layer. Michael directed each task, reviewed every change before committing, and rejected or corrected approaches when needed (for example, catching that the original shuffle test was biased by trading costs). Michael also used Claude in chat for design review, research on significance thresholds, and code skeletons he completed himself.

## Hand-written (Michael)
- permute_within_date + tests
- permutation_pvalue + tests
- bonferroni_adjust + tests

## AI-assisted (Claude Code, reviewed by Michael)
- Backtest pipeline extraction (pipeline.py)
- Permutation runner (run_permutations.py)
- Data extension to 2005, EA removal
- app/api.py

## Frontend (BP)
- Hand-written: [BP fills in]
- AI-assisted: Analyzer page

## Methodology decisions (ours, not AI's)
 zero-cost significance test, Bonferroni, verdict cutoffs