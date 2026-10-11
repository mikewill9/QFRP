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
- app/api.py, including reversed-verdict logic and roll_spread exclusion
- CONTRACT.md
- Shipping precomputed results
- High-precision rerun tooling: --config flag in run_permutations.py, compare_permutations.py, tests/test_run_permutations_cli.py
- PROJECT_HANDOFF.md update (post-Checkpoint 11 section)


## Drafted with Claude (chat), edited and approved by Michael
- Verdict text, reversed-verdict text, survivorship note
- LIMITATIONS.md
- Planning and teamwork messages and outlines
- docs/amihud_survivorship.md: outline and some wording drafted with Claude chat; data checks run, claims verified, and final text written by Michael

## Methodology decisions (made by Michael after reviewing options with Claude)
- Zero-cost significance test, Bonferroni correction, verdict cutoffs, dropping roll_spread, featuring the survivorship result