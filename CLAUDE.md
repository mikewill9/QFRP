# QFRP Claude Code Instructions

## Project Overview

QFRP is a Python quantitative finance research framework.

The goal is to build a serious research environment for:
- financial data analysis
- factor research
- signal generation
- portfolio construction
- backtesting
- risk analysis

This is not intended to be a "money printing bot."
The goal is learning, experimentation, and building a defensible quantitative research workflow.

---

## Important Rules

The repository is the source of truth.

Before making changes:
1. Inspect the existing code.
2. Read relevant documentation.
3. Understand current architecture.
4. Avoid unnecessary rewrites.

Do not replace working systems with new designs unless there is a clear reason.

---

## Development Workflow

Always follow this process:

1. Inspect current repository state.
2. Make focused changes.
3. Add or update tests.
4. Run the test suite.
5. Review changes.
6. Commit a stable checkpoint.

Avoid creating huge unreviewed changes.

---

## Architecture Principles

Prioritize:

- correctness over complexity
- clear interfaces
- realistic assumptions
- testing
- documentation

Avoid:

- unnecessary abstraction
- premature optimization
- adding features without research justification

---

## Quantitative Finance Standards

Be careful about:

- lookahead bias
- survivorship bias
- overfitting
- unrealistic transaction assumptions
- data quality problems

Document assumptions and limitations.

A simpler correct model is better than a complicated unreliable one.

---

## Current Development Stage

The current focus is the Congressional App Challenge submission
(deadline Oct 26, 2026):

- the permutation significance test
- the app in `app/api.py` and `app/ui/`
- submission materials

Parked until after the submission:

- portfolio construction
- EDGAR sentiment
- the asset-growth factor

See PROJECT_HANDOFF.md for the details.

---

## User Context

The developer is a high school student learning:

- Python
- quantitative finance
- software engineering

Explain important technical decisions.

Do not assume advanced knowledge.

When introducing complex concepts:
- explain what it does
- explain why it exists
- explain tradeoffs

The goal is not only to produce code, but to help the developer understand the system.