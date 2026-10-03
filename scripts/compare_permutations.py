"""Compare a high-N permutation rerun against the shipped N=2000 results.

For every JSON in the rerun directory (default results/permutations_hp/),
loads the same (factor, period) from the baseline directory (default
results/permutations/) and prints one row: raw p-values, k=1 verdicts,
reverse p-values, whether the k=1 verdict flipped, and why the config was
selected for a rerun (SELECTED_FOR). A second table lists configs whose
verdict changes at some k in 2..K_MAX but NOT at k=1.

Verdicts come from app.api (get_analysis + verdict_for), switched between
directories with its QFRP_RESULTS_DIR override, so the cutoffs, the
Bonferroni adjustment and the reversed ("worse than random") rule are the
app's own -- nothing is re-implemented here.

The two runs use different base seeds, so their p-values are independent
Monte Carlo estimates of the same true p. Each row also checks that the
real gross Sharpe is identical in both files: if it isn't, the backtest
code changed between runs and the p-values are not comparable.

    python scripts/compare_permutations.py [--out results/permutations_hp/comparison.md]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))  # so `app` imports when run as a script

from app.api import AnalysisResult, Verdict, get_analysis, verdict_for  # noqa: E402

K_MAX = 5

# Why each config was rerun, decided from the N=2000 results before any rerun.
# k=1 bands are about +-3 Monte Carlo SE (N=2000) of the 0.05 / 0.20 cutoffs;
# k=2..5 picks are configs whose p was within 3 SE of 0.05/k or 0.20/k.
SELECTED_FOR: dict[tuple[str, str], str] = {
    ("st_reversal", "2005_2009"): "k=1: p near 0.20",
    ("st_reversal", "2015_2019"): "k=1: p near 0.20",
    ("st_reversal", "2020_2024"): "k=1: p near 0.05",
    ("amihud_illiquidity", "2005_2009"): "k=1: PASS",
    ("amihud_illiquidity", "2005_2024"): "k=1: PASS",
    ("amihud_illiquidity", "2010_2014"): "k=1: PASS",
    ("amihud_illiquidity", "2015_2019"): "k=1: PASS",
    ("price_to_ma", "2015_2019"): "k=1: PASS",
    ("ewma_vol", "2005_2009"): "k=1: reverse p near 0.05",
    ("ewma_vol", "2010_2014"): "k=1: reverse p near 0.05",
    ("realized_vol", "2015_2019"): "k=1: reverse p near 0.05",
    ("dollar_volume", "2005_2009"): "k=4: reverse p near 0.05/4; k=5: reverse p near 0.05/5",
    ("dollar_volume", "2010_2014"): "k=5: reverse p near 0.05/5",
    ("price_to_ma", "2020_2024"): "k=2: p near 0.20/2",
    ("realized_vol", "2005_2009"): "k=2: reverse p near 0.05/2",
}


def load(results_dir: Path, factor: str, period: str) -> AnalysisResult:
    os.environ["QFRP_RESULTS_DIR"] = str(results_dir)
    return get_analysis(factor, period, k=1)


def label(v: Verdict) -> str:
    return f"{v.label} (reversed)" if v.reversed else v.label


def mc_se(p: float, n: int) -> float:
    """Monte Carlo standard error of a permutation p-value estimated from n shuffles."""
    return math.sqrt(p * (1 - p) / n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", type=Path, default=_REPO_ROOT / "results" / "permutations")
    parser.add_argument("--hp-dir", type=Path, default=_REPO_ROOT / "results" / "permutations_hp")
    parser.add_argument("--out", type=Path, help="also write the tables to this markdown file")
    args = parser.parse_args()

    paths = sorted(args.hp_dir.glob("*.json"))
    if not paths:
        sys.exit(f"no rerun JSONs in {args.hp_dir}")

    main_rows, k_rows, notes = [], [], []
    for path in paths:
        rec = json.loads(path.read_text())
        factor, period = rec["factor"], rec["period"]
        old = load(args.base_dir, factor, period)
        new = load(args.hp_dir, factor, period)
        if old.sharpe_gross != new.sharpe_gross:
            notes.append(
                f"{factor} {period}: real gross Sharpe differs "
                f"({old.sharpe_gross} vs {new.sharpe_gross}) -- not comparable"
            )

        v_old = {k: verdict_for(old, k) for k in range(1, K_MAX + 1)}
        v_new = {k: verdict_for(new, k) for k in range(1, K_MAX + 1)}
        flipped = label(v_old[1]) != label(v_new[1])
        # Distance between the two independent estimates in combined standard errors.
        se = math.hypot(
            mc_se(old.p_value, old.n_permutations), mc_se(new.p_value, new.n_permutations)
        )
        z = (new.p_value - old.p_value) / se if se > 0 else 0.0
        main_rows.append(
            f"| {factor} | {period} | {old.p_value:.4f} | {new.p_value:.4f} "
            f"| {label(v_old[1])} | {label(v_new[1])} "
            f"| {old.reverse_p_value:.4f} | {new.reverse_p_value:.4f} "
            f"| {'yes' if flipped else 'no'} | {z:+.1f} "
            f"| {SELECTED_FOR.get((factor, period), '-')} |"
        )
        if not flipped:
            for k in range(2, K_MAX + 1):
                a, b = v_old[k], v_new[k]
                if label(a) != label(b):
                    k_rows.append(
                        f"| {factor} | {period} | {k} | {a.adjusted_p_value:.4f} "
                        f"| {b.adjusted_p_value:.4f} | {a.adjusted_reverse_p_value:.4f} "
                        f"| {b.adjusted_reverse_p_value:.4f} | {label(a)} | {label(b)} |"
                    )

    n_old = old.n_permutations
    n_new = new.n_permutations
    lines = [
        f"## N={n_old:,} vs N={n_new:,} (verdicts at k=1)",
        "",
        "| factor | period | p_2k | p_20k | verdict_2k | verdict_20k "
        "| reverse_p_2k | reverse_p_20k | flipped | z | selected for |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *main_rows,
        "",
        "z = (p_20k - p_2k) / combined Monte Carlo SE; |z| > 2 is unusual for "
        "two independent estimates of the same p.",
        "",
        f"## Verdict changes at k=2..{K_MAX} only (not at k=1)",
        "",
    ]
    if k_rows:
        lines += [
            "| factor | period | k | adj_p_2k | adj_p_20k | adj_rev_p_2k | adj_rev_p_20k "
            "| verdict_2k | verdict_20k |",
            "|---|---|---|---|---|---|---|---|---|",
            *k_rows,
        ]
    else:
        lines.append("None.")
    if notes:
        lines += ["", "## Warnings", "", *(f"- {n}" for n in notes)]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
