"""
One-off reporting helper for the Core-vs-Extended ROI decision.

Prints (and saves) N / Mean / Median / SD of the AUC metric for each of the
four variants — Raw Core, Raw Extended, Normalized Core, Normalized Extended
— across the three groups (Young, Elderly, MCI), plus pairwise p-values
for all three pairs.

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/roi_decision_stats.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, ttest_ind

from utils.config import BASE_DIR, CENTRAL_PARIETAL_ROI, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects

from step4_distribution_analysis import filter_subjects_by_detection_rate
from step6_groups_comparison import (
    load_and_process_roi_data,
    load_and_process_roi_data_normalized,
)


METRIC = "auc"

GROUPS = [
    ("Young",   "control_clean",         Path("results/new_iso_results")),
    ("Elderly", "elderly_control_clean", Path("results/new_elderly_results")),
    ("MCI",     "MCI_clean",             Path("results/new_MCI_results")),
]


def load_subjects_for_group(base_subdir: str, results_dir: Path) -> list[str]:
    subjects = get_all_subjects(f"{BASE_DIR}/{base_subdir}/")
    subjects = [s for s in subjects if (results_dir / s).exists() and s != "dashboards"]
    subjects, _ = filter_subjects_by_detection_rate(subjects, dir_path=results_dir)
    return subjects


def descriptive(values: pd.Series) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
    }


def _is_normal(values: pd.Series) -> bool:
    return len(values) >= 3 and shapiro(values).pvalue > 0.05


def pairwise_p(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    if _is_normal(a) and _is_normal(b):
        _, p = ttest_ind(a, b, equal_var=False)
        return float(p), "Welch-t"
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p), "MWU"


def omnibus_p(groups: list[pd.Series]) -> tuple[float, str]:
    if all(_is_normal(g) for g in groups):
        _, p = f_oneway(*groups)
        return float(p), "ANOVA"
    _, p = kruskal(*groups)
    return float(p), "KW"


def compute_variant(
    group_subjects: dict[str, tuple[list[str], Path]],
    *,
    normalize: bool,
    roi_channels: list[str],
) -> dict[str, pd.Series]:
    loader = (
        load_and_process_roi_data_normalized if normalize else load_and_process_roi_data
    )
    per_group: dict[str, pd.Series] = {}
    for gname, (subjects, results_dir) in group_subjects.items():
        df, _, _ = loader(subjects, results_dir, roi_channels=roi_channels)
        per_group[gname] = (
            pd.Series(dtype=float) if df is None else df[METRIC].dropna().reset_index(drop=True)
        )
    return per_group


def format_block(title: str, per_group: dict[str, pd.Series]) -> str:
    names = list(per_group.keys())
    lines = [title, "-" * len(title)]
    lines.append(f"{'Group':<9} {'N':>3}  {'Mean':>9}  {'Median':>9}  {'SD (σ)':>9}")
    for g in names:
        d = descriptive(per_group[g])
        lines.append(
            f"{g:<9} {d['n']:>3}  {d['mean']:>9.4f}  {d['median']:>9.4f}  {d['sd']:>9.4f}"
        )

    om_p, om_test = omnibus_p([per_group[g] for g in names])
    lines.append("")
    lines.append(f"Omnibus ({om_test}):              p = {om_p:.4f}")

    for a, b in combinations(names, 2):
        p, test = pairwise_p(per_group[a], per_group[b])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"{a:<8} vs {b:<8} ({test:<7}) p = {p:.4f}  [{stars}]")
    return "\n".join(lines)


def main() -> None:
    print("Loading subjects per group…")
    group_subjects: dict[str, tuple[list[str], Path]] = {}
    for gname, base_subdir, results_dir in GROUPS:
        subs = load_subjects_for_group(base_subdir, results_dir)
        group_subjects[gname] = (subs, results_dir)
        print(f"  {gname}: {len(subs)} subjects ({results_dir})")

    variants = [
        ("RAW  | Core ROI (20-ch)",
         dict(normalize=False, roi_channels=CENTRAL_PARIETAL_ROI)),
        ("RAW  | Extended ROI (36-ch)",
         dict(normalize=False, roi_channels=EXTENDED_CENTRAL_PARIETAL_ROI)),
        ("NORM | Core ROI (20-ch)",
         dict(normalize=True,  roi_channels=CENTRAL_PARIETAL_ROI)),
        ("NORM | Extended ROI (36-ch)",
         dict(normalize=True,  roi_channels=EXTENDED_CENTRAL_PARIETAL_ROI)),
    ]

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("AUC — ROI vs EXTENDED ROI DECISION TABLE")
    lines.append("Raw   = per-subject ROI mean of AUC (µV²·Hz or whatever AUC units are).")
    lines.append("Norm  = each subject's channel values / scalp mean (3-σ-trimmed),")
    lines.append("        then averaged over ROI channels. Values > 1 ⇒ ROI > scalp.")
    lines.append("Pairwise test auto-picked: Shapiro-Wilk per group → Welch-t or MWU.")
    lines.append("=" * 72)
    lines.append("")

    for title, kwargs in variants:
        per_group = compute_variant(group_subjects, **kwargs)
        lines.append(format_block(title, per_group))
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path("results/group_comparison_results/three_groups/roi_vs_extended_decision_table.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
