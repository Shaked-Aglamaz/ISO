"""Regenerate F3 — the whole-scalp three-group parameter violins — into
three_groups_V10 WITHOUT the figure title (suptitle), overriding the V10 PNG.

Replicates the whole-scalp portion of
step6_groups_comparison.run_three_group_comparison exactly (sigma_fix_* inputs,
final cohort, all three metrics, post-hoc brackets), but passes show_title=False.
Only the figure is written; the existing stats text file is left untouched.

Run from repo root:  python code/replot_f3_no_title.py
"""
import os
from contextlib import redirect_stdout
from pathlib import Path

from utils.config import BASE_DIR
from utils.utils import get_all_subjects

from step6_groups_comparison import (
    filter_subjects_by_detection_rate,
    load_and_process_all_channels_data,
    plot_group_comparison,
    run_three_group_tests,
)


def _subjects(group_dirname: str, results_dir: Path):
    subs = get_all_subjects(f"{BASE_DIR}/{group_dirname}/")
    subs = [s for s in subs if (results_dir / s).exists() and s != "dashboards"]
    subs, _ = filter_subjects_by_detection_rate(subs, dir_path=results_dir)
    return subs


def main() -> None:
    young_dir = Path("results/sigma_fix_YA")
    elderly_dir = Path("results/sigma_fix_HE")
    mci_dir = Path("results/sigma_fix_MCI")

    young = _subjects("control_clean", young_dir)
    elderly = _subjects("elderly_control_clean", elderly_dir)
    mci = _subjects("MCI_clean", mci_dir)
    print(f"N: Young={len(young)}, Elderly={len(elderly)}, MCI={len(mci)}")

    output_dir = Path("results/group_comparison_results/three_groups_V11")
    metrics = {
        "peak_frequency": {"name": "Peak Frequency", "unit": "Hz"},
        "bandwidth": {"name": "Bandwidth", "unit": "Hz"},
        "auc": {"name": "Area Under Curve", "unit": "AU"},
    }

    group_data = {
        "Young": {"data": load_and_process_all_channels_data(young, young_dir)[0], "n_subjects": len(young)},
        "Elderly": {"data": load_and_process_all_channels_data(elderly, elderly_dir)[0], "n_subjects": len(elderly)},
        "MCI": {"data": load_and_process_all_channels_data(mci, mci_dir)[0], "n_subjects": len(mci)},
    }

    # Re-run stats to drive the post-hoc significance brackets; do not overwrite
    # the existing stats .txt.
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        results = run_three_group_tests(group_data, metrics)

    groups_dict = {
        "Young": (young, young_dir),
        "Elderly": (elderly, elderly_dir),
        "MCI": (mci, mci_dir),
    }
    plot_group_comparison(
        groups_dict, output_dir=output_dir,
        test_results=results["posthoc"], show_title=False, paper_style=True,
    )


if __name__ == "__main__":
    main()
