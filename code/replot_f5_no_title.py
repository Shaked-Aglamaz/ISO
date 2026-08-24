"""Regenerate F5 — the extended-ROI normalized AUC violin — into three_groups_V10
WITHOUT the figure title (suptitle), overriding the existing V10 PNG.

Replicates the ROI portion of step6_groups_comparison.run_three_group_comparison
exactly (sigma_fix_* inputs, final cohort, extended ROI, normalized, AUC only),
but passes show_title=False so the plot has no title. Only the figure is written;
the existing stats text file is left untouched.

Run from repo root:  python code/replot_f5_no_title.py
"""
import os
from contextlib import redirect_stdout
from pathlib import Path

from utils.config import BASE_DIR, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects

from step6_groups_comparison import (
    filter_subjects_by_detection_rate,
    load_and_process_roi_data_normalized,
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
    roi_channels = EXTENDED_CENTRAL_PARIETAL_ROI
    roi_label = "extended_ROI"
    auc_metric = {"auc": {"name": "Area Under Curve", "unit": "AU"}}

    groups_dict = {
        "Young": (young, young_dir),
        "Elderly": (elderly, elderly_dir),
        "MCI": (mci, mci_dir),
    }

    roi_group_data = {
        name: {
            "data": load_and_process_roi_data_normalized(subs, dpath, roi_channels=roi_channels)[0],
            "n_subjects": len(subs),
        }
        for name, (subs, dpath) in groups_dict.items()
    }

    # Re-run the stats only to drive the (absent) significance brackets; do not
    # overwrite the existing stats .txt.
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        roi_results = run_three_group_tests(roi_group_data, auc_metric)

    plot_group_comparison(
        groups_dict, output_dir=output_dir,
        test_results=roi_results["posthoc"],
        normalize=True, roi_only=True,
        roi_channels=roi_channels, roi_label=roi_label,
        metrics_filter=["auc"], show_title=False, paper_style=True,
    )


if __name__ == "__main__":
    main()
