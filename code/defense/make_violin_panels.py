"""Re-emit the group-comparison violins as one panel per metric, for the defense deck.

Same inputs, same plotting function and same statistics as the manuscript figures
(F3 / F5): `code/replot_f3_no_title.py` and `code/replot_f5_no_title.py` are the
templates. The only difference is `metrics_filter`, which makes
`plot_group_comparison` emit a single wide panel per metric instead of the
portrait three-metric column that fits a Doc page, so each result gets its own
slide. No statistic is recomputed for reporting: the omnibus/post-hoc tests are
re-run only to drive the significance brackets, exactly as the replot scripts do,
and no stats file is written.

Outputs (results/defense_slides_V1/):
    group_comparison_violin_peak_frequency.png
    group_comparison_violin_bandwidth.png
    group_comparison_violin_auc.png
    group_comparison_violin_extended_ROI_normalized_auc.png

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/defense/make_violin_panels.py
"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from utils.config import BASE_DIR, EXTENDED_CENTRAL_PARIETAL_ROI  # noqa: E402
from utils.utils import get_all_subjects  # noqa: E402

from step6_groups_comparison import (  # noqa: E402
    filter_subjects_by_detection_rate,
    load_and_process_all_channels_data,
    load_and_process_roi_data_normalized,
    plot_group_comparison,
    run_three_group_tests,
)

OUT_DIR = Path("results/defense_slides_V1")

GROUPS = {
    "Young": ("control_clean", Path("results/sigma_fix_YA")),
    "Elderly": ("elderly_control_clean", Path("results/sigma_fix_HE")),
    "MCI": ("MCI_clean", Path("results/sigma_fix_MCI")),
}

METRICS = {
    "peak_frequency": {"name": "Peak Frequency", "unit": "Hz"},
    "bandwidth": {"name": "Bandwidth", "unit": "Hz"},
    "auc": {"name": "Area Under Curve", "unit": "AU"},
}


def _subjects(group_dirname: str, results_dir: Path):
    subs = get_all_subjects(f"{BASE_DIR}/{group_dirname}/")
    subs = [s for s in subs if (results_dir / s).exists() and s != "dashboards"]
    subs, _ = filter_subjects_by_detection_rate(subs, dir_path=results_dir)
    return subs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups_dict = {}
    for name, (dirname, results_dir) in GROUPS.items():
        subs = _subjects(dirname, results_dir)
        groups_dict[name] = (subs, results_dir)
        print(f"N: {name}={len(subs)}")

    # Whole-scalp tests, only to drive the post-hoc brackets (as in replot_f3).
    group_data = {
        name: {"data": load_and_process_all_channels_data(subs, d)[0],
               "n_subjects": len(subs)}
        for name, (subs, d) in groups_dict.items()
    }
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        results = run_three_group_tests(group_data, METRICS)

    for metric in METRICS:
        plot_group_comparison(
            groups_dict, output_dir=OUT_DIR,
            test_results=results["posthoc"], show_title=False, paper_style=True,
            metrics_filter=[metric],
        )

    # F5 equivalent: normalized AUC inside the pre-defined central-parietal ROI.
    # Mirrors replot_f5_no_title.py, including re-running the ROI test purely to
    # drive the (absent, because ns) brackets.
    roi_channels = EXTENDED_CENTRAL_PARIETAL_ROI
    roi_group_data = {
        name: {
            "data": load_and_process_roi_data_normalized(subs, d, roi_channels=roi_channels)[0],
            "n_subjects": len(subs),
        }
        for name, (subs, d) in groups_dict.items()
    }
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        roi_results = run_three_group_tests(roi_group_data, {"auc": METRICS["auc"]})

    plot_group_comparison(
        groups_dict, output_dir=OUT_DIR,
        test_results=roi_results["posthoc"], show_title=False, paper_style=True,
        normalize=True, roi_only=True,
        roi_channels=roi_channels, roi_label="extended_ROI",
        metrics_filter=["auc"],
    )


if __name__ == "__main__":
    main()
