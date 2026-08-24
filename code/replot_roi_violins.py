"""Re-render the three-group ROI AUC violin plots into V5.

Drops the word 'extended' from the displayed title (uses roi_label='ROI'),
and emits both the normalized and the raw versions of the violin.

Files written into results/group_comparison_results/three_groups_V5/:
  group_comparison_violin_ROI_auc.png             (raw)
  group_comparison_violin_ROI_normalized_auc.png  (normalized)

Stats text reports written next to them.
"""
from contextlib import redirect_stdout
from pathlib import Path

from utils.config import BASE_DIR, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects

from step6_groups_comparison import (
    filter_subjects_by_detection_rate,
    load_and_process_roi_data,
    load_and_process_roi_data_normalized,
    plot_group_comparison,
    run_three_group_tests,
)


def _resolve_subjects(group_dirname: str, results_dir: Path):
    subs = get_all_subjects(f"{BASE_DIR}/{group_dirname}/")
    subs = [s for s in subs if (results_dir / s).exists() and s != "dashboards"]
    subs, _ = filter_subjects_by_detection_rate(subs, dir_path=results_dir)
    return subs


def main() -> None:
    young_dir = Path("results/new_iso_results")
    elderly_dir = Path("results/new_elderly_results")
    mci_dir = Path("results/new_MCI_results")

    young = _resolve_subjects("control_clean", young_dir)
    elderly = _resolve_subjects("elderly_control_clean", elderly_dir)
    mci = _resolve_subjects("MCI_clean", mci_dir)

    output_dir = Path("results/group_comparison_results/three_groups_V5")
    output_dir.mkdir(exist_ok=True, parents=True)

    auc_metric = {"auc": {"name": "Area Under Curve", "unit": "AU"}}
    roi_channels = EXTENDED_CENTRAL_PARIETAL_ROI
    roi_label = "ROI"

    groups_dict = {
        "Young":   (young,   young_dir),
        "Elderly": (elderly, elderly_dir),
        "MCI":     (mci,     mci_dir),
    }

    for normalize in (False, True):
        suffix = "normalized" if normalize else "raw"
        loader = load_and_process_roi_data_normalized if normalize else load_and_process_roi_data

        roi_group_data = {
            name: {"data": loader(subs, dpath, roi_channels=roi_channels)[0],
                   "n_subjects": len(subs)}
            for name, (subs, dpath) in groups_dict.items()
        }

        stats_path = output_dir / f"three_group_statistics_{roi_label}_{suffix}_auc.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                roi_results = run_three_group_tests(roi_group_data, auc_metric)
        print(f"Stats: {stats_path}")

        plot_group_comparison(
            groups_dict, output_dir=output_dir,
            test_results=roi_results["posthoc"],
            normalize=normalize, roi_only=True,
            roi_channels=roi_channels, roi_label=roi_label,
            metrics_filter=["auc"],
        )


if __name__ == "__main__":
    main()
