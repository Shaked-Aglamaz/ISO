"""Quick runner to generate just the ROI topo and ROI violin plots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import BASE_DIR, CENTRAL_PARIETAL_ROI, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects
from step4_distribution_analysis import filter_subjects_by_detection_rate
from step5_topo_comparison import prepare_three_group_data, plot_three_group_topos_roi
from step6_groups_comparison import (
    plot_group_comparison, load_and_process_roi_data, run_three_group_tests,
)

# --- Setup subjects ---
young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
young_dir = Path("results/new_iso_results")
young_subjects = [s for s in young_subjects if (young_dir / s).exists() and s != "dashboards"]
young_subjects, _ = filter_subjects_by_detection_rate(young_subjects, dir_path=young_dir)

elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
elderly_dir = Path("results/new_elderly_results")
elderly_subjects = [s for s in elderly_subjects if (elderly_dir / s).exists() and s != "dashboards"]
elderly_subjects, _ = filter_subjects_by_detection_rate(elderly_subjects, dir_path=elderly_dir)

mci_subjects = get_all_subjects(f"{BASE_DIR}/MCI_clean/")
mci_dir = Path("results/new_MCI_results")
mci_subjects = [s for s in mci_subjects if (mci_dir / s).exists() and s != "dashboards"]
mci_subjects, _ = filter_subjects_by_detection_rate(mci_subjects, dir_path=mci_dir)

output_dir = Path("results/group_comparison_results/three_groups_V2")
output_dir.mkdir(exist_ok=True, parents=True)

# --- AUC-only violin plots for ROI and extended_ROI ---
metrics = {'auc': {'name': 'Area Under Curve', 'unit': 'AU'}}
groups_dict = {
    'Young': (young_subjects, young_dir),
    'Elderly': (elderly_subjects, elderly_dir),
    'MCI': (mci_subjects, mci_dir),
}

roi_variants = [
    ('ROI', CENTRAL_PARIETAL_ROI),
    ('extended_ROI', EXTENDED_CENTRAL_PARIETAL_ROI),
]

for label, channels in roi_variants:
    print(f"\n=== {label}: computing stats ===")
    g_data = {
        'Young': {
            'data': load_and_process_roi_data(young_subjects, young_dir, roi_channels=channels)[0],
            'n_subjects': len(young_subjects),
        },
        'Elderly': {
            'data': load_and_process_roi_data(elderly_subjects, elderly_dir, roi_channels=channels)[0],
            'n_subjects': len(elderly_subjects),
        },
        'MCI': {
            'data': load_and_process_roi_data(mci_subjects, mci_dir, roi_channels=channels)[0],
            'n_subjects': len(mci_subjects),
        },
    }
    stats = run_three_group_tests(g_data, metrics)

    print(f"\n=== {label}: plotting AUC violin ===")
    plot_group_comparison(
        groups_dict, output_dir=output_dir,
        test_results=stats['posthoc'],
        roi_only=True, roi_channels=channels, roi_label=label,
        metrics_filter=['auc'],
    )

print("\nDone!")
