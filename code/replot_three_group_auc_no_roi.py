"""Re-render the three-group AUC topo without the green ROI dots.

Runs the same AUC-only stats path as main_three_group() in step5_topo_comparison
(the AUC metric only — peak_freq/bandwidth are skipped to save time), then calls
plot_three_group_topos with show_roi=False so the resulting file is named
three_group_topo_auc_no_roi.png and lands in the V5 output dir.
"""
from pathlib import Path

from utils.config import BASE_DIR
from utils.utils import get_all_subjects

from step5_topo_comparison import (
    prepare_three_group_data,
    electrode_wise_anova,
    cluster_permutation_anova,
    extract_significant_clusters_f,
    posthoc_tukey_at_clusters,
    plot_three_group_topos,
)


def main() -> None:
    young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    young_dir = Path("results/new_iso_results")
    young_subjects = [s for s in young_subjects
                      if (young_dir / s).exists() and s != "dashboards"]

    elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
    elderly_dir = Path("results/new_elderly_results")
    elderly_subjects = [s for s in elderly_subjects
                        if (elderly_dir / s).exists() and s != "dashboards"]

    mci_subjects = get_all_subjects(f"{BASE_DIR}/MCI_clean/")
    mci_dir = Path("results/new_MCI_results")
    mci_subjects = [s for s in mci_subjects
                    if (mci_dir / s).exists() and s != "dashboards"]

    output_dir = Path("results/group_comparison_results/three_groups_V5")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Young={len(young_subjects)}  Elderly={len(elderly_subjects)}  "
          f"MCI={len(mci_subjects)}")
    print(f"Output: {output_dir}")

    for metric in ('auc', 'peak_frequency', 'bandwidth'):
        print(f"\n=== METRIC: {metric} ===")
        print(f"Preparing {metric} normalized group evokeds...")
        group_evokeds_list, group_names, adjacency, available_channels, _ = \
            prepare_three_group_data(
                young_subjects, elderly_subjects, mci_subjects,
                metric, young_dir, elderly_dir, mci_dir,
                normalize=True,
            )

        print("Electrode-wise ANOVA...")
        electrode_wise_anova(group_evokeds_list, group_names, available_channels)

        print("Cluster permutation ANOVA (5000 perms)...")
        F_obs, clusters, cluster_pv, _ = cluster_permutation_anova(
            group_evokeds_list, adjacency,
            n_permutations=5000, threshold_p=0.05, n_jobs=-1,
        )
        _, sig_channel_indices = extract_significant_clusters_f(
            F_obs, clusters, cluster_pv, available_channels
        )
        posthoc_results = posthoc_tukey_at_clusters(
            group_evokeds_list, group_names, sig_channel_indices, available_channels
        )

        info = group_evokeds_list[0][0].info
        print(f"Plotting {metric} (mean, no ROI)...")
        plot_three_group_topos(
            group_evokeds_list, group_names, metric, info,
            posthoc_results, F_obs, sig_channel_indices, output_dir,
            clusters=clusters, cluster_pv=cluster_pv,
            agg='mean', show_roi=False,
        )
    print("Done.")


if __name__ == "__main__":
    main()
