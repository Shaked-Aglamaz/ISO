"""Regenerate the F4 / S1 topography panels into three_groups_V11 with the
paper font sizes and spelled-out colorbar legends.

The cluster-permutation test is NOT re-run. The post-hoc electrode sets from the
V10 run are read back out of

    three_groups_V10/three_group_topo_statistics.txt

and handed to the plotting function, which already takes them as an argument.
Only the displayed group topographies are recomputed, and those are plain means
over the same per-subject files, so every number stays identical to V10.

Emits into three_groups_V11:
    three_group_topo_auc.png            (normalized + cluster overlay)  -> F4 B
    three_group_topo_auc_raw.png        (raw)                           -> F4 A
    three_group_topo_peak_frequency.png                                 -> S1 A
    three_group_topo_bandwidth.png                                      -> S1 B

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/replot_topos_paper.py
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import BASE_DIR
from utils.utils import get_all_subjects
from utils.topo_aggregation import build_raw_group_topo

from step5_topo_comparison import (
    filter_subjects_by_detection_rate,
    load_all_subjects_data,
    plot_three_group_raw_topos,
    plot_three_group_topos,
    prepare_three_group_data,
)

V10_DIR = Path("results/group_comparison_results/three_groups_V10")
OUT_DIR = Path("results/group_comparison_results/three_groups_V11")
STATS_FILE = V10_DIR / "three_group_topo_statistics.txt"

METRICS = ["peak_frequency", "bandwidth", "auc"]
PAIRS = [("Young", "Elderly"), ("Young", "MCI"), ("Elderly", "MCI")]


def parse_v10_posthoc(stats_path: Path) -> dict[str, dict[tuple[str, str], list[str]]]:
    """Read the per-metric post-hoc electrode names out of the V10 stats report.

    Returns {metric: {(g1, g2): [electrode names]}}. Parsed from each metric's
    SUMMARY block, which lists one line per pair; the electrode names come from
    the preceding POST-HOC section (absent when a pair has zero electrodes).
    """
    text = stats_path.read_text(encoding="utf-8")

    # Split into per-metric sections on the "METRIC: X" banners.
    sections: dict[str, str] = {}
    parts = re.split(r"#+\n\s+METRIC:\s+(\w+)\n#+", text)
    for i in range(1, len(parts), 2):
        sections[parts[i].strip().lower()] = parts[i + 1]

    out: dict[str, dict[tuple[str, str], list[str]]] = {}
    for metric in METRICS:
        body = sections.get(metric, "")
        pair_channels: dict[tuple[str, str], list[str]] = {p: [] for p in PAIRS}
        for g1, g2 in PAIRS:
            # "  Young vs Elderly: 5 significant electrodes\n    Electrodes: [...]"
            m = re.search(
                rf"^\s*{g1} vs {g2}: (\d+) significant electrodes"
                rf"(?:\n\s*Electrodes: \[(.*?)\])?",
                body, re.MULTILINE,
            )
            if not m:
                continue
            n_expected = int(m.group(1))
            names = re.findall(r"'([^']+)'", m.group(2) or "")
            if n_expected and len(names) != n_expected:
                raise ValueError(
                    f"{metric} {g1} vs {g2}: report says {n_expected} electrodes "
                    f"but {len(names)} names were parsed"
                )
            pair_channels[(g1, g2)] = names
        out[metric] = pair_channels
    return out


def to_indices(pair_channels, available_channels):
    """Map electrode names to positional indices in ``available_channels``."""
    lookup = {ch: i for i, ch in enumerate(available_channels)}
    result = {}
    for pair, names in pair_channels.items():
        missing = [n for n in names if n not in lookup]
        if missing:
            raise ValueError(f"electrodes not in the montage: {missing}")
        result[pair] = np.array(sorted(lookup[n] for n in names), dtype=int)
    return result


def _subjects(group_dirname: str, results_dir: Path):
    subs = get_all_subjects(f"{BASE_DIR}/{group_dirname}/")
    return [s for s in subs if (results_dir / s).exists() and s != "dashboards"]


def main() -> None:
    young_dir = Path("results/sigma_fix_YA")
    elderly_dir = Path("results/sigma_fix_HE")
    mci_dir = Path("results/sigma_fix_MCI")

    young = _subjects("control_clean", young_dir)
    elderly = _subjects("elderly_control_clean", elderly_dir)
    mci = _subjects("MCI_clean", mci_dir)

    posthoc_v10 = parse_v10_posthoc(STATS_FILE)
    print(f"Post-hoc electrode sets reused from {STATS_FILE}:")
    for metric in METRICS:
        counts = ", ".join(f"{g1} vs {g2}: {len(chs)}"
                           for (g1, g2), chs in posthoc_v10[metric].items())
        print(f"  {metric:15s} {counts}")
    print("No permutation test is run.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- normalized topos (F4 B, S1 A, S1 B) ---
    for metric in METRICS:
        print(f"Preparing {metric} (normalized)...")
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
            group_evokeds_list, group_names, _, available_channels, _ = \
                prepare_three_group_data(
                    young, elderly, mci, metric,
                    young_dir, elderly_dir, mci_dir, normalize=True,
                )
        print("  N: " + ", ".join(f"{n}={len(e)}"
                                  for n, e in zip(group_names, group_evokeds_list)))

        posthoc = to_indices(posthoc_v10[metric], available_channels)
        sig_idx = np.array(sorted({i for arr in posthoc.values() for i in arr}), dtype=int)

        info = group_evokeds_list[0][0].info
        plot_three_group_topos(
            group_evokeds_list, group_names, metric, info,
            posthoc, None, sig_idx, OUT_DIR,
            clusters=None, cluster_pv=None, agg="mean",
            normalize=True, fstat_fig=False,
        )

    # --- raw AUC topo (F4 A) ---
    print("\nPreparing raw AUC topographies via utils.topo_aggregation...")
    raw_topos, raw_means, raw_n = [], [], []
    raw_info = None
    for name, subjects, dir_path in [("Young", young, young_dir),
                                     ("Elderly", elderly, elderly_dir),
                                     ("MCI", mci, mci_dir)]:
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
            filt, _ = filter_subjects_by_detection_rate(
                subjects, min_detection_rate=0.2, dir_path=dir_path)
            subs_data = load_all_subjects_data(filt, dir_path)
            topo, m, _, info_ = build_raw_group_topo(subs_data, "auc")
        raw_topos.append(topo)
        raw_means.append(m)
        raw_n.append(len(subs_data))
        raw_info = info_
        print(f"  {name}: N={len(subs_data)}, displayed_mean={m:.4f}")

    plot_three_group_raw_topos(
        raw_topos, raw_means, ["Young", "Elderly", "MCI"],
        raw_n, raw_info, "auc", OUT_DIR,
    )

    print(f"\nDone. Panels in {OUT_DIR}")


if __name__ == "__main__":
    main()
