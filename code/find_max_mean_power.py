"""Find max/min of the baseline-corrected mean spectrum across all young-adult channels."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/new_iso_results")

EXCLUDE_DIRS = {"a_excluded", "a_group_plots_V1", "a_group_plots_V2", "sigma_boxplot"}

subject_dirs = [
    p for p in ROOT.iterdir()
    if p.is_dir() and p.name not in EXCLUDE_DIRS
]

global_max = -np.inf
global_min = np.inf
top_records = []

for sub_dir in subject_dirs:
    sub = sub_dir.name
    sub_max = -np.inf
    for ch_dir in sub_dir.iterdir():
        if not ch_dir.is_dir() or not ch_dir.name.endswith("_output"):
            continue
        csvs = list(ch_dir.glob("*_spectral_power.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        freq = df["frequency"].to_numpy()
        mp = df["mean_power"].to_numpy()
        baseline_mask = (freq > 0.06) & (freq < 0.102)
        if not baseline_mask.any():
            continue
        shift = np.nanmean(mp[baseline_mask])
        shifted = mp - shift
        ch_max = np.nanmax(shifted)
        ch_min = np.nanmin(shifted)
        ch_name = ch_dir.name.replace(f"{sub}_", "").replace("_output", "")
        top_records.append((ch_max, sub, ch_name))
        if ch_max > sub_max:
            sub_max = ch_max
        if ch_max > global_max:
            global_max = ch_max
        if ch_min < global_min:
            global_min = ch_min
    print(f"{sub:>10s}  max={sub_max:.3f}")

print()
print(f"GLOBAL max = {global_max:.4f}")
print(f"GLOBAL min = {global_min:.4f}")
print()
top_records.sort(reverse=True)
print("Top 15 highest-peak channels:")
for v, sub, ch in top_records[:15]:
    print(f"  {v:.3f}   {sub} {ch}")
