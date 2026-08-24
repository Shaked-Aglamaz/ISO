"""Plot successful-ISFS channels as green dots on an empty EGI-256 topography
for the three low-detection excluded young controls (EL3017, EL3018, EL3021).

Uses the SAME topo conventions as the thesis figures (step4):
  - face/neck/ear electrodes excluded (utils.config lists)
  - sphere='auto', extrapolate='head', then clip_topo_to_head
so the dots project to identical positions as every other topo in the thesis.

Output: debug/tmp/{subject}_isfs_channels.png
"""
import sys
from pathlib import Path

# code/utils/ is auto-added to sys.path[0] and contains utils.py, which shadows
# the `utils` package. Drop it and add code/ so `utils.config` resolves correctly.
_HERE = Path(__file__).resolve().parent          # code/utils
_CODE = _HERE.parent                             # code
sys.path = [p for p in sys.path if Path(p).resolve() != _HERE]
sys.path.insert(0, str(_CODE))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from utils.config import FACE_ELECTRODES, NECK_ELECTRODES, EAR_ELECTRODES

# Thesis topo conventions (see step4_distribution_analysis.py).
TOPO_SPHERE = "auto"
TOPO_EXTRAPOLATE = "head"


def clip_topo_to_head(ax, info, sphere=TOPO_SPHERE):
    """Clip the colored field + contours to the head circle (leave dots intact).
    Identical to step4_distribution_analysis.clip_topo_to_head."""
    from matplotlib.patches import Circle
    from matplotlib.collections import PathCollection
    from mne.viz.topomap import _check_sphere
    s = np.asarray(_check_sphere(sphere, info), dtype=float)
    cx, cy, r = float(s[0]), float(s[1]), float(s[-1])
    clip = Circle((cx, cy), r, transform=ax.transData)
    for im in ax.images:
        im.set_clip_path(clip)
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            coll.set_clip_path(clip)
    return clip


EXCLUDED = set(FACE_ELECTRODES) | set(NECK_ELECTRODES) | set(EAR_ELECTRODES)

RESULTS_DIR = Path("I:/Shaked/ISO/results/new_iso_results/a_excluded_V3")
OUT_DIR = Path("I:/Shaked/ISO/debug/tmp")
SUBJECTS = ["EL3017", "EL3018", "EL3021"]


def load_channels(subject):
    """Return (all_scalp_channels, successful_channels) excluding face/neck/ear."""
    csv = RESULTS_DIR / subject / f"{subject}_all_channels_summary.csv"
    df = pd.read_csv(csv, comment="#")
    df = df[~df["channel"].isin(EXCLUDED)]
    all_ch = df["channel"].tolist()
    good = df.loc[df["peak_frequency"].notna(), "channel"].tolist()
    return all_ch, good


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    montage = mne.channels.make_standard_montage("EGI_256")

    for subject in SUBJECTS:
        all_ch, good = load_channels(subject)
        # Keep only channels with known montage positions, same as step4.
        ch_names = [ch for ch in all_ch if ch in montage.ch_names]
        good = [ch for ch in good if ch in ch_names]

        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
        info.set_montage(montage)

        fig, ax = plt.subplots(figsize=(6, 6))
        data = np.zeros(len(ch_names))  # uniform/empty field (white)
        mask = np.array([ch in good for ch in ch_names])

        mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            sphere=TOPO_SPHERE, extrapolate=TOPO_EXTRAPOLATE,
            cmap="Greys", vlim=(0, 1), contours=0,
            sensors="k.",
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="limegreen",
                             markeredgecolor="green", markersize=9, linewidth=0),
        )
        clip_topo_to_head(ax, info)
        ax.set_title(f"{subject}: {len(good)} ISFS channels (green)", fontsize=12)
        out = OUT_DIR / f"{subject}_isfs_channels.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"{subject}: {len(good)} ISFS channels of {len(ch_names)} scalp "
              f"channels -> {out}")


if __name__ == "__main__":
    main()
