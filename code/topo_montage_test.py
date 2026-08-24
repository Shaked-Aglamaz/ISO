"""
Standalone diagnostic / fix-exploration script for the EGI-256 topomap projection.

Motivation
----------
In ``three_group_topo_auc_raw.png`` the posterior (occipital) electrodes appear
to sit too high on the scalp -- in what looks like the parietal region -- and the
bottom of the head outline looks "empty". The user hypothesised this comes from
excluding the neck electrodes.

Findings while writing this script (printed at runtime):
  * The production plotting code calls ``mne.viz.plot_topomap`` with NO sphere
    argument => ``sphere=None`` => resolves to the FIXED sphere (0, 0, 0, 0.095).
  * The 'auto'/fitted sphere is (0, 0.0089, 0.0418, 0.095) -- center shifted up
    and back. The fit is IDENTICAL whether or not the neck electrodes are present,
    so excluding the neck does NOT change the projection.
  * The real question is therefore: under each sphere, where does the posterior
    midline chain (... E101 -> E126, occipital) actually land relative to MNE's
    own head outline?

This script ONLY produces diagnostic figures + tries candidate spheres. It does
not touch step4/step5 plotting code.

Run from repo root with the venv active:
    python code/topo_montage_test.py
Figures are written to results/topo_montage_test/.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.viz.topomap import _check_sphere
from mne.channels.layout import _find_topomap_coords

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.config import FACE_ELECTRODES, NECK_ELECTRODES, EAR_ELECTRODES

OUT_DIR = os.path.join("results", "topo_montage_test")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDED = set(FACE_ELECTRODES) | set(NECK_ELECTRODES) | set(EAR_ELECTRODES)


def make_info(ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("EGI_256"))
    return info


def analysis_channels():
    m = mne.channels.make_standard_montage("EGI_256")
    return [c for c in m.ch_names if c not in EXCLUDED and c != "VREF"]


def posterior_midline_chain():
    """Electrodes on (or near) the posterior midline x~0, ordered top->bottom,
    among KEPT channels. These are the parietal->occipital landmarks."""
    m = mne.channels.make_standard_montage("EGI_256")
    pos = m.get_positions()["ch_pos"]
    kept = analysis_channels()
    cand = [c for c in kept if abs(pos[c][0]) < 0.012 and pos[c][1] < 0.005]
    cand.sort(key=lambda c: -pos[c][2])  # high z (top) first -> low z (occiput)
    return cand


# ---------------------------------------------------------------------------
CAT_COLORS = {"face": "tab:red", "neck": "tab:orange",
              "ear": "tab:green", "kept": "tab:blue"}


def category_of(ch):
    if ch in FACE_ELECTRODES:
        return "face"
    if ch in NECK_ELECTRODES:
        return "neck"
    if ch in EAR_ELECTRODES:
        return "ear"
    return "kept"


def test_full_montage_inside_circle():
    """User Q1: with the FULL 256-EGI montage, do all electrodes fall inside the
    head circle? Rendered with MNE's REAL head outline; sensors colored by the
    category we exclude (face/neck/ear) vs keep."""
    m = mne.channels.make_standard_montage("EGI_256")
    all_ch = m.ch_names
    info = make_info(all_ch)
    n = len(all_ch)
    cats = [category_of(c) for c in all_ch]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    spheres = [(tuple(float(v) for v in _check_sphere(None, info)),
                "PRODUCTION sphere=None (0,0,0,.095)"),
               (tuple(float(v) for v in _check_sphere("auto", info)),
                "sphere='auto' (fitted)")]
    for ax, (sph, ttl) in zip(axes, spheres):
        # real MNE outline via a flat topomap
        mne.viz.plot_topomap(np.zeros(n), info, axes=ax, show=False, sphere=sph,
                             cmap="Greys", contours=0, sensors=False,
                             outlines="head")
        pos = _find_topomap_coords(info, picks=list(range(n)), sphere=sph)
        for cat in CAT_COLORS:
            idx = [i for i, c in enumerate(cats) if c == cat]
            if idx:
                ax.scatter(pos[idx, 0], pos[idx, 1], s=12,
                           c=CAT_COLORS[cat], label=cat, zorder=6,
                           edgecolors="none")
        cx, cy, _, rr = sph
        dist = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
        frac_out = float(np.mean(dist > rr * 1.02))
        ax.set_title(f"{ttl}\n{frac_out*100:.0f}% of 256 electrodes outside circle",
                     fontsize=9)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle("Q1: are all 256 EGI electrodes inside the head circle? "
                 "(real MNE outline)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "q1_full_montage_inside_circle.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


def render_with_landmarks(ax, info, sphere, title, highlight):
    """Render a real MNE topomap (flat dummy data) so the head outline is
    exactly MNE's, then overlay sensor dots + names for the highlight chain
    using the SAME projection MNE used."""
    n = len(info["ch_names"])
    flat = np.zeros(n)
    mne.viz.plot_topomap(flat, info, axes=ax, show=False, sphere=sphere,
                         cmap="Greys", contours=0, sensors=False,
                         outlines="head")
    pos = _find_topomap_coords(info, picks=list(range(n)), sphere=sphere)
    # all kept sensors faint
    ax.scatter(pos[:, 0], pos[:, 1], s=6, c="0.6", zorder=5)
    # highlight chain
    name_to_idx = {c: i for i, c in enumerate(info["ch_names"])}
    cx, cy = sphere[0], sphere[1]
    rr = sphere[3]
    for rank, ch in enumerate(highlight):
        if ch not in name_to_idx:
            continue
        i = name_to_idx[ch]
        ax.scatter(pos[i, 0], pos[i, 1], s=30, c="red", zorder=6)
        ax.annotate(ch, (pos[i, 0], pos[i, 1]), fontsize=6, color="darkred",
                    xytext=(3, 3), textcoords="offset points", zorder=7)
    ax.set_title(title, fontsize=9)


def test_landmarks():
    """Decisive test: where does the parietal->occipital chain land, faithfully,
    under sphere=None (production) vs 'auto' vs a manually lowered sphere?"""
    kept = analysis_channels()
    info = make_info(kept)
    chain = posterior_midline_chain()

    m = mne.channels.make_standard_montage("EGI_256")
    pos3d = m.get_positions()["ch_pos"]
    print("\nPosterior midline chain (kept), top->occiput, with 3D (y,z):")
    for c in chain:
        print(f"  {c:6s}  y={pos3d[c][1]:+.3f}  z={pos3d[c][2]:+.3f}")

    sph_none = tuple(float(v) for v in _check_sphere(None, info))   # production
    sph_auto = tuple(float(v) for v in _check_sphere("auto", info))
    # Manual: keep default radius but drop the z-center so the equator (z=0)
    # maps nearer the rim, pushing occipital electrodes lower.
    sph_low = (0.0, 0.0, -0.02, 0.095)

    spheres = [
        (sph_none, "PRODUCTION  sphere=None  (0,0,0,.095)"),
        (sph_auto, "sphere='auto' (fitted)"),
        (sph_low,  "sphere=(0,0,-0.02,.095)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (sph, ttl) in zip(axes, spheres):
        render_with_landmarks(ax, info, sph, ttl, chain)
    fig.suptitle("TEST: parietal->occipital chain (red) vs MNE head outline, "
                 "by sphere\n(kept channels only; chain ends at occiput)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "test_landmarks.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nsaved {p}")


# ---------------------------------------------------------------------------
def test_coverage():
    """Show the actual interpolation field for a synthetic 'occipital hot spot'
    so we can see whether removing neck/inferior coverage makes a true occipital
    signal smear up into the parietal area, under each sphere."""
    kept = analysis_channels()
    info = make_info(kept)
    m = mne.channels.make_standard_montage("EGI_256")
    pos3d = m.get_positions()["ch_pos"]

    # synthetic signal: peak at the most occipital kept electrode, gaussian in 3D
    occ = min(kept, key=lambda c: pos3d[c][1] + pos3d[c][2])  # posterior & low
    p0 = pos3d[occ]
    data = np.array([np.exp(-((np.array(pos3d[c]) - p0) ** 2).sum() / (2 * 0.04 ** 2))
                     for c in kept])
    print(f"\n[coverage] synthetic occipital peak placed at {occ} "
          f"(y={p0[1]:+.3f}, z={p0[2]:+.3f})")

    info2 = make_info(kept)
    sph_none = tuple(float(v) for v in _check_sphere(None, info2))
    sph_auto = tuple(float(v) for v in _check_sphere("auto", info2))
    sph_low = (0.0, 0.0, -0.02, 0.095)
    spheres = [(sph_none, "PRODUCTION None"), (sph_auto, "auto"),
               (sph_low, "(0,0,-.02,.095)")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (sph, ttl) in zip(axes, spheres):
        mne.viz.plot_topomap(data, info, axes=ax, show=False, sphere=sph,
                             cmap="RdBu_r", contours=6, sensors=True)
        ax.set_title(ttl, fontsize=10)
    fig.suptitle("Synthetic OCCIPITAL hot-spot: where does it render? "
                 "(should be at the very bottom)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "test_coverage_occipital_peak.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    print("Production default sphere (None):", np.round(_check_sphere(None, make_info(analysis_channels())), 4))
    print("Fitted 'auto' sphere           :", np.round(_check_sphere("auto", make_info(analysis_channels())), 4))
    test_full_montage_inside_circle()
    test_landmarks()
    test_coverage()
    print("\nDone. See", OUT_DIR)
