"""Re-emit the Figure 1 methods panels at slide geometry for the defense deck.

Nothing is recomputed and no manuscript asset is touched: the panels are drawn by
the very functions that build `thesis/figures/methods_flow_roi_v3.png`
(`code/make_f2_figure.py`), just laid out landscape and with the fonts scaled up
so they stay legible when projected.

Outputs (results/defense_slides_V1/):
    methods_panel_AB.png   spindle timeline + raw / sigma / envelope traces (landscape)
    methods_panel_C.png    mean envelope FFT + Gaussian fit, one channel, all its bouts
    methods_panel_ROI.png  Dimitriades young-adult AUC hotspot -> our 256-ch ROI

The two expensive preparation steps (FIF load + yasa spindle detection, and the
full-recording pipeline run for panel C) are cached under
results/defense_slides_V1/_cache/, so re-runs that only change styling are fast.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/defense/make_methods_panels.py
    PYTHONIOENCODING=utf-8 python code/defense/make_methods_panels.py --refresh
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from make_f2_figure import (  # noqa: E402
    ZOOM_DUR,
    ZOOM_START,
    panel_A,
    panel_B,
    panel_C,
    prepare_data,
    prepare_panelC,
)

OUT_DIR = Path("results/defense_slides_V1")
CACHE_DIR = OUT_DIR / "_cache"

# The young-adult AUC topography of Dimitriades et al. (2024) that the ROI was
# read off, and the resulting 256-channel electrode set.
SOURCE_TOPO = Path("code/debug/YA_AUC.png")
ROI_MAP = Path("code/debug/AUC ROI 2.png")

FONT_SCALE = 1.9   # panel fonts are sized for a 9.5 in page figure


def bump_fonts(ax, scale=FONT_SCALE):
    """Scale every piece of text on an axis, leaving the drawing untouched."""
    items = [ax.title, ax.xaxis.label, ax.yaxis.label]
    items += ax.get_xticklabels() + ax.get_yticklabels()
    legend = ax.get_legend()
    if legend is not None:
        items += legend.get_texts()
        legend.set_title(legend.get_title().get_text() or None)
    for item in items:
        item.set_fontsize(item.get_fontsize() * scale)


def cached(name, builder, refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists() and not refresh:
        print(f"  (cache) {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    print(f"  computing {name} …")
    value = builder()
    with open(path, "wb") as fh:
        pickle.dump(value, fh)
    return value


def make_panel_AB(d):
    """A) spindle timeline over the example bout, B) the signal chain below it."""
    fig = plt.figure(figsize=(13.0, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.42, 2.0],
                             left=0.175, right=0.985, top=0.91, bottom=0.145,
                             hspace=0.45)

    ax_A = fig.add_subplot(outer[0])
    panel_A(ax_A, d)
    bump_fonts(ax_A)
    # the tick numbers carry the scale; dropping A's axis label keeps the
    # zoom-in arrows from crossing text on their way down to panel B
    ax_A.set_xlabel("")

    gsB = outer[1].subgridspec(3, 1, hspace=0.12)
    axes_B = [fig.add_subplot(gsB[i]) for i in range(3)]
    panel_B(axes_B, d)
    for ax in axes_B:
        bump_fonts(ax)
        # panel_B writes the row label horizontally in the left margin, so it
        # needs its own size and enough padding to clear the tick numbers
        ax.yaxis.label.set_fontsize(15)
        ax.yaxis.labelpad = 52

    for x_box, x_b in [(ZOOM_START, 0.0), (ZOOM_START + ZOOM_DUR, 1.0)]:
        fig.add_artist(ConnectionPatch(
            xyA=(x_box, 0), coordsA=ax_A.transData,
            xyB=(x_b, 1), coordsB=axes_B[0].transAxes,
            arrowstyle="-|>", color="green", lw=1.8, mutation_scale=18))

    for ax, label in [(ax_A, "A"), (axes_B[0], "B")]:
        pos = ax.get_position()
        fig.text(0.008, pos.y1 + 0.01, label, fontsize=26, fontweight="bold",
                 va="bottom")

    out = OUT_DIR / "methods_panel_AB.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def make_panel_C(pd_c):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    fig.subplots_adjust(left=0.145, right=0.97, top=0.95, bottom=0.135)
    panel_C(ax, pd_c)
    bump_fonts(ax, 1.75)
    out = OUT_DIR / "methods_panel_C.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def make_panel_ROI():
    """The ROI provenance: published young-adult hotspot -> our 256-ch electrode set."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.06)

    for ax, img, title in zip(
            axes,
            [SOURCE_TOPO, ROI_MAP],
            # The right-hand image is the same published map with our montage
            # drawn on top; that is how the ROI was read off, so say so.
            ["Young-adult AUC hotspot\n(Dimitriades et al., 2024; 128 channels)",
             "Same map, our 256-channel montage overlaid\nROI electrodes in green"]):
        ax.imshow(np.asarray(Image.open(img).convert("RGB")))
        ax.set_title(title, fontsize=17)
        ax.axis("off")

    # arrow from the published map to ours
    fig.add_artist(ConnectionPatch(
        xyA=(1.02, 0.5), coordsA=axes[0].transAxes,
        xyB=(-0.02, 0.5), coordsB=axes[1].transAxes,
        arrowstyle="-|>", color="#2f9c86", lw=3.0, mutation_scale=26))

    out = OUT_DIR / "methods_panel_ROI.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the cached example-subject data")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = cached("panel_ab_data", prepare_data, args.refresh)
    pd_c = cached("panel_c_data", prepare_panelC, args.refresh)

    make_panel_AB(d)
    make_panel_C(pd_c)
    make_panel_ROI()


if __name__ == "__main__":
    main()
