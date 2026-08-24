"""
Compose the two stacked-topography paper figures from the V11 panels.

F4 (AUC): A = raw                 (`three_group_topo_auc_raw.png`)
          B = normalized + cluster (`three_group_topo_auc.png`)
S1      : A = peak-frequency topo (`three_group_topo_peak_frequency.png`)
          B = bandwidth topo      (`three_group_topo_bandwidth.png`)

Each composite stacks the two panels vertically with capital A/B letters.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/make_topo_composites.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = Path("results/group_comparison_results/three_groups_V11")
OUT_DIR = Path("thesis/figures")

COMPOSITES = {
    "f4_auc_composite_V11.png": ("three_group_topo_auc_raw.png", "three_group_topo_auc.png"),
    "s1_topo_composite_V11.png": ("three_group_topo_peak_frequency.png", "three_group_topo_bandwidth.png"),
}


def stack(top_png: str, bot_png: str, out_name: str) -> None:
    top = np.asarray(Image.open(SRC / top_png).convert("RGB"))
    bot = np.asarray(Image.open(SRC / bot_png).convert("RGB"))

    # row height in inches scaled to each panel's aspect at a fixed width
    width_in = 13.0
    h_top = width_in * top.shape[0] / top.shape[1]
    h_bot = width_in * bot.shape[0] / bot.shape[1]

    fig = plt.figure(figsize=(width_in, h_top + h_bot + 0.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_top, h_bot], hspace=0.04,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)

    for i, (img, letter) in enumerate([(top, "A)"), (bot, "B)")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.005, 0.98, letter, transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="top", ha="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    for out_name, (top_png, bot_png) in COMPOSITES.items():
        stack(top_png, bot_png, out_name)


if __name__ == "__main__":
    main()
