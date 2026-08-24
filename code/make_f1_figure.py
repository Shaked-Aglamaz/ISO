"""
Compose Figure 1 (sleep overview) for the ISFS paper.

Portrait layout, three stacked rows, each = [ wide hypnospectrogram | group pie ].
Young=EL3011, Elderly=MCI27, MCI=YS0; the per-subject title strip is cropped off
and a group-name+n banner is drawn above each hypno in that group's colour, so
the green / red / blue coding matches Figures 3, 4 and 5.

Panel letters: A) hypnos, B) pies (column).

The sleep-architecture / N2-bout comparison table that used to sit in panel C is
now a standalone table (see code/make_table2_sleep.py). It was unreadable as a
pasted bitmap panel.

Pie values are read from results/demographics_V3/sleep_stage_means.csv, the
group means written by sleep_stage_pies.py, so this figure rebuilds offline and
cannot drift from the V3 numbers.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/make_f1_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import sleep_stage_pies as ssp  # reuse stage names + stage colors
from utils.config import GROUP_COLORS_DARK, group_label

HYPNO_DIR = Path("results/hypnospectrograms")
DEMO_DIR = Path("results/demographics_V3")
OUT = Path("thesis/figures/hypno_sleep_stages_V11.png")

# subject -> (group banner) ; order top-to-bottom in panel (a)
HYPNOS = [
    ("EL3011", "Young"),
    ("MCI27",  "Elderly"),
    ("YS0",    "MCI"),
]
TITLE_CROP_PX = 105  # rows removed from the top of each hypno (subject title strip)

FS_BANNER = 24
FS_PIE_LABEL = 16
FS_LETTER = 26

LABEL_DIST = 1.22       # default radial distance of a pie label
THIN_WEDGE_DEG = 20.0   # wedges narrower than this get their label pushed out
THIN_LABEL_DIST = 1.62  # so a thin slice's label clears its fat neighbour


def _push_out_thin_labels(wedges, texts) -> None:
    """Move the labels of narrow wedges further out along their own angle.

    Adjacent labels collide when one stage is only a few percent (N1 in the
    young group): both anchors land at almost the same place. Pushing only the
    thin wedge's label outward separates them without moving anything else.
    """
    for wedge, text in zip(wedges, texts):
        if (wedge.theta2 - wedge.theta1) >= THIN_WEDGE_DEG:
            continue
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        r = wedge.r * THIN_LABEL_DIST
        text.set_position((r * np.cos(mid), r * np.sin(mid)))


def load_cropped_hypno(subject_id: str) -> np.ndarray:
    p = HYPNO_DIR / f"{subject_id}_hypnospectrogram.png"
    arr = np.asarray(Image.open(p).convert("RGB"))
    return arr[TITLE_CROP_PX:, :, :]


def load_stage_means() -> pd.DataFrame:
    """Group sleep-stage means (% of recording) as written by sleep_stage_pies."""
    return pd.read_csv(DEMO_DIR / "sleep_stage_means.csv", index_col="group")


def main() -> None:
    means = load_stage_means()
    hypno_imgs = [load_cropped_hypno(sid) for sid, _ in HYPNOS]

    # --- figure scaffold (portrait) ---
    # Explicit margins + no tight-bbox on save so the anchored images sit where
    # the gridspec cells put them rather than hugging the figure edges.
    fig = plt.figure(figsize=(13, 11.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.6, 1.0],
                            wspace=0.02, hspace=0.22,
                            left=0.012, right=0.988, top=0.925, bottom=0.015)

    colors = [ssp.STAGE_COLORS[s] for s in ssp.STAGES]
    for i, (img, (_, banner), group) in enumerate(
        zip(hypno_imgs, HYPNOS, ssp.GROUP_ORDER)
    ):
        row = means.loc[group]

        # hypno (wide): height-limited in its cell, so west-anchor it to hug the
        # left edge and use the freed horizontal slack.
        ax_h = fig.add_subplot(grid[i, 0])
        ax_h.imshow(img)
        ax_h.axis("off")
        ax_h.set_anchor("W")
        ax_h.set_title(f"{group_label(banner)} (n={int(row['n'])})",
                       fontsize=FS_BANNER, fontweight="bold", pad=8,
                       color=GROUP_COLORS_DARK[banner])

        # pie for the same group (stacked vertically, so no collision risk)
        ax_p = fig.add_subplot(grid[i, 1])
        # West-anchor + smaller radius so the right-hand stage labels (Wake/N1)
        # don't run off the figure's right edge.
        ax_p.set_anchor("W")
        sizes = [row[s] for s in ssp.STAGES]
        labels = [f"{s}\n{v:.1f}%" for s, v in zip(ssp.STAGES, sizes)]
        wedges, texts = ax_p.pie(
            sizes, labels=labels, colors=colors, labeldistance=LABEL_DIST,
            radius=0.80, startangle=90, counterclock=False,
            textprops={"fontsize": FS_PIE_LABEL},
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        _push_out_thin_labels(wedges, texts)
        ax_p.set_aspect("equal")

    # --- panel letters ---
    fig.text(0.012, 0.992, "A)", fontsize=FS_LETTER, fontweight="bold", va="top")  # hypnos
    fig.text(0.728, 0.992, "B)", fontsize=FS_LETTER, fontweight="bold", va="top")  # pies col

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
