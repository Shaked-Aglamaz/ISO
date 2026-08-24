"""Extra slide assets for the defense deck (nothing here is a manuscript figure).

1. lc_ne_schematic.png        Background slide: the infra-slow cycle and the
                              noradrenergic account. Our own drawing, after
                              Lecci et al. (2017) and Osorio-Forero et al. (2021);
                              no panel is taken from a paper.
2. bandwidth_vs_n2_duration.png  The C429 answer as a picture: ISFS bandwidth
                              against how much N2 sleep each participant
                              contributed (r = +0.386, from the V11 ANCOVA
                              per-subject CSV; nothing recomputed but the line).
3. moca_panel_peak_frequency.png / moca_panel_roi_auc.png
                              Two quadrants cropped out of the S3 MoCA grid, so
                              the axis text survives projection.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/defense/make_slide_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
from utils.config import GROUP_COLORS, GROUP_COLORS_DARK, group_label  # noqa: E402

OUT_DIR = Path("results/defense_slides_V1")
ANCOVA_CSV = Path("results/group_comparison_results/three_groups_V11/"
                  "three_group_ancova_per_subject.csv")
MOCA_GRID = Path("results/moca_correlation_V4/moca_correlations_grid.png")

PERIOD = 50.0        # s, the ~0.02 Hz cycle
C_SIGMA = "#2E86AB"
C_NE = "#7d3c98"
C_SPINDLE = "#0b2e6b"


def lc_ne_schematic():
    """Anti-phase noradrenaline and sigma power, with the two NREM substates."""
    t = np.linspace(0, 3 * PERIOD, 2000)
    sigma = 0.5 + 0.45 * np.cos(2 * np.pi * t / PERIOD)
    ne = 0.5 - 0.45 * np.cos(2 * np.pi * t / PERIOD)

    fig, (ax_sp, ax, ax_band) = plt.subplots(
        3, 1, figsize=(13.6, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[0.30, 1.0, 0.16], hspace=0.10))
    # room on the right for the two curve labels, which replace the legend
    fig.subplots_adjust(left=0.085, right=0.745, top=0.95, bottom=0.145)

    # substate shading: sigma peaks = offline / stable, sigma troughs = fragile
    for k in range(3):
        centre_stable = k * PERIOD
        ax_sp.axvspan(centre_stable - 12, centre_stable + 12, color="#8dd3c7", alpha=0.35, lw=0)
        ax.axvspan(centre_stable - 12, centre_stable + 12, color="#8dd3c7", alpha=0.35, lw=0)
        centre_fragile = centre_stable + PERIOD / 2
        ax_sp.axvspan(centre_fragile - 12, centre_fragile + 12, color="#fb8072", alpha=0.30, lw=0)
        ax.axvspan(centre_fragile - 12, centre_fragile + 12, color="#fb8072", alpha=0.30, lw=0)

    # spindle bars: dense in the offline substate, sparse in the fragile one
    rng = np.random.default_rng(7)
    for k in range(3):
        for centre, n in [(k * PERIOD, 7), (k * PERIOD + PERIOD / 2, 1)]:
            for pos in centre + rng.uniform(-11, 11, n):
                if 0 <= pos <= t[-1]:
                    ax_sp.axvspan(pos, pos + 0.9, color=C_SPINDLE, alpha=0.75, lw=0)
    ax_sp.set_ylim(0, 1)
    ax_sp.set_yticks([])
    ax_sp.set_ylabel("Spindles", fontsize=17, rotation=0, ha="right", va="center",
                     labelpad=12)
    for s in ("top", "right", "left", "bottom"):
        ax_sp.spines[s].set_visible(False)

    # both traces solid: the dashed line read as a different kind of quantity.
    # They are told apart by colour and by the label sitting at the end of each curve.
    ax.plot(t, sigma, color=C_SIGMA, lw=4.0)
    ax.plot(t, ne, color=C_NE, lw=4.0)
    ax.text(1.015, sigma[-1], "Sigma power\n(13-16 Hz)",
            transform=ax.get_yaxis_transform(),
            color=C_SIGMA, fontsize=21, fontweight="bold", va="center", ha="left")
    ax.text(1.015, ne[-1], "Noradrenaline\n(LC activity)",
            transform=ax.get_yaxis_transform(),
            color=C_NE, fontsize=21, fontweight="bold", va="center", ha="left")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.10, 1.22)
    ax.set_yticks([])
    ax.tick_params(labelsize=16, bottom=False, labelbottom=False)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    # period annotation between two sigma peaks
    ax.annotate("", xy=(PERIOD, 1.10), xytext=(2 * PERIOD, 1.10),
                arrowprops=dict(arrowstyle="<->", lw=2, color="black"))
    ax.text(1.5 * PERIOD, 1.13, "~50 s  (≈0.02 Hz)", ha="center", va="bottom",
            fontsize=17)

    # substate band: names the two alternating states under the traces
    for k in range(3):
        for centre, label, colour, text_colour in [
                (k * PERIOD, "offline", "#8dd3c7", "#1d6555"),
                (k * PERIOD + PERIOD / 2, "fragile", "#fb8072", "#8f2a1c")]:
            ax_band.axvspan(centre - 12, centre + 12, color=colour, alpha=0.55, lw=0)
            if 0 <= centre <= t[-1]:
                ax_band.text(centre, 0.5, label, ha="center", va="center",
                             fontsize=15, color=text_colour)
    ax_band.set_ylim(0, 1)
    ax_band.set_yticks([])
    ax_band.set_xticks(np.arange(0, 3 * PERIOD + 1, 25))
    ax_band.set_xlabel("Time (s)", fontsize=18)
    ax_band.tick_params(labelsize=16)
    for s in ("top", "right", "left"):
        ax_band.spines[s].set_visible(False)

    out = OUT_DIR / "lc_ne_schematic.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def bandwidth_vs_duration():
    df = pd.read_csv(ANCOVA_CSV)
    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.95, bottom=0.145)

    for group in ["Young", "Elderly", "MCI"]:
        sub = df[df["group"] == group]
        ax.scatter(sub["total_dur_min"], sub["bandwidth"], s=95,
                   facecolor=GROUP_COLORS[group], edgecolor=GROUP_COLORS_DARK[group],
                   linewidth=1.2, alpha=0.9, label=f"{group_label(group)} (N={len(sub)})")

    x = df["total_dur_min"].to_numpy()
    y = df["bandwidth"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", lw=2.2, ls="--")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.97, 0.06, f"r = {r:+.3f}   (N = {len(df)})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=18)

    ax.set_xlabel("Analyzed N2 sleep (min)", fontsize=19)
    ax.set_ylabel("ISFS bandwidth (Hz)", fontsize=19)
    ax.tick_params(labelsize=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.9)

    out = OUT_DIR / "bandwidth_vs_n2_duration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}  (r = {r:+.4f}, slope = {slope:.3e} Hz/min)")


def crop_moca_panels():
    """Cut the 2x2 MoCA grid into single panels so the type survives projection.

    Panel order in code/moca_correlation.py plot_grid: peak frequency, bandwidth
    (top row), whole-scalp AUC, ROI AUC (bottom row); a shared legend sits below
    the grid, so the vertical split is taken above it.
    """
    img = Image.open(MOCA_GRID)
    w, h = img.size
    grid_bottom = int(h * 0.925)          # exclude the shared legend strip
    mid_x, mid_y = w // 2, grid_bottom // 2

    crops = {
        "moca_panel_peak_frequency.png": (0, 0, mid_x, mid_y),
        "moca_panel_roi_auc.png": (mid_x, mid_y, w, grid_bottom),
    }
    for name, box in crops.items():
        out = OUT_DIR / name
        img.crop(box).save(out)
        print(f"Saved: {out}  {img.crop(box).size}")


HYPNOS = {
    "hypno_young.png": "results/hypnospectrograms/EL3011_hypnospectrogram.png",
    "hypno_elderly.png": "results/hypnospectrograms/MCI27_hypnospectrogram.png",
    "hypno_amci.png": "results/hypnospectrograms/YS0_hypnospectrogram.png",
}


def crop_hypnograms():
    """Drop the title band, which prints the subject code.

    Two of the three example recordings carry codes that name the wrong group
    (`MCI27` is an elderly participant, `YS0` an aMCI patient), so showing the
    code would tell the audience something false. The manuscript figure crops
    these titles for the same reason.
    """
    for name, src in HYPNOS.items():
        img = Image.open(src)
        w, h = img.size
        out = OUT_DIR / name
        img.crop((0, int(h * 0.11), w, h)).save(out)
        print(f"Saved: {out}  {Image.open(out).size}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lc_ne_schematic()
    bandwidth_vs_duration()
    crop_moca_panels()
    crop_hypnograms()


if __name__ == "__main__":
    main()
