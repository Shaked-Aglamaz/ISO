"""
Compose Figure S3 (example ISFS spectra) for the ISFS thesis.

Answers Yuval's C432: "Could we include some figure or supplementary figure with
various examples from different subjects?"

Three rows, one per group, two hand-picked example channels each. Every panel is
redrawn from the production outputs of that channel — the baseline-corrected mean
spectrum in {sub}_{ch}_spectral_power.csv and the fitted parameters parsed out of
{sub}_{ch}_analysis_summary.txt — so nothing is recomputed and nothing is cropped
out of an existing PNG. The per-channel titles the pipeline stamps on
*_mean_spectrum.png are dropped; the group banner and the caption carry that
information instead.

Subject codes are deliberately NOT drawn. Two of the elderly examples are coded
MCI43 and SL44 and two of the aMCI examples are coded MCI04 and MCI41, so printing
the raw IDs would suggest the wrong group membership to a reader.

Visual grammar matches Figure 1C (code/make_f2_figure.py): same curve colours,
same names-only legend, numbers in the panel annotation rather than the legend.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/make_s3_figure.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import GROUP_COLORS_DARK, group_label

OUT = Path("thesis/figures/s3_example_spectra_V11.png")

# group -> the two example channels, in the order the user picked them.
EXAMPLES = {
    "Young": [
        "results/sigma_fix_YA/ON68/ON68_E144_output/ON68_E144",
        "results/sigma_fix_YA/EL3004/EL3004_E122_output/EL3004_E122",
    ],
    "Elderly": [
        "results/sigma_fix_HE/MCI43/MCI43_E19_output/MCI43_E19",
        "results/sigma_fix_HE/SL44/SL44_VREF_output/SL44_VREF",
    ],
    "MCI": [
        "results/sigma_fix_MCI/MCI04/MCI04_E195_output/MCI04_E195",
        "results/sigma_fix_MCI/MCI41/MCI41_E99_output/MCI41_E99",
    ],
}

# Figure 1C's palette, so the supplementary examples read as the same object as
# the worked example in the methods figure.
C_POWER = "#2E86AB"
C_FIT = "#A23B72"
C_BW = "#F18F01"

# The figure is pasted at 6.5 in on a 9.5 in canvas, so on-page type is 0.68x
# what is set here. Everything below is chosen to land at 10 pt or more.
FS_BANNER = 22
FS_LABEL = 17
FS_TICK = 14
FS_ANNOT = 14
FS_LEGEND = 15

SUMMARY_FIELDS = {
    "n_bouts": r"Number of N2 bouts: (\d+)",
    "mu": r"Peak Frequency \(μ\): ([\d.]+)",
    "bandwidth": r"Bandwidth \(2σ\): ([\d.]+)",
    "amplitude": r"Peak Amplitude: ([\d.]+)",
    "auc": r"Area Under Curve \(AUC\): ([\d.]+)",
    "sigma": r"Sigma \(σ\): ([\d.]+)",
    "threshold": r"Detection threshold: ([\d.]+)",
}


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(((x - mu) / sigma) ** 2))


def load_example(base: str) -> dict:
    """Read one channel's mean spectrum and its fitted parameters."""
    spectrum = pd.read_csv(f"{base}_spectral_power.csv")
    text = Path(f"{base}_analysis_summary.txt").read_text(encoding="utf-8")
    frequencies = spectrum["frequency"].to_numpy()
    # The CSV column named mean_power is main_loop.py's mean_power_no_baseline,
    # but the Gaussian was fitted to the baseline-corrected spectrum. Plotting the
    # column as-is leaves the curve sitting a whole baseline above its own fit, so
    # apply the same shift isfs_presence.py does before fitting.
    power = spectrum["mean_power"].to_numpy()
    baseline = (frequencies > 0.06) & (frequencies < 0.102)
    power = power - np.nanmean(power[baseline])
    out = {"frequencies": frequencies, "mean_power": power}
    for key, pattern in SUMMARY_FIELDS.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"{key} not found in {base}_analysis_summary.txt")
        out[key] = float(match.group(1))
    out["n_bouts"] = int(out["n_bouts"])

    # The summary's "Peak Frequency (μ)" is isfs_presence.py's actual_pf, the
    # frequency-grid point nearest the fitted curve's apex, not the fitted mu that
    # centres the Gaussian. Only the former is written to disk, so recover the
    # curve by repeating the pipeline's fit on the same data, then assert it
    # reproduces the amplitude and sigma the pipeline recorded.
    valid = ~np.isnan(power)
    popt, _ = curve_fit(gaussian, frequencies[valid], power[valid],
                        p0=[np.nanmax(power[valid]), 0.02, 0.01])
    amplitude_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])
    for name, refit, recorded in (("amplitude", amplitude_fit, out["amplitude"]),
                                  ("sigma", sigma_fit, out["sigma"])):
        if abs(refit - recorded) > 1e-4:
            raise ValueError(f"{base}: refitted {name} {refit:.6f} does not "
                             f"reproduce the recorded {recorded:.6f}")
    out["mu_curve"] = mu_fit
    out["sigma"] = sigma_fit
    out["amplitude"] = amplitude_fit
    return out


def draw_panel(ax, ex: dict) -> None:
    freqs, power = ex["frequencies"], ex["mean_power"]
    ax.plot(freqs, power, color=C_POWER, linewidth=2.2,
            label="Mean relative power", zorder=2)

    x_fit = np.linspace(freqs[0], freqs[-1], 500)
    y_fit = gaussian(x_fit, ex["amplitude"], ex["mu_curve"], ex["sigma"])
    ax.plot(x_fit, y_fit, color=C_FIT, linewidth=2.2, alpha=0.9,
            label="Gaussian fit", zorder=3)
    ax.plot(ex["mu_curve"], ex["amplitude"], "o", color=C_FIT, markersize=8, zorder=4)

    x1, x2 = ex["mu_curve"] - ex["sigma"], ex["mu_curve"] + ex["sigma"]
    ax.hlines(gaussian(x1, ex["amplitude"], ex["mu_curve"], ex["sigma"]), x1, x2,
              colors=C_BW, linewidth=3, label="Bandwidth", zorder=3)
    band = (x_fit >= x1) & (x_fit <= x2)
    ax.fill_between(x_fit[band], y_fit[band], color=C_FIT, alpha=0.2,
                    label="±1σ area", zorder=1)

    ax.axhline(ex["threshold"], color="gray", linestyle="--", linewidth=1.5,
               alpha=0.7, label="Threshold", zorder=1)

    ax.set_xlim(0, freqs[-1])
    # The corrected spectrum dips below zero in the baseline band, so the floor
    # cannot be pinned at 0 the way it could on the uncorrected curve.
    top = max(power.max(), ex["amplitude"])
    ax.set_ylim(min(power.min(), 0) * 1.15 - 0.02 * top, top * 1.42)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax.text(0.97, 0.96,
            f"Peak frequency {ex['mu']:.4f} Hz\n"
            f"Bandwidth {ex['bandwidth']:.4f} Hz\n"
            f"AUC {ex['auc']:.2f}   {ex['n_bouts']} bouts",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.75", alpha=0.9))


def main() -> None:
    fig = plt.figure(figsize=(9.5, 9.5))
    grid = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.16,
                            left=0.085, right=0.985, top=0.928, bottom=0.155)

    handles = None
    for row, (group, bases) in enumerate(EXAMPLES.items()):
        for col, base in enumerate(bases):
            ax = fig.add_subplot(grid[row, col])
            draw_panel(ax, load_example(base))
            if handles is None:
                handles, _ = ax.get_legend_handles_labels()

        # Group banner spanning the row, in that group's colour (as in Figure 2).
        first = fig.axes[row * 2]
        last = fig.axes[row * 2 + 1]
        x_mid = (first.get_position().x0 + last.get_position().x1) / 2
        fig.text(x_mid, first.get_position().y1 + 0.017, group_label(group),
                 fontsize=FS_BANNER, fontweight="bold", ha="center", va="bottom",
                 color=GROUP_COLORS_DARK[group])

    fig.supxlabel("Frequency (Hz)", fontsize=FS_LABEL, y=0.102)
    fig.supylabel("Relative power (AU)", fontsize=FS_LABEL, x=0.018)
    # Three columns, not five: at this font a single row of five entries runs off
    # both edges of the canvas.
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, 0.005))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
