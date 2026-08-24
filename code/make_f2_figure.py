"""
Compose Figure 2 (methods flow + ROI) for the ISFS paper — fully code-generated.

Every panel is a real matplotlib plot computed from the same example subject
(RD43, YA group, channel VREF, 745 s clean N2 bout cropped at 7418-8163 s):

    A) Spindle timeline over the whole bout, with the zoom-in window marked.
    B) Signal chain over a 45 s window (68-113 s): broadband VREF, 13-16 Hz
       sigma, and the Gabor sigma-power envelope, with detected spindles shaded.
    C) FFT of the sigma envelope + Gaussian fit (pipeline-faithful, via
       new_iso.isfs_presence.extract_isfs_parameters on the single example bout).
    D) Central-parietal ROI map (kept as the existing clean topo bitmap).

Layout (portrait, reads top-to-bottom so it fills a portrait Doc page):
A (timeline) full width on top, B (3 stacked traces) full width below it, and
C (FFT + fit) beside D (ROI map) on the bottom row. No divider line.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/make_f2_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from PIL import Image

import mne
import yasa

# Pipeline modules (same imports main_loop.py uses).
sys.path.insert(0, str(Path(__file__).resolve().parent / "new_iso"))
from morlet import calculate_gabor_wavelet                       # noqa: E402
from mult_chan import extract_clean_sleep_bouts                  # noqa: E402
from isfs_presence import extract_isfs_parameters, gaussian_func  # noqa: E402

mne.set_log_level("error")

# ----------------------------------------------------------------------------- config
SUBJECT = "RD43"
RAW_PATH = Path(
    "I:/Shaked/ISO_data/control_clean/RD43/"
    "RD43_176-channels_resample250_filtered_scored_bad-epochs_avgref_interpolate_raw.fif"
)
CROP = (7418.0, 8163.0)          # clean N2 bout (s) — matches try.ipynb cell 25
CHANNEL = "VREF"
SIGMA_LOW, SIGMA_HIGH = 13, 16   # sigma band (Hz)
ZOOM_START, ZOOM_DUR = 68, 45    # panel B window (s into the crop): 68-113 s
FREQ_RES = 0.2                   # Gabor frequency step (Hz)

ROI = Path("code/debug/AUC ROI 2.png")
OUT = Path("thesis/figures/methods_flow_roi_v3.png")

# colour palette (reuse the pipeline's visualization colours for C)
C_POWER = "#2E86AB"
C_FIT = "#A23B72"
C_BW = "#F18F01"
C_SPINDLE = "#0b2e6b"


def prepare_data():
    """Load RD43 VREF example bout and compute the broadband / sigma / envelope traces."""
    raw = mne.io.read_raw_fif(RAW_PATH, preload=False)
    raw = raw.copy().crop(*CROP).pick([CHANNEL]).load_data()
    sfreq = raw.info["sfreq"]

    # Spindle detection on VREF (Start/Duration are seconds from the crop start).
    spindles = yasa.spindles_detect(raw, freq_sp=(SIGMA_LOW, SIGMA_HIGH))
    if spindles is not None:
        sp = spindles.summary()
        sp = sp[sp["Channel"] == CHANNEL][["Start", "Duration"]].to_numpy()
    else:
        sp = np.empty((0, 2))

    vref_v = raw.get_data(picks=CHANNEL)[0]                       # V (broadband)
    sigma_v = raw.copy().filter(SIGMA_LOW, SIGMA_HIGH).get_data(picks=CHANNEL)[0]

    # Sigma-power envelope: Gabor-Morlet (13-16 Hz @0.2 Hz), mean |transform| over freqs
    # — identical to main_loop.analyze_channel.
    transform = calculate_gabor_wavelet(vref_v, sfreq, SIGMA_LOW, SIGMA_HIGH, freq_res=FREQ_RES)
    envelope = np.mean(np.abs(transform), axis=0)

    return dict(sfreq=sfreq, spindles=sp, vref_v=vref_v, sigma_v=sigma_v,
                envelope=envelope, total_dur=raw.times[-1])


def prepare_panelC():
    """Pipeline-faithful spectrum for panel C: ALL clean N2 bouts of RD43 VREF.

    Mirrors main_loop.analyze_channel — extract clean N2 bouts from the full
    recording, Gabor envelope on the concatenated VREF data, then the multi-bout
    FFT / normalization / baseline / Gaussian fit via extract_isfs_parameters.
    """
    raw = mne.io.read_raw_fif(RAW_PATH, preload=False).pick([CHANNEL]).load_data()
    sfreq = raw.info["sfreq"]

    n2_data, bout_metadata = extract_clean_sleep_bouts(raw)
    channel_data = n2_data[0, :]
    transform = calculate_gabor_wavelet(channel_data, sfreq, SIGMA_LOW, SIGMA_HIGH,
                                        freq_res=FREQ_RES)
    envelope = np.mean(np.abs(transform), axis=0)

    (_pf, _bw, _auc, _pp), plot_data = extract_isfs_parameters(
        envelope, bout_metadata, sfreq)
    plot_data["num_bouts"] = bout_metadata.shape[1]
    return plot_data


def shade_spindles(ax, spindles, lo=None, hi=None):
    """Blue spindle bars; restricted to [lo, hi] if given."""
    for start, dur in spindles:
        if hi is not None and (start > hi or start + dur < lo):
            continue
        ax.axvspan(start, start + dur, color=C_SPINDLE, alpha=0.35, lw=0)


# ----------------------------------------------------------------------------- panels
def panel_A(ax, d):
    """Spindle timeline over the whole bout + zoom-in marker. Only the X-axis spine."""
    shade_spindles(ax, d["spindles"])
    ax.add_patch(Rectangle((ZOOM_START, 0), ZOOM_DUR, 1,
                           transform=ax.get_xaxis_transform(),
                           fill=False, edgecolor="green", linewidth=2.2, zorder=5))
    ax.set_xlim(0, d["total_dur"])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, d["total_dur"] + 1, 120))
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.tick_params(labelsize=11)
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=C_SPINDLE, alpha=0.5, label="Spindle")],
              loc="upper right", fontsize=11, framealpha=0.9)


def panel_B(axes, d):
    """Three stacked traces over the 68-113 s window. Shared x, minimal spines."""
    sfreq = d["sfreq"]
    lo, hi = ZOOM_START, ZOOM_START + ZOOM_DUR
    s0, s1 = int(lo * sfreq), int(hi * sfreq)
    t = np.arange(s0, s1) / sfreq

    # last field = y-limit zoom-out factor (>1 squeezes the trace, like MNE's '-')
    # Labels wrapped narrow (3 lines) so the left margin can shrink and the
    # A+B traces fill more of the figure width.
    traces = [
        (d["vref_v"][s0:s1] * 1e6, "Raw\nchannel\n(µV)", "black", 0.4, 1.8),
        (d["sigma_v"][s0:s1] * 1e6, f"Sigma\n{SIGMA_LOW}-{SIGMA_HIGH} Hz\n(µV)", C_FIT, 0.4, 1.0),
        (d["envelope"][s0:s1] * 1e6, "Sigma\nenvelope\n(AU)", C_POWER, 0.9, 1.0),
    ]
    for i, (ax, (y, label, color, lw, ysqueeze)) in enumerate(zip(axes, traces)):
        shade_spindles(ax, d["spindles"], lo, hi)
        ax.plot(t, y, color=color, lw=lw)
        ax.set_xlim(lo, hi)
        if ysqueeze != 1.0:
            ymax = np.nanmax(np.abs(y)) * ysqueeze
            ax.set_ylim(-ymax, ymax)
        ax.set_ylabel(label, fontsize=10, rotation=0, ha="center", va="center",
                      ma="center", labelpad=30)
        ax.tick_params(labelsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i < len(axes) - 1:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=12)


def panel_C(ax, pd_):
    """Mean envelope FFT + Gaussian fit across ALL of RD43's VREF N2 bouts.

    Drawn straight from the production pipeline's plot_data (baseline-corrected
    mean spectrum, fitted Gaussian, threshold, ±1σ AUC).
    """
    freqs = pd_["frequencies"]
    ax.plot(freqs, pd_["mean_power"], color=C_POWER, lw=2.2,
            label="Mean relative power", zorder=2)

    fp = pd_["fitted_params"]
    if fp is not None:
        amp, mu, sigma = fp
        x_fit = np.linspace(freqs[0], freqs[-1], 500)
        y_fit = gaussian_func(x_fit, amp, mu, sigma)
        x1, x2 = mu - sigma, mu + sigma
        band = (x_fit >= x1) & (x_fit <= x2)
        ax.plot(x_fit, y_fit, color=C_FIT, lw=2.2, alpha=0.9,
                label="Gaussian fit", zorder=3)
        ax.plot(mu, amp, "o", color=C_FIT, ms=8, zorder=4)
        ax.hlines(gaussian_func(x1, amp, mu, sigma), x1, x2, colors=C_BW, lw=3,
                  label="Bandwidth", zorder=3)
        ax.fill_between(x_fit[band], y_fit[band], color=C_FIT, alpha=0.2,
                        label="±1σ area", zorder=1)
        ax.axhline(pd_["threshold"], color="gray", ls="--", lw=1.5, alpha=0.7,
                   label="Threshold", zorder=1)
        # numeric values reported in the caption, not the legend
        print(f"  panel C example values: n_bouts={pd_['num_bouts']}, "
              f"mu={mu:.3f} Hz, bandwidth={2 * sigma:.3f} Hz, "
              f"AUC={pd_['auc']:.2f}, threshold={pd_['threshold']:.2f}")
    else:
        print(f"  ! Gaussian fit failed for panel C: {pd_['failure_reason']}")

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative power (AU)", fontsize=12)
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Small top headroom so the (now narrow, names-only) upper-right legend sits
    # just above the data — the peak is on the far left and the right half is
    # low, so the narrow legend clears the curve.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.12)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)


def panel_D(ax):
    ax.imshow(np.asarray(Image.open(ROI).convert("RGB")))
    ax.axis("off")


# ----------------------------------------------------------------------------- compose
def main():
    d = prepare_data()
    pd_c = prepare_panelC()

    # Portrait stack (rebuilt 2026-06-22): the figure reads top-to-bottom and
    # fills a portrait Doc page, so the text stays legible once scaled down.
    #   A) spindle timeline           — full width (top)
    #   B) raw / sigma / envelope     — full width, 3 stacked traces
    #   C) FFT + fit  |  D) ROI map   — side by side (bottom row)
    # Wide left margin leaves room for panel B's horizontal row labels (the
    # longest, "Sigma 13-16 Hz", was clipping); the C+D row is given a smaller
    # height share so the spectrum and ROI map aren't oversized.
    fig = plt.figure(figsize=(9.5, 9.8))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.95, 2.15],
                             left=0.15, right=0.975, top=0.95, bottom=0.055,
                             hspace=0.45)

    ax_A = fig.add_subplot(outer[0])
    panel_A(ax_A, d)

    gsB = outer[1].subgridspec(3, 1, hspace=0.12)
    axes_B = [fig.add_subplot(gsB[i]) for i in range(3)]
    panel_B(axes_B, d)

    # C narrower than its cell so the spectrum is taller-than-wide; D takes the
    # extra width as a larger ROI map.
    gsCD = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.18)
    ax_C = fig.add_subplot(gsCD[0])
    panel_C(ax_C, pd_c)

    ax_D = fig.add_subplot(gsCD[1])
    panel_D(ax_D)

    # zoom-in arrows: A's green window "opens" downward into panel B
    for x_box, x_b in [(ZOOM_START, 0.0), (ZOOM_START + ZOOM_DUR, 1.0)]:
        fig.add_artist(ConnectionPatch(
            xyA=(x_box, 0), coordsA=ax_A.transData,
            xyB=(x_b, 1), coordsB=axes_B[0].transAxes,
            arrowstyle="-|>", color="green", lw=1.5, mutation_scale=14))

    # panel letters — left-column letters (A/B/C) sit flush at the far left,
    # D's sits just left of its own panel.
    def place_letter(ax, label, x=None, dx=-0.03):
        pos = ax.get_position()
        xx = x if x is not None else max(0.004, pos.x0 + dx)
        fig.text(xx, pos.y1 + 0.008, label, fontsize=20, fontweight="bold",
                 va="bottom")

    place_letter(ax_A, "A)", x=0.012)
    place_letter(axes_B[0], "B)", x=0.012)
    place_letter(ax_C, "C)", x=0.012)
    place_letter(ax_D, "D)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
