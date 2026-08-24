"""Find young-adult channels with a fitted ISFS peak in [1.4, 1.7] AU and a flat off-peak FFT.

Replots the selected channels with a tall (W<H) figure and ylim=(-0.5, 2.5),
saving to results/new_iso_results/example_gaussians/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("results/new_iso_results")
OUT_DIR = ROOT / "example_gaussians"
EXCLUDE = {"a_excluded", "a_group_plots_V1", "a_group_plots_V2", "sigma_boxplot",
           "example_gaussians"}

PEAK_LO, PEAK_HI = 1.4, 1.7
TOP_N = 20                       # only save the cleanest N
MU_RANGE = (0.0075, 0.04)        # canonical ISFS band for the Gaussian centre
BLUE_MAX_LIMIT = 2.4             # blue line must fit under ylim 2.5


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def baseline_corrected(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    freq = df["frequency"].to_numpy()
    mp = df["mean_power"].to_numpy()
    mask = (freq > 0.06) & (freq < 0.102)
    return freq, mp - np.nanmean(mp[mask])


def flatness(freq: np.ndarray, shifted: np.ndarray, mu: float, sigma: float) -> float:
    """Max |shifted| outside ±2σ of μ, divided by peak (the lower the cleaner)."""
    off_peak = (freq < mu - 2 * sigma) | (freq > mu + 2 * sigma)
    if not off_peak.any():
        return np.inf
    return np.nanmax(np.abs(shifted[off_peak])) / np.nanmax(shifted)


def find_candidates() -> list[dict]:
    rows = []
    for sub_dir in ROOT.iterdir():
        if not sub_dir.is_dir() or sub_dir.name in EXCLUDE:
            continue
        sub = sub_dir.name
        sum_csv = sub_dir / f"{sub}_all_channels_summary.csv"
        if not sum_csv.exists():
            continue
        try:
            summary = pd.read_csv(sum_csv, comment="#")
        except Exception:
            continue
        summary = summary[summary["peak_amplitude"].notna()]
        summary = summary[(summary["peak_amplitude"] >= PEAK_LO)
                          & (summary["peak_amplitude"] <= PEAK_HI)
                          & (summary["peak_frequency"] >= MU_RANGE[0])
                          & (summary["peak_frequency"] <= MU_RANGE[1])]
        for _, r in summary.iterrows():
            ch = r["channel"]
            ch_csv = sub_dir / f"{sub}_{ch}_output" / f"{sub}_{ch}_spectral_power.csv"
            if not ch_csv.exists():
                continue
            df = pd.read_csv(ch_csv)
            freq, shifted = baseline_corrected(df)
            sigma = r["bandwidth_sigma"]
            mu = r["peak_frequency"]
            f_score = flatness(freq, shifted, mu, sigma)
            rows.append(dict(
                subject=sub, channel=ch,
                fit_peak=r["peak_amplitude"], mu=mu, sigma=sigma,
                bandwidth=r["bandwidth"], auc=r["auc"],
                blue_max=float(np.nanmax(shifted)),
                flatness=f_score,
            ))
    return rows


def replot(rec: dict, df: pd.DataFrame) -> Path:
    sub, ch = rec["subject"], rec["channel"]
    freq, shifted = baseline_corrected(df)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.plot(freq, shifted, color="#2E86AB", linewidth=2.5,
            label="Mean Relative Power (Baseline Corrected)", zorder=2)

    a, mu, sigma = rec["fit_peak"], rec["mu"], rec["sigma"]
    x_fit = np.linspace(freq[0], freq[-1], 500)
    y_fit = gaussian(x_fit, a, mu, sigma)
    ax.plot(x_fit, y_fit, "#A23B72", linewidth=2.5, alpha=0.9,
            label=f"Gaussian Fit (μ={mu:.4f} Hz, σ={sigma:.4f})", zorder=3)
    ax.plot(mu, a, "o", color="#A23B72", markersize=10,
            label=f"Peak: {a:.3f} AU", zorder=4)

    x1, x2 = mu - sigma, mu + sigma
    bw_h = gaussian(x1, a, mu, sigma)
    ax.hlines(bw_h, x1, x2, colors="#F18F01", linewidth=3,
              label=f"Bandwidth: {2*sigma:.4f} Hz", zorder=3)
    mask = (x_fit >= x1) & (x_fit <= x2)
    ax.fill_between(x_fit[mask], y_fit[mask], color="#A23B72", alpha=0.2,
                    label=f"±1σ Area (AUC={rec['auc']:.3f})", zorder=1)

    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, 0.10)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative Power (AU)", fontsize=12)
    ax.set_title(f"{sub} — {ch}", fontsize=13)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{sub}_{ch}_mean_spectrum.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    cands = find_candidates()
    print(f"Channels with fit peak in [{PEAK_LO}, {PEAK_HI}] AU: {len(cands)}")
    if not cands:
        return

    cands = [c for c in cands if c["blue_max"] <= BLUE_MAX_LIMIT]
    cands.sort(key=lambda r: r["flatness"])
    selected = cands[:TOP_N]
    print(f"Filtered by blue_max ≤ {BLUE_MAX_LIMIT}: {len(cands)} remain")
    print(f"Saving top {len(selected)} by flatness:\n")

    print(f"{'subj':>10s} {'ch':>5s} {'fit':>5s} {'flat':>5s} {'mu':>7s} {'sigma':>7s}")
    for r in selected:
        print(f"{r['subject']:>10s} {r['channel']:>5s} "
              f"{r['fit_peak']:5.2f} {r['flatness']:5.2f} "
              f"{r['mu']:7.4f} {r['sigma']:7.4f}")

    for r in selected:
        ch_csv = ROOT / r["subject"] / f"{r['subject']}_{r['channel']}_output" / \
                 f"{r['subject']}_{r['channel']}_spectral_power.csv"
        df = pd.read_csv(ch_csv)
        out = replot(r, df)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
