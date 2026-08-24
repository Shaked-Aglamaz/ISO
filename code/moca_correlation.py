"""
Exploratory correlation: ISFS scalars vs MoCA, pooled across all subjects with a MoCA score.

Reduction per subject (NaN-aware mean over channels, matching step6 violins / topo_aggregation.py):
    peak_freq_whole : nanmean over all channels of peak_frequency
    bw_whole        : nanmean over all channels of bandwidth
    auc_whole       : nanmean over all channels of auc
    auc_roi         : nanmean over EXTENDED_CENTRAL_PARIETAL_ROI of auc

MoCA is read live from the Google Sheet "subjects" tab — no local CSV mirror.

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/moca_correlation.py
"""
from __future__ import annotations

from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from scipy.stats import pearsonr, spearmanr

from utils.config import EXTENDED_CENTRAL_PARIETAL_ROI, GROUP_COLORS_DARK, group_label

from step4_distribution_analysis import load_subject_data


SPREADSHEET_ID = "1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw"
SHEET_TAB = "subjects"
SA_KEY_PATH = Path("C:/Users/Shaked/.gcp/refined-spirit-494512-e8-059d94c0eb34.json")

# Source-of-truth per-channel ISFS data is the sigma-fix (V9) re-run, matching
# the rest of the paper (Results 4.3-4.6 / three_groups_V9). The older new_*_results
# dirs are pre-sigma-fix and differ for several subjects.
GROUP_DIR_TO_RESULTS = {
    "control_clean":         Path("results/sigma_fix_YA"),
    "elderly_control_clean": Path("results/sigma_fix_HE"),
    "MCI_clean":             Path("results/sigma_fix_MCI"),
}

OUTPUT_DIR = Path("results/moca_correlation_V3")

GROUP_COLORS = {"YA": "#4477AA", "HE": "#4477AA", "MCI": "#CC3311"}

# paper_style palette: the group colours used by every other paper figure.
# NOTE the exploratory GROUP_COLORS above has elderly BLUE and MCI RED, which is
# the opposite of Figures 3-5 (elderly red, MCI blue). Only paper_style is
# corrected; the exploratory default is left as it was.
PAPER_GROUP_COLORS = GROUP_COLORS_DARK               # Young/Elderly/MCI by name
PAPER_GROUP_LABELS = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}

SCALAR_PANELS = [
    ("peak_freq_whole", "Peak frequency (whole-scalp)", "Hz"),
    ("bw_whole",        "Bandwidth (whole-scalp)",      "Hz"),
    ("auc_whole",       "AUC (whole-scalp)",            "AU"),
    ("auc_roi",         "AUC (extended ROI, 36-ch)",    "AU"),
]

# Reader-facing panel labels: there is just "the ROI" in the manuscript, never
# "extended" and never an electrode count (manifest, ROI naming rule).
PAPER_PANEL_LABELS = {
    "peak_freq_whole": "Peak frequency (whole-scalp)",
    "bw_whole":        "Bandwidth (whole-scalp)",
    "auc_whole":       "ISFS strength (whole-scalp)",
    "auc_roi":         "ISFS strength (central-parietal ROI)",
}

# Fonts sized so the grid stays legible after it is shrunk to page width.
S2_FS_TITLE = 17
S2_FS_STATS = 13
S2_FS_LABEL = 15
S2_FS_TICK = 13
S2_FS_LEGEND = 13


def load_moca_table() -> pd.DataFrame:
    """Read the subjects tab live from Google Sheets, return rows with non-blank MoCA."""
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    client = gspread.authorize(creds)
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TAB)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    keep = ["subject_id", "group", "group_dir", "moca"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet missing expected columns: {missing}. Got: {list(df.columns)}")
    df = df[keep].copy()
    df["moca"] = pd.to_numeric(df["moca"], errors="coerce")
    df = df.dropna(subset=["moca"]).reset_index(drop=True)
    return df


def per_subject_scalars(subject_id: str, group_dir: str) -> dict[str, float]:
    """Compute the four scalars for one subject; NaN if data unavailable."""
    results_dir = GROUP_DIR_TO_RESULTS.get(group_dir)
    if results_dir is None:
        return _all_nan()
    df = load_subject_data(subject_id, dir_path=str(results_dir))
    if df is None or df.empty:
        return _all_nan()

    roi_set = set(EXTENDED_CENTRAL_PARIETAL_ROI)
    roi_mask = df["channel"].isin(roi_set)
    return {
        "peak_freq_whole": _nanmean(df["peak_frequency"]),
        "bw_whole":        _nanmean(df["bandwidth"]),
        "auc_whole":       _nanmean(df["auc"]),
        "auc_roi":         _nanmean(df.loc[roi_mask, "auc"]),
    }


def _nanmean(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _all_nan() -> dict[str, float]:
    return {k: float("nan") for k in ("peak_freq_whole", "bw_whole", "auc_whole", "auc_roi")}


def build_subject_table(moca_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in moca_df.iterrows():
        scalars = per_subject_scalars(r["subject_id"], r["group_dir"])
        rows.append({
            "subject_id": r["subject_id"],
            "group":      r["group"],
            "group_dir":  r["group_dir"],
            "moca":       float(r["moca"]),
            **scalars,
        })
    return pd.DataFrame(rows)


def correlation_summary(subj_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col, label, _unit in SCALAR_PANELS:
        valid = subj_df[["moca", col]].dropna()
        n = len(valid)
        if n < 3:
            out.append({"scalar": col, "label": label, "n": n,
                        "pearson_r": np.nan, "pearson_p": np.nan,
                        "spearman_r": np.nan, "spearman_p": np.nan})
            continue
        pr, pp = pearsonr(valid["moca"], valid[col])
        sr, sp = spearmanr(valid["moca"], valid[col])
        out.append({"scalar": col, "label": label, "n": n,
                    "pearson_r": pr, "pearson_p": pp,
                    "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(out)


def plot_grid(subj_df: pd.DataFrame, summary: pd.DataFrame, out_path: Path,
              suptitle: str = "ISFS scalars vs MoCA (HE+MCI pooled)",
              ols_label: str = "linear fit", paper_style: bool = False) -> None:
    """2x2 grid of ISFS scalar vs MoCA scatters with a pooled linear fit.

    ``paper_style`` renders the manuscript version of Figure S2: the shared
    group palette, reader-facing panel labels, page-legible fonts, the
    correlation statistics on their own line above each panel instead of
    crammed into a small title, and no figure title (it lives in the caption).
    """
    figsize = (9.5, 9.6) if paper_style else (13, 10)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()
    for ax, (col, label, unit) in zip(axes_flat, SCALAR_PANELS):
        panel_label = PAPER_PANEL_LABELS.get(col, label) if paper_style else label
        valid = subj_df[["moca", col, "group"]].dropna()
        for grp, sub in valid.groupby("group"):
            if paper_style:
                # PAPER_GROUP_LABELS is the palette key, group_label() is what
                # the legend prints, so relabelling MCI -> aMCI cannot silently
                # turn the series grey.
                key = PAPER_GROUP_LABELS.get(grp, grp)
                color = PAPER_GROUP_COLORS.get(key, "gray")
                series_label = f"{group_label(key)} (n={len(sub)})"
            else:
                color = GROUP_COLORS.get(grp, "gray")
                series_label = f"{grp} (n={len(sub)})"
            ax.scatter(sub["moca"], sub[col], color=color, label=series_label,
                       s=70 if paper_style else 55, alpha=0.85,
                       edgecolor="k", linewidth=0.5)
        # Pooled regression line
        if len(valid) >= 2:
            x = valid["moca"].to_numpy(dtype=float)
            y = valid[col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", linestyle="--",
                    linewidth=1.6 if paper_style else 1.2,
                    label="Linear fit" if paper_style else ols_label)
        row = summary.loc[summary["scalar"] == col].iloc[0]

        if paper_style:
            ax.set_title(panel_label, fontsize=S2_FS_TITLE, fontweight="bold", pad=30)
            # Statistics on their own line just above the axes: readable, and it
            # cannot collide with the data the way an in-axes box would.
            ax.text(0.5, 1.015,
                    f"r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.2f}    "
                    f"ρ = {row['spearman_r']:.3f}, p = {row['spearman_p']:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=S2_FS_STATS)
            ax.set_xlabel("MoCA score", fontsize=S2_FS_LABEL)
            ax.set_ylabel(f"{panel_label.split(' (')[0]} ({unit})", fontsize=S2_FS_LABEL)
            ax.tick_params(labelsize=S2_FS_TICK)
            # No per-panel legend: all four panels share the same series, and a
            # 'best'-placed box lands on top of data points in every panel. One
            # shared legend is added below the grid instead.
        else:
            ax.set_title(
                f"{label}\n"
                f"Pearson r={row['pearson_r']:.3f}, p={row['pearson_p']:.3g}   |   "
                f"Spearman ρ={row['spearman_r']:.3f}, p={row['spearman_p']:.3g}   |   N={int(row['n'])}",
                fontsize=10,
            )
            ax.set_xlabel("MoCA")
            ax.set_ylabel(f"{label} ({unit})")
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    if not paper_style:
        fig.suptitle(suptitle, fontsize=13, y=1.00)
        fig.tight_layout()
    else:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=S2_FS_LEGEND, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=300 if paper_style else 150, bbox_inches="tight")
    plt.close(fig)


def print_summary(subj_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 80)
    print(f"MoCA-bearing subjects loaded: {len(subj_df)}")
    print(f"  by group: " + ", ".join(f"{g}={n}" for g, n in subj_df["group"].value_counts().items()))
    print("=" * 80)
    print()
    print(f"{'Scalar':<18} {'Label':<35} {'N':>4} {'Pearson r':>11} {'p':>9} {'Spearman ρ':>12} {'p':>9}")
    print("-" * 100)
    for _, r in summary.iterrows():
        print(f"{r['scalar']:<18} {r['label']:<35} {int(r['n']):>4} "
              f"{r['pearson_r']:>11.4f} {r['pearson_p']:>9.4g} "
              f"{r['spearman_r']:>12.4f} {r['spearman_p']:>9.4g}")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading MoCA from Google Sheet...")
    moca_df = load_moca_table()
    print(f"  → {len(moca_df)} subjects with non-blank MoCA")

    print("Loading per-subject ISFS scalars...")
    subj_df = build_subject_table(moca_df)

    summary = correlation_summary(subj_df)

    subj_path = OUTPUT_DIR / "subject_scalars.csv"
    summary_path = OUTPUT_DIR / "correlation_summary.csv"
    fig_path = OUTPUT_DIR / "moca_correlations_grid.png"

    subj_df.to_csv(subj_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_grid(subj_df, summary, fig_path)

    print_summary(subj_df, summary)
    print(f"Wrote: {subj_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {fig_path}")

    # MCI-only correlations (controls excluded). Saved under separate names so the
    # pooled HE+MCI outputs (S2) are not overwritten.
    print("\n" + "=" * 80)
    print("MCI-ONLY correlations (controls excluded)")
    mci_df = subj_df[subj_df["group"] == "MCI"].copy()
    mci_summary = correlation_summary(mci_df)
    mci_subj_path = OUTPUT_DIR / "subject_scalars_mci_only.csv"
    mci_summary_path = OUTPUT_DIR / "correlation_summary_mci_only.csv"
    mci_fig_path = OUTPUT_DIR / "moca_correlations_grid_mci_only.png"
    mci_df.to_csv(mci_subj_path, index=False)
    mci_summary.to_csv(mci_summary_path, index=False)
    plot_grid(mci_df, mci_summary, mci_fig_path,
              suptitle="ISFS scalars vs MoCA (MCI only)", ols_label="linear fit")
    print_summary(mci_df, mci_summary)
    print(f"Wrote: {mci_subj_path}")
    print(f"Wrote: {mci_summary_path}")
    print(f"Wrote: {mci_fig_path}")


if __name__ == "__main__":
    main()
