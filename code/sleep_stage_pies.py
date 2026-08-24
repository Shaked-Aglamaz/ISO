"""
Per-group sleep-stage pies (Wake / N1 / N2 / N3 / REM).

Slices are % of total recording time, group-mean of per-subject percentages.
Per-subject conversion:
    pct_n*  in subjects.csv is % of TST  -> multiplied by sleep_efficiency/100
    pct_wake in subjects.csv is % of recording -> used as-is
So the five slices sum to ~100 per subject, and the displayed mean is
mean-of-subject-percentages (independent of subject duration).

Reads the "subjects" tab live from Google Sheets (same pattern as
code/moca_correlation.py).

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/sleep_stage_pies.py
"""
from __future__ import annotations

from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from google.oauth2.service_account import Credentials
from scipy.stats import f_oneway, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd


SPREADSHEET_ID = "1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw"
SHEET_TAB = "subjects"
SA_KEY_PATH = Path("C:/Users/Shaked/.gcp/refined-spirit-494512-e8-059d94c0eb34.json")

OUTPUT_DIR = Path("results/demographics_V3")

GROUP_ORDER = ["YA", "HE", "MCI"]
GROUP_TITLES = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "Wake": "#BBBBBB",  # grey
    "N1":   "#CCBB44",  # yellow
    "N2":   "#228833",  # green
    "N3":   "#AA3377",  # purple
    "REM":  "#EE6677",  # red
}


def load_subjects_table() -> pd.DataFrame:
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    client = gspread.authorize(creds)
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TAB)
    df = pd.DataFrame(ws.get_all_records())
    needed = ["subject_id", "group",
              "pct_n1", "pct_n2", "pct_n3", "pct_rem", "pct_wake",
              "sleep_efficiency_pct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet missing columns: {missing}")
    for c in needed[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["subject_id", "group", *needed[2:]]].copy()


def per_subject_recording_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TST-relative %s to recording-relative %s so all five slices sum to ~100."""
    eff = df["sleep_efficiency_pct"] / 100.0
    out = pd.DataFrame({
        "subject_id": df["subject_id"],
        "group": df["group"],
        "Wake": df["pct_wake"],
        "N1":   df["pct_n1"]  * eff,
        "N2":   df["pct_n2"]  * eff,
        "N3":   df["pct_n3"]  * eff,
        "REM":  df["pct_rem"] * eff,
    })
    return out


def group_means(rec_pct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GROUP_ORDER:
        sub = rec_pct[rec_pct["group"] == g]
        rows.append({
            "group": g,
            "n": len(sub),
            **{stage: float(sub[stage].mean()) for stage in STAGES},
        })
    return pd.DataFrame(rows).set_index("group")


def _sig_marker(p: float, sig: bool) -> str:
    if not sig:
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _format_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def make_figure(means: pd.DataFrame, out_path: Path,
                stats: dict | None = None, orientation: str = "horizontal") -> None:
    if orientation == "vertical":
        # Pies stacked top-to-bottom (YA / HE / MCI), table beneath all three.
        fig = plt.figure(figsize=(7.5, 19))
        gs = fig.add_gridspec(
            4, 1, height_ratios=[1.0, 1.0, 1.0, 0.55],
            hspace=0.30,
        )
        pie_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        table_spec = gs[3, 0]
    else:
        # Original: pies in a row, table spanning beneath.
        fig = plt.figure(figsize=(15, 7.5))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[3.2, 1.0],
            hspace=0.05, wspace=0.05,
        )
        pie_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        table_spec = gs[1, :]
    colors = [STAGE_COLORS[s] for s in STAGES]

    for ax, group in zip(pie_axes, GROUP_ORDER):
        row = means.loc[group]
        sizes = [row[s] for s in STAGES]
        labels = [f"{s}\n{v:.1f}%" for s, v in zip(STAGES, sizes)]
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            labeldistance=1.12,
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 10},
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        ax.set_title(f"{GROUP_TITLES[group]} (n={int(row['n'])})",
                     fontsize=13, pad=14)
        ax.set_aspect("equal")

    if stats is not None:
        tax = fig.add_subplot(table_spec)
        tax.set_axis_off()

        pair_keys_canonical = [("YA", "HE"), ("YA", "MCI"), ("HE", "MCI")]
        pair_headers = ["Young vs Elderly", "Young vs MCI", "Elderly vs MCI"]
        col_labels = ["Stage", "Omnibus p", *pair_headers]

        cell_text = []
        cell_colors = []
        for stage in STAGES:
            s = stats[stage]
            row_vals = [stage, _format_p(s["p"])]
            row_colors = ["white", "white"]
            for (g1, g2) in pair_keys_canonical:
                # pairs dict may store the pair in either order; look up both
                pd_ = s["pairs"].get((g1, g2)) or s["pairs"].get((g2, g1))
                if pd_ is None:
                    row_vals.append("—")
                    row_colors.append("white")
                else:
                    marker = _sig_marker(pd_["p"], pd_["sig"])
                    row_vals.append(f"{marker}  ({_format_p(pd_['p'])})")
                    row_colors.append("#EAF4EA" if pd_["sig"] else "white")
            cell_text.append(row_vals)
            cell_colors.append(row_colors)

        table = tax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=["#F0F0F0"] * len(col_labels),
            cellLoc="center",
            loc="center",
            colWidths=[0.08, 0.12, 0.22, 0.22, 0.22],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.55)

        # Color the stage column cells with the stage swatch
        for i, stage in enumerate(STAGES, start=1):  # row 0 is the header
            cell = table[(i, 0)]
            cell.set_facecolor(STAGE_COLORS[stage])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

        # Remove the table outer frame
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#DDDDDD")

        tax.text(
            0.5, -0.18,
            "*** p<.001   ** p<.01   * p<.05   ns = not significant   |   "
            "post-hoc: Tukey HSD if all-normal, else Dunn (Holm)",
            ha="center", va="center", transform=tax.transAxes,
            fontsize=9, color="#555555",
        )

    fig.suptitle("Sleep stage distribution (% of total recording, group mean)",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _eta_squared_anova(groups: list[np.ndarray]) -> float:
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = sum(((np.concatenate(groups) - grand) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _eta_squared_kw(h: float, n_total: int, k: int) -> float:
    # Tomczak & Tomczak (2014): eta^2 = (H - k + 1) / (n - k)
    denom = n_total - k
    return float((h - k + 1) / denom) if denom > 0 else float("nan")


def run_stats(rec_pct: pd.DataFrame) -> dict:
    """Per-stage 3-group test: Shapiro -> ANOVA/KW -> Tukey HSD / Dunn-Holm."""
    out: dict = {}
    groups_data = {g: rec_pct[rec_pct["group"] == g] for g in GROUP_ORDER}

    for stage in STAGES:
        arrs = {g: groups_data[g][stage].dropna().to_numpy(dtype=float)
                for g in GROUP_ORDER}

        shapiro_p = {g: float(shapiro(arrs[g]).pvalue) for g in GROUP_ORDER}
        all_normal = all(p > 0.05 for p in shapiro_p.values())

        if all_normal:
            f_stat, p_omni = f_oneway(*arrs.values())
            test_name = "One-Way ANOVA"
            stat_label, stat_value = "F", float(f_stat)
            eta2 = _eta_squared_anova(list(arrs.values()))
        else:
            h_stat, p_omni = kruskal(*arrs.values())
            test_name = "Kruskal-Wallis"
            stat_label, stat_value = "H", float(h_stat)
            n_total = sum(len(a) for a in arrs.values())
            eta2 = _eta_squared_kw(h_stat, n_total, k=len(arrs))

        pairs: dict = {}
        if p_omni < 0.05:
            combined = pd.concat(
                [pd.DataFrame({"value": v, "group": g}) for g, v in arrs.items()],
                ignore_index=True,
            )
            if all_normal:
                tukey = pairwise_tukeyhsd(combined["value"], combined["group"], alpha=0.05)
                posthoc_name = "Tukey HSD"
                for k in range(len(tukey.pvalues)):
                    g1 = str(tukey.groupsunique[tukey._multicomp.pairindices[0][k]])
                    g2 = str(tukey.groupsunique[tukey._multicomp.pairindices[1][k]])
                    pairs[(g1, g2)] = {
                        "p": float(tukey.pvalues[k]),
                        "sig": bool(tukey.reject[k]),
                        "mean_diff": float(tukey.meandiffs[k]),
                    }
            else:
                dunn = sp.posthoc_dunn(combined, val_col="value",
                                       group_col="group", p_adjust="holm")
                posthoc_name = "Dunn (Holm)"
                for i in range(len(GROUP_ORDER)):
                    for j in range(i + 1, len(GROUP_ORDER)):
                        g1, g2 = GROUP_ORDER[i], GROUP_ORDER[j]
                        p_val = float(dunn.loc[g1, g2])
                        pairs[(g1, g2)] = {
                            "p": p_val,
                            "sig": p_val < 0.05,
                            "median_diff": float(np.median(arrs[g1]) - np.median(arrs[g2])),
                        }
        else:
            posthoc_name = None

        out[stage] = {
            "shapiro_p": shapiro_p,
            "all_normal": all_normal,
            "test": test_name,
            "stat_label": stat_label,
            "stat_value": stat_value,
            "p": float(p_omni),
            "eta2": eta2,
            "posthoc_name": posthoc_name,
            "pairs": pairs,
        }

    return out


def write_stats_report(stats: dict, means: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("Sleep stage proportions — 3-group comparison")
    lines.append("=" * 60)
    lines.append("Per-subject value: % of total recording (Wake as-is; "
                 "N1/N2/N3/REM = TST-relative pct * sleep_efficiency/100).")
    lines.append("Pipeline: Shapiro-Wilk -> One-Way ANOVA or Kruskal-Wallis "
                 "-> Tukey HSD or Dunn (Holm) if omnibus p<0.05.")
    lines.append("")
    lines.append(f"N per group: " + ", ".join(
        f"{g}={int(means.loc[g, 'n'])}" for g in GROUP_ORDER))
    lines.append("")

    for stage in STAGES:
        s = stats[stage]
        lines.append(f"--- {stage} ---")
        lines.append("Group means: " + ", ".join(
            f"{g}={means.loc[g, stage]:.2f}%" for g in GROUP_ORDER))
        lines.append("Shapiro p:   " + ", ".join(
            f"{g}={s['shapiro_p'][g]:.3f}" for g in GROUP_ORDER)
            + ("  (all normal)" if s["all_normal"] else "  (non-normal -> KW)"))
        lines.append(f"Omnibus:     {s['test']}  "
                     f"{s['stat_label']}={s['stat_value']:.3f}, "
                     f"p={s['p']:.4f}, eta2={s['eta2']:.3f}")
        if s["posthoc_name"]:
            lines.append(f"Post-hoc:    {s['posthoc_name']}")
            for (g1, g2), pd_ in s["pairs"].items():
                tag = "*" if pd_["sig"] else "ns"
                lines.append(f"  {g1} vs {g2}: p={pd_['p']:.4f} ({tag})")
        else:
            lines.append("Post-hoc:    not run (omnibus ns)")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_subjects_table()
    rec_pct = per_subject_recording_pct(df)
    means = group_means(rec_pct)

    print("Group-mean stage % (of total recording):")
    print(means[["n", *STAGES]].round(2).to_string())
    print(f"\nRow sums (sanity, should be ~100): "
          f"{means[STAGES].sum(axis=1).round(2).to_dict()}")

    means.to_csv(OUTPUT_DIR / "sleep_stage_means.csv")

    stats = run_stats(rec_pct)
    stats_path = OUTPUT_DIR / "sleep_stage_stats.txt"
    write_stats_report(stats, means, stats_path)

    out_path = OUTPUT_DIR / "sleep_stage_pies.png"
    make_figure(means, out_path, stats=stats, orientation="horizontal")

    out_path_vertical = OUTPUT_DIR / "sleep_stage_pies_vertical.png"
    make_figure(means, out_path_vertical, stats=stats, orientation="vertical")

    print("\nOmnibus p-values:")
    for stage in STAGES:
        s = stats[stage]
        print(f"  {stage:<5} {s['test']:<16} "
              f"{s['stat_label']}={s['stat_value']:6.2f}  "
              f"p={s['p']:.4f}  eta2={s['eta2']:.3f}")
        for (g1, g2), pd_ in s["pairs"].items():
            tag = "*" if pd_["sig"] else "ns"
            print(f"      {g1} vs {g2}: p={pd_['p']:.4f} ({tag})")

    print(f"\nSaved: {out_path}")
    print(f"Saved: {out_path_vertical}")
    print(f"Saved: {OUTPUT_DIR / 'sleep_stage_means.csv'}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
