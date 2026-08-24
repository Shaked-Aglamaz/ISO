"""
Per-group N2 bout summary table + 3-group statistical comparison.

Bout definition (matches the project's ISFS pipeline,
code/new_iso/mult_chan.extract_clean_sleep_bouts):
  - NREM2 segments split around BAD annotations
  - Adjacent / overlapping segments merged
  - Filtered by minimum duration MIN_BOUT_SEC (300 s)

Note: the reference paper used 280 s; we use 300 s to stay consistent with
every downstream ISFS analysis in this repo.

Per-subject metrics (paper definitions):
  - Bout count
  - Mean Bout Length (s)
  - Total Duration of Bouts (min)
  - Mean Relative Location (%): mean(bout-midpoint / recording_length)
  - Proportion of all N2 (%): total_bout_duration / total_NREM2_duration

Reads the "subjects" tab live (recording length, group assignment) and walks
each subject's cleaned_annotations.txt for the bout positions.

Outputs (results/demographics_V1/):
  - n2_bouts_per_subject.csv    one row per subject, all five metrics
  - n2_bouts_table.csv          per-group means/SDs + omnibus + post-hoc
  - n2_bouts_table.txt          plain-text report with categorization detail
  - n2_bouts_table.png          rendered comparison table

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/n2_bouts_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from google.oauth2.service_account import Credentials
from scipy.stats import f_oneway, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "utils"))
from subject_summary import parse_annotations  # noqa: E402


SPREADSHEET_ID = "1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw"
SHEET_TAB = "subjects"
SA_KEY_PATH = Path("C:/Users/Shaked/.gcp/refined-spirit-494512-e8-059d94c0eb34.json")

BASE_DIR = Path("I:/Shaked/ISO_data")
OUTPUT_DIR = Path("results/demographics_V3")

GROUP_ORDER = ["YA", "HE", "MCI"]
GROUP_TITLES = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}

MIN_BOUT_SEC = 300.0  # paper used 280; we use 300 to match the ISFS pipeline

METRICS = [
    ("bout_count",       "Bout count",                       2),
    ("mean_bout_len",    "Mean Bout Length (s)",             1),
    ("total_dur_min",    "Total Duration of Bouts (min)",    1),
    ("mean_rel_loc_pct", "Mean Relative Location (%)",       1),
    ("prop_of_n2_pct",   "Proportion of all N2 (%)",         1),
]


def load_subjects() -> pd.DataFrame:
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    client = gspread.authorize(creds)
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TAB)
    df = pd.DataFrame(ws.get_all_records())
    for c in ("recording_sec", "n2_sec"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["subject_id", "group", "group_dir",
               "recording_sec", "n2_sec"]].copy()


def _split_around_bads(seg_start: float, seg_end: float,
                       bads: list[tuple[float, float]]) -> list[tuple[float, float]]:
    overlapping = [(bs, be) for bs, be in bads
                   if not (be <= seg_start or bs >= seg_end)]
    if not overlapping:
        return [(seg_start, seg_end)]
    overlapping.sort(key=lambda x: x[0])
    out, cur = [], seg_start
    for bs, be in overlapping:
        if cur < bs:
            out.append((cur, bs))
        cur = max(cur, be)
    if cur < seg_end:
        out.append((cur, seg_end))
    return out


def extract_bouts_from_annotations(annot_path: Path,
                                   min_dur: float = MIN_BOUT_SEC
                                   ) -> list[tuple[float, float]]:
    """Replicates extract_clean_sleep_bouts using only annotation timing."""
    anns = parse_annotations(str(annot_path))
    n2_segments = []
    bad_segments = []
    for onset, dur, desc in anns:
        if desc in ("NREM2", "N2"):
            n2_segments.append((onset, onset + dur))
        elif desc.upper().startswith("BAD"):
            bad_segments.append((onset, onset + dur))

    clean = []
    for s, e in n2_segments:
        clean.extend(_split_around_bads(s, e, bad_segments))

    if not clean:
        return []
    clean.sort(key=lambda x: x[0])
    merged = [clean[0]]
    for s, e in clean[1:]:
        if s <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) >= min_dur]


def annotation_path_for(subject_id: str, group_dir: str) -> Path | None:
    base = BASE_DIR / group_dir / subject_id
    for name in (f"{subject_id}_cleaned_annotations.txt",
                 f"{subject_id}_annotations.txt"):
        p = base / name
        if p.exists():
            return p
    return None


def per_subject_metrics(row) -> dict:
    annot = annotation_path_for(row["subject_id"], row["group_dir"])
    out = {"subject_id": row["subject_id"],
           "group": row["group"],
           **{k: float("nan") for k, _, _ in METRICS}}
    if annot is None:
        return out

    bouts = extract_bouts_from_annotations(annot)
    if not bouts:
        out["bout_count"] = 0
        out["total_dur_min"] = 0.0
        return out

    durations = np.array([e - s for s, e in bouts], dtype=float)
    midpoints = np.array([(s + e) / 2.0 for s, e in bouts], dtype=float)
    rec_len = float(row["recording_sec"]) if np.isfinite(row["recording_sec"]) else float("nan")
    n2_total = float(row["n2_sec"]) if np.isfinite(row["n2_sec"]) else float("nan")

    out["bout_count"] = float(len(bouts))
    out["mean_bout_len"] = float(durations.mean())
    out["total_dur_min"] = float(durations.sum() / 60.0)
    if rec_len and rec_len > 0:
        out["mean_rel_loc_pct"] = float(np.mean(midpoints / rec_len) * 100)
    if n2_total and n2_total > 0:
        out["prop_of_n2_pct"] = float(durations.sum() / n2_total * 100)
    return out


# ---------- stats ----------

def _eta_squared_anova(groups: list[np.ndarray]) -> float:
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = float(((np.concatenate(groups) - grand) ** 2).sum())
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def _eta_squared_kw(h: float, n_total: int, k: int) -> float:
    return float((h - k + 1) / (n_total - k)) if (n_total - k) > 0 else float("nan")


def run_metric_stats(df_sub: pd.DataFrame, metric: str) -> dict:
    arrs = {g: df_sub.loc[df_sub["group"] == g, metric].dropna().to_numpy(dtype=float)
            for g in GROUP_ORDER}
    shapiro_p = {g: float(shapiro(arrs[g]).pvalue) if len(arrs[g]) >= 3 else float("nan")
                 for g in GROUP_ORDER}
    all_normal = all(np.isfinite(p) and p > 0.05 for p in shapiro_p.values())

    if all_normal:
        f_stat, p_omni = f_oneway(*arrs.values())
        test_name, stat_label, stat_value = "ANOVA", "F", float(f_stat)
        eta2 = _eta_squared_anova(list(arrs.values()))
    else:
        h_stat, p_omni = kruskal(*arrs.values())
        test_name, stat_label, stat_value = "Kruskal-Wallis", "H", float(h_stat)
        n_total = sum(len(a) for a in arrs.values())
        eta2 = _eta_squared_kw(h_stat, n_total, k=len(arrs))

    pairs: dict = {}
    if np.isfinite(p_omni) and p_omni < 0.05:
        combined = pd.concat(
            [pd.DataFrame({"value": v, "group": g}) for g, v in arrs.items()],
            ignore_index=True,
        )
        if all_normal:
            tukey = pairwise_tukeyhsd(combined["value"], combined["group"], alpha=0.05)
            posthoc = "Tukey HSD"
            for k in range(len(tukey.pvalues)):
                g1 = str(tukey.groupsunique[tukey._multicomp.pairindices[0][k]])
                g2 = str(tukey.groupsunique[tukey._multicomp.pairindices[1][k]])
                pairs[(g1, g2)] = {"p": float(tukey.pvalues[k]),
                                   "sig": bool(tukey.reject[k]),
                                   "diff": float(tukey.meandiffs[k])}
        else:
            dunn = sp.posthoc_dunn(combined, val_col="value",
                                   group_col="group", p_adjust="holm")
            posthoc = "Dunn (Holm)"
            for i in range(len(GROUP_ORDER)):
                for j in range(i + 1, len(GROUP_ORDER)):
                    g1, g2 = GROUP_ORDER[i], GROUP_ORDER[j]
                    p_val = float(dunn.loc[g1, g2])
                    pairs[(g1, g2)] = {"p": p_val,
                                       "sig": p_val < 0.05,
                                       "diff": float(np.median(arrs[g1])
                                                     - np.median(arrs[g2]))}
    else:
        posthoc = None

    return {
        "shapiro_p": shapiro_p,
        "all_normal": all_normal,
        "test": test_name,
        "stat_label": stat_label,
        "stat_value": stat_value,
        "p": float(p_omni) if np.isfinite(p_omni) else float("nan"),
        "eta2": eta2,
        "posthoc_name": posthoc,
        "pairs": pairs,
    }


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


# ---------- rendering ----------

def render_table_png(per_subj: pd.DataFrame, stats: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(13, 3.6))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    pair_keys_canonical = [("YA", "HE"), ("YA", "MCI"), ("HE", "MCI")]
    pair_headers = ["Y vs E", "Y vs MCI", "E vs MCI"]

    header = ["Variable", *(GROUP_TITLES[g] for g in GROUP_ORDER),
              "Omnibus p", *pair_headers]

    cell_text = [header]
    cell_colors = [["#E5E5E5"] * len(header)]
    bold_cells: list[tuple[int, int]] = []

    for key, label, decimals in METRICS:
        row_vals = [label]
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            if len(vals) == 0:
                row_vals.append("—")
            else:
                m, s = vals.mean(), vals.std(ddof=1)
                row_vals.append(f"{m:.{decimals}f} ± {s:.{decimals}f}")

        s = stats[key]
        row_vals.append(_format_p(s["p"]))

        for (g1, g2) in pair_keys_canonical:
            pd_ = s["pairs"].get((g1, g2)) or s["pairs"].get((g2, g1))
            if pd_ is None:
                row_vals.append("—")
            else:
                row_vals.append(_format_p(pd_["p"]))

        cell_text.append(row_vals)
        cell_colors.append(["white"] * len(header))

        # Mark significant cells for bolding (omnibus + each pair)
        row_idx = len(cell_text) - 1
        if np.isfinite(s["p"]) and s["p"] < 0.05:
            bold_cells.append((row_idx, len(GROUP_ORDER) + 1))  # omnibus column
        for col_offset, (g1, g2) in enumerate(pair_keys_canonical):
            pd_ = s["pairs"].get((g1, g2)) or s["pairs"].get((g2, g1))
            if pd_ is not None and pd_["sig"]:
                bold_cells.append((row_idx,
                                   len(GROUP_ORDER) + 2 + col_offset))

    col_widths = [0.26, 0.12, 0.12, 0.12, 0.10, 0.09, 0.09, 0.09]
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    # Left-align variable column
    for i in range(len(cell_text)):
        table[(i, 0)].get_text().set_horizontalalignment("left")
    # Header bold
    for j in range(len(header)):
        table[(0, j)].get_text().set_fontweight("bold")
    # Bold significant cells
    for (i, j) in bold_cells:
        table[(i, j)].get_text().set_fontweight("bold")
        table[(i, j)].set_facecolor("#EAF4EA")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDDDDD")

    fig.suptitle("N2 bout properties — group comparison", fontsize=14, y=0.96)
    footer = (
        f"Bout = clean NREM2 segment ≥ {int(MIN_BOUT_SEC)} s, "
        "split around BAD epochs. Values are mean ± SD.\n"
        "Omnibus: ANOVA if all groups normal, else Kruskal-Wallis. "
        "Post-hoc: Tukey HSD or Dunn (Holm). Bold = p<.05."
    )
    fig.text(
        0.5, 0.06, footer,
        ha="center", va="center", fontsize=10.5, color="#555555",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_text_report(per_subj: pd.DataFrame, stats: dict, path: Path) -> None:
    lines = []
    lines.append("N2 bout properties — group comparison")
    lines.append("=" * 60)
    lines.append(f"Bout = NREM2 ≥ {int(MIN_BOUT_SEC)} s after BAD-split + merge.")
    lines.append("")
    for key, label, decimals in METRICS:
        s = stats[key]
        lines.append(f"--- {label} ---")
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            n = len(vals)
            if n == 0:
                lines.append(f"  {GROUP_TITLES[g]}: n=0")
                continue
            lines.append(f"  {GROUP_TITLES[g]:<7} (n={n}): "
                         f"{vals.mean():.{decimals}f} ± {vals.std(ddof=1):.{decimals}f}  "
                         f"(median={vals.median():.{decimals}f})")
        lines.append(f"  Shapiro p: " + ", ".join(
            f"{g}={s['shapiro_p'][g]:.3f}" for g in GROUP_ORDER)
            + ("  (all normal)" if s["all_normal"] else "  (non-normal -> KW)"))
        lines.append(f"  Omnibus:   {s['test']}  "
                     f"{s['stat_label']}={s['stat_value']:.3f}, "
                     f"p={s['p']:.4f}, eta2={s['eta2']:.3f}")
        if s["posthoc_name"]:
            lines.append(f"  Post-hoc:  {s['posthoc_name']}")
            for (g1, g2), pd_ in s["pairs"].items():
                tag = "*" if pd_["sig"] else "ns"
                lines.append(f"    {g1} vs {g2}: p={pd_['p']:.4f} ({tag})")
        else:
            lines.append("  Post-hoc:  not run (omnibus ns)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_subjects()
    print(f"Loaded {len(df)} subjects from sheet.")

    rows = []
    missing = []
    for _, r in df.iterrows():
        m = per_subject_metrics(r)
        if annotation_path_for(r["subject_id"], r["group_dir"]) is None:
            missing.append(r["subject_id"])
        rows.append(m)
    per_subj = pd.DataFrame(rows)

    if missing:
        print(f"WARNING: missing annotation file for {len(missing)} subjects: "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subj.to_csv(OUTPUT_DIR / "n2_bouts_per_subject.csv", index=False)

    stats = {key: run_metric_stats(per_subj, key) for key, _, _ in METRICS}

    # Wide CSV summary
    summary_rows = []
    for key, label, decimals in METRICS:
        row = {"metric": label}
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            row[f"{g}_n"]    = int(len(vals))
            row[f"{g}_mean"] = round(vals.mean(), decimals + 1) if len(vals) else ""
            row[f"{g}_std"]  = round(vals.std(ddof=1), decimals + 1) if len(vals) > 1 else ""
        s = stats[key]
        row["omnibus_test"]  = s["test"]
        row["omnibus_stat"]  = round(s["stat_value"], 3)
        row["omnibus_p"]     = round(s["p"], 4) if np.isfinite(s["p"]) else ""
        row["eta_squared"]   = round(s["eta2"], 3) if np.isfinite(s["eta2"]) else ""
        for (g1, g2) in [("YA", "HE"), ("YA", "MCI"), ("HE", "MCI")]:
            pd_ = s["pairs"].get((g1, g2)) or s["pairs"].get((g2, g1))
            row[f"p_{g1}_vs_{g2}"] = round(pd_["p"], 4) if pd_ else ""
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "n2_bouts_table.csv", index=False)

    write_text_report(per_subj, stats, OUTPUT_DIR / "n2_bouts_table.txt")
    render_table_png(per_subj, stats, OUTPUT_DIR / "n2_bouts_table.png")

    print("\nGroup means (mean ± SD):")
    for key, label, decimals in METRICS:
        s = stats[key]
        parts = [label + ":"]
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            if len(vals):
                parts.append(f"{g}={vals.mean():.{decimals}f}±{vals.std(ddof=1):.{decimals}f}")
            else:
                parts.append(f"{g}=—")
        parts.append(f"[{s['test']} p={_format_p(s['p'])}]")
        print("  " + "  ".join(parts))
        for (g1, g2), pd_ in s["pairs"].items():
            tag = "*" if pd_["sig"] else "ns"
            print(f"      {g1} vs {g2}: p={pd_['p']:.4f} ({tag})")

    print(f"\nSaved: {OUTPUT_DIR / 'n2_bouts_per_subject.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'n2_bouts_table.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'n2_bouts_table.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'n2_bouts_table.png'}")


if __name__ == "__main__":
    main()
