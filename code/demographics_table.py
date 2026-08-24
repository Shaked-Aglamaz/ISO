"""
Per-group demographics + exclusion-reason summary table.

Reads two tabs live from the Google Sheet (same pattern as moca_correlation.py
and sleep_stage_pies.py):
  - subjects: active cohort     -> age, MoCA, %bad channels, %bad epochs (in N2)
  - excluded: rejected subjects -> counted by the "recorded reason" column

Exclusion categories (per user spec):
  - not enough clean bouts (fewer than 3 clean N2 bouts >= 300s)
  - too much bad channels
  - too much bad epochs
  - TST < 210 min
Plus an "other" bucket for reasons that do not fit. AD-excluded subjects are
NOT counted (footnoted instead).

Outputs (results/demographics_V1/):
  - demographics_table.csv
  - demographics_table.txt
  - demographics_table.png

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/demographics_table.py
"""
from __future__ import annotations

from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials


SPREADSHEET_ID = "1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw"
SA_KEY_PATH = Path("C:/Users/Shaked/.gcp/refined-spirit-494512-e8-059d94c0eb34.json")

OUTPUT_DIR = Path("results/demographics_V3")

GROUP_ORDER = ["YA", "HE", "MCI"]
GROUP_TITLES = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}

# Exclusion category = the sheet's "recorded reason" column verbatim. The labels
# below must match that column's values exactly; the keys are stable slugs used
# for CSV/dict column names. Anything not listed here is counted as "Other".
EXCL_CATEGORIES = [
    ("not_enough_n2",   "Not enough clean bouts"),  # < 3 clean N2 bouts >= 300s
    ("bad_channels",    "Too many bad channels"),
    ("bad_epochs",      "Too many bad epochs"),
    ("tst_too_short",   "TST < 210 min"),
    ("other",           "Other"),
]
_LABEL_TO_KEY = {label: key for key, label in EXCL_CATEGORIES}


def _gs_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return gspread.authorize(creds)


def load_sheets() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = _gs_client()
    sh = client.open_by_key(SPREADSHEET_ID)
    subjects = pd.DataFrame(sh.worksheet("subjects").get_all_records())
    excluded = pd.DataFrame(sh.worksheet("excluded").get_all_records())
    for c in ("age", "moca", "pct_bad_channels", "pct_bad_epochs_in_n2"):
        subjects[c] = pd.to_numeric(subjects[c], errors="coerce")
    return subjects, excluded


def _site(subject_id: str) -> str:
    """Recording site from the subject ID. Subjects named 'MCI…' come from the
    Sydney cohort (Woolcock); everyone else is from TASMC (Tel Aviv / Ichilov)."""
    return "Sydney" if "MCI" in str(subject_id).upper() else "TASMC"


def _mean_std(s: pd.Series) -> tuple[float, float, int]:
    s = s.dropna()
    if s.empty:
        return float("nan"), float("nan"), 0
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))


def _fmt_ms(mean: float, std: float, n: int | None = None,
            decimals: int = 1) -> str:
    if not np.isfinite(mean):
        return "—"
    s = f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    if n is not None:
        s += f" (n={n})"
    return s


def build_demographics_row(group: str, df: pd.DataFrame) -> dict:
    sub = df[df["group"] == group]
    age_m, age_s, _ = _mean_std(sub["age"])
    moca_m, moca_s, moca_n = _mean_std(sub["moca"])
    bch_m, bch_s, _ = _mean_std(sub["pct_bad_channels"])
    bep_m, bep_s, _ = _mean_std(sub["pct_bad_epochs_in_n2"])
    sites = sub["subject_id"].map(_site)
    return {
        "group": group,
        "n": int(len(sub)),
        "n_tasmc": int((sites == "TASMC").sum()),
        "n_sydney": int((sites == "Sydney").sum()),
        "age_mean": age_m, "age_std": age_s,
        "moca_mean": moca_m, "moca_std": moca_s, "moca_n": moca_n,
        "bch_mean": bch_m, "bch_std": bch_s,
        "bep_mean": bep_m, "bep_std": bep_s,
    }


def build_exclusion_row(group: str, excl: pd.DataFrame) -> dict:
    """Counts of exclusions, taken straight from the sheet's "recorded reason"
    column. AD subjects are not counted."""
    g = excl[excl["group"] == group].copy()
    n_ad = int((g["excluded_dir"].str.lower() == "excluded_ad").sum())
    g = g[g["excluded_dir"].str.lower() != "excluded_ad"].copy()

    # The "recorded reason" column already holds the display label; map it to its
    # stable key, defaulting anything unrecognized/blank to "other".
    g["category"] = [
        _LABEL_TO_KEY.get(str(rr).strip(), "other")
        for rr in g["recorded reason"]
    ]
    counts = g["category"].value_counts().to_dict()

    row = {"group": group,
           "excluded_total": int(len(g)),
           "excluded_ad_excluded_from_table": n_ad}
    for key, _ in EXCL_CATEGORIES:
        row[f"excl_{key}"] = int(counts.get(key, 0))
    row["_non_ad_rows"] = g
    return row


def render_table_png(demo: list[dict], excl: list[dict], out_path: Path) -> None:
    fig = plt.figure(figsize=(11, 3.6))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    header = ["Variable"] + [GROUP_TITLES[g] for g in GROUP_ORDER]

    # --- demographics block
    rows = [
        ["N",                    *[f"{d['n']}" for d in demo]],
        ["Age (years)",          *[_fmt_ms(d['age_mean'], d['age_std']) for d in demo]],
        ["MoCA (where available)", *[_fmt_ms(d['moca_mean'], d['moca_std'], d['moca_n']) for d in demo]],
        ["Exclusions",           "", "", ""],   # section divider
        ["  Total excluded",     *[f"{e['excluded_total']}" for e in excl]],
    ]
    for key, label in EXCL_CATEGORIES:
        # Hide an empty 'other' row; always show the four user-specified categories.
        if key == "other" and all(e[f"excl_{key}"] == 0 for e in excl):
            continue
        rows.append([f"  {label}",
                     *[f"{e[f'excl_{key}']}" for e in excl]])

    cell_text = [header] + rows

    cell_colors = []
    cell_colors.append(["#E5E5E5"] * len(header))                  # header row
    for r in rows:
        if r[0] == "Exclusions":
            cell_colors.append(["#F5F5F5"] * 4)                    # section row
        else:
            cell_colors.append(["white"] * 4)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colWidths=[0.28, 0.24, 0.24, 0.24],
        cellLoc="center",
        loc="center",
    )
    # Left-align the variable column
    for i in range(len(cell_text)):
        c = table[(i, 0)]
        c.get_text().set_horizontalalignment("left")

    # Bold header + section divider
    for j in range(len(header)):
        table[(0, j)].get_text().set_fontweight("bold")
    section_idx = next(i for i, r in enumerate(rows) if r[0] == "Exclusions") + 1
    for j in range(len(header)):
        table[(section_idx, j)].get_text().set_fontweight("bold")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDDDDD")

    # The N cell shows the group total at the normal cell size; give the row
    # extra height to leave room for a smaller per-site breakdown overlaid below
    # the total (so the total stays the same size as the other data cells).
    n_row_idx = 1  # header is row 0, "N" is the first data row
    for j in range(len(header)):
        table[(n_row_idx, j)].set_height(table[(n_row_idx, j)].get_height() * 1.6)

    # Overlay the per-site breakdown just below each group's N total, in a
    # smaller font. Done after a draw so the cell geometry is finalized.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for j, d in enumerate(demo, start=1):
        bb = table[(n_row_idx, j)].get_window_extent(renderer).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 + bb.height * 0.24,
                 f"(TASMC {d['n_tasmc']} + Sydney {d['n_sydney']})",
                 ha="center", va="center", fontsize=8, color="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_text_report(demo: list[dict], excl: list[dict], path: Path) -> None:
    lines = []
    lines.append("Cohort demographics and exclusion reasons")
    lines.append("=" * 60)
    lines.append("")
    for d in demo:
        g = d["group"]
        lines.append(f"--- {GROUP_TITLES[g]} (n={d['n']}) ---")
        lines.append(f"  N (TASMC):         {d['n_tasmc']}")
        lines.append(f"  N (Sydney):        {d['n_sydney']}")
        lines.append(f"  Age:               {_fmt_ms(d['age_mean'], d['age_std'])} years")
        lines.append(f"  MoCA:              {_fmt_ms(d['moca_mean'], d['moca_std'], d['moca_n'])}")
        lines.append(f"  % bad channels:    {_fmt_ms(d['bch_mean'], d['bch_std'])}")
        lines.append(f"  % bad epochs (N2): {_fmt_ms(d['bep_mean'], d['bep_std'])}")
        lines.append("")

    lines.append("Exclusions (AD-excluded MCI subjects not counted)")
    lines.append("-" * 60)
    header = f"{'Reason':<28}" + "".join(f"{GROUP_TITLES[g]:>10}" for g in GROUP_ORDER)
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(f"{'Total excluded':<28}" + "".join(
        f"{e['excluded_total']:>10}" for e in excl))
    for key, label in EXCL_CATEGORIES:
        lines.append(f"  {label:<26}" + "".join(
            f"{e[f'excl_{key}']:>10}" for e in excl))

    n_ad = next(e["excluded_ad_excluded_from_table"]
                for e in excl if e["group"] == "MCI")
    lines.append("")
    lines.append(f"Note: {n_ad} MCI subjects flagged AD are excluded from these counts.")

    # Per-subject categorization detail for QA
    lines.append("")
    lines.append("Per-subject exclusion categorization (for QA)")
    lines.append("-" * 60)
    for e in excl:
        g = e["group"]
        rows = e["_non_ad_rows"]
        if rows.empty:
            continue
        lines.append(f"[{GROUP_TITLES[g]}]")
        for _, r in rows.iterrows():
            lines.append(f"  {r['subject_id']:<10} -> {r['category']:<18} "
                         f"| {r['reason']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    subjects, excluded = load_sheets()

    demo = [build_demographics_row(g, subjects) for g in GROUP_ORDER]
    excl = [build_exclusion_row(g, excluded)    for g in GROUP_ORDER]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV (machine-readable)
    csv_rows = []
    for d, e in zip(demo, excl):
        csv_rows.append({
            "group": d["group"],
            "n": d["n"],
            "age_mean": round(d["age_mean"], 2),
            "age_std":  round(d["age_std"], 2),
            "moca_mean": round(d["moca_mean"], 2) if np.isfinite(d["moca_mean"]) else "",
            "moca_std":  round(d["moca_std"], 2) if np.isfinite(d["moca_std"]) else "",
            "moca_n":    d["moca_n"],
            "pct_bad_channels_mean": round(d["bch_mean"], 2),
            "pct_bad_channels_std":  round(d["bch_std"], 2),
            "pct_bad_epochs_in_n2_mean": round(d["bep_mean"], 2),
            "pct_bad_epochs_in_n2_std":  round(d["bep_std"], 2),
            "excluded_total": e["excluded_total"],
            **{f"excl_{k}": e[f"excl_{k}"] for k, _ in EXCL_CATEGORIES},
            "ad_excluded_not_counted": e["excluded_ad_excluded_from_table"],
        })
    pd.DataFrame(csv_rows).to_csv(OUTPUT_DIR / "demographics_table.csv", index=False)

    write_text_report(demo, excl, OUTPUT_DIR / "demographics_table.txt")
    render_table_png(demo, excl, OUTPUT_DIR / "demographics_table.png")

    # Console preview
    print(f"\nGroup demographics (mean ± SD):")
    for d in demo:
        print(f"  {GROUP_TITLES[d['group']]:<7} n={d['n']:<3} "
              f"age={_fmt_ms(d['age_mean'], d['age_std'])}  "
              f"moca={_fmt_ms(d['moca_mean'], d['moca_std'], d['moca_n'])}  "
              f"bad_ch={_fmt_ms(d['bch_mean'], d['bch_std'])}%  "
              f"bad_ep={_fmt_ms(d['bep_mean'], d['bep_std'])}%")

    print(f"\nExclusion counts (AD not counted):")
    for e in excl:
        cats = ", ".join(f"{lbl}={e[f'excl_{k}']}" for k, lbl in EXCL_CATEGORIES)
        print(f"  {GROUP_TITLES[e['group']]:<7} total={e['excluded_total']}  "
              f"({cats})")

    print(f"\nSaved: {OUTPUT_DIR / 'demographics_table.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'demographics_table.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'demographics_table.png'}")


if __name__ == "__main__":
    main()
