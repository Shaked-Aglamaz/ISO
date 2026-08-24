"""
Table 2 — sleep architecture, sleep continuity and N2 bout properties.

Promoted out of Figure 1 panel C, which was an unreadable pasted bitmap
(review comments C389 / C390): the table is now standalone, in larger type, and
carries four extra continuity metrics plus the test name and effect size for
every row.

Nothing is recomputed. Every value is read back from files already on disk:

  results/demographics_V3/combined_sleep_n2_table.csv   mean +/- SD and p-values
                                                        for the architecture and
                                                        N2-bout rows
  results/demographics_V3/sleep_stage_stats.txt         test + eta^2 per stage
  results/demographics_V3/n2_bouts_table.csv            test + eta^2 per bout metric
  results/demographics_V4/sleep_statistics_table.csv    test, p and eta^2 for the
                                                        four continuity metrics
  results/demographics_V4/sleep_statistics_per_subject.csv
                                                        the continuity mean +/- SD,
                                                        formatted from the raw
                                                        per-subject values

Sleep onset latency is reported even though its omnibus test is not significant
(p = 0.38); it was explicitly requested.

Rendering reuses combined_demographics_table.render, so the layout matches the
table this replaces.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/make_table2_sleep.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import combined_demographics_table as cdt

V3_DIR = Path("results/demographics_V3")
V4_DIR = Path("results/demographics_V4")
OUT_DIR = V4_DIR

COMBINED_CSV = V3_DIR / "combined_sleep_n2_table.csv"
STAGE_STATS_TXT = V3_DIR / "sleep_stage_stats.txt"
BOUTS_CSV = V3_DIR / "n2_bouts_table.csv"
CONTINUITY_CSV = V4_DIR / "sleep_statistics_table.csv"
CONTINUITY_SUBJ_CSV = V4_DIR / "sleep_statistics_per_subject.csv"

GROUP_ORDER = cdt.GROUP_ORDER            # ["YA", "HE", "MCI"]
GROUP_TITLES = cdt.GROUP_TITLES          # YA -> Young, ...
PAIR_KEYS = cdt.PAIR_KEYS

# Section names as they appear in demographics_V3/combined_sleep_n2_table.csv.
ARCH_SECTION = "Sleep architecture (% of recording)"
CONT_SECTION = "Sleep continuity"
BOUT_SECTION = "N2 bout properties (≥ 300 s)"

# Shorter band labels for the rendered table. The band label is the longest
# string in column 0, so it is what sets that column's width — trimming it is
# what balances the table. The qualifiers it drops ("% of recording",
# ">= 300 s") are stated in the caption instead.
# The architecture rows are percentages and carry no unit of their own (the
# other two sections put units in the row label), so the band keeps a short
# "(%)"; "of total recording time" is spelled out in the caption.
SECTION_DISPLAY = {
    ARCH_SECTION: "Sleep architecture (%)",
    CONT_SECTION: "Sleep continuity",
    BOUT_SECTION: "N2 bout properties",
}

# Continuity rows, in the order Yuval asked for them: (row key in the summary
# CSV, displayed label, per-subject column, decimals).
# The mean and SD are formatted from the per-subject column rather than from the
# summary CSV, which stores them already rounded to two decimals: rounding 46.25
# again to one decimal lands on 46.2, while the same SD formatted once from the
# raw values is 46.3 — the figure sleep_statistics_stats.txt prints and the one
# the Results text quotes. Formatting once, from the raw values, keeps the table
# and the prose identical by construction.
CONTINUITY_ROWS = [
    ("WASO (min)", "WASO (min)", "waso_min", 1),
    ("Sleep onset latency (min)", "Sleep onset latency (min)", "sol_min", 1),
    ("REM latency (min)", "REM latency (min)", "rem_latency_min", 1),
    ("Sleep efficiency (%)", "Sleep efficiency (%)", "sleep_efficiency_pct", 1),
]

# The table is pasted at page width, where its on-page type size is set by its
# total text width (columns auto-size to their content). Abbreviating the two
# widest repeated strings is what actually buys legibility; both are spelled out
# in the footnote.
TEST_ABBREV = {"Kruskal-Wallis": "KW", "One-Way ANOVA": "ANOVA", "ANOVA": "ANOVA"}

# No on-figure title and no on-figure footnote: the table number and title come
# from the caption in the manuscript, matching F3 / F5 / S2, and the definitions (mean ± SD, WASO, KW, η², the dash for
# post-hoc tests that were not run, bold = p < 0.05) all live in the caption in
# thesis/figure_manifest.md, which is where the reader meets them.


def _stats_entry(test, p, eta2, pair_ps):
    """Build the stats dict shape that combined_demographics_table.render wants."""
    pairs = {}
    for (g1, g2), p_val in pair_ps.items():
        if p_val is None or not np.isfinite(p_val):
            continue
        pairs[(g1, g2)] = {"p": p_val, "sig": p_val < 0.05}
    # 'test' is what the rendered cell shows; 'test_full' is what the CSV keeps.
    return {"test": TEST_ABBREV.get(test, test), "test_full": test,
            "p": p, "eta2": eta2, "pairs": pairs}


def _p(value):
    """Blank / NaN cells in the source CSVs mean 'post-hoc not run'."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def parse_stage_eta() -> dict[str, tuple[str, float]]:
    """{stage: (omnibus test name, eta^2)} from the V3 sleep-stage report."""
    text = STAGE_STATS_TXT.read_text(encoding="utf-8")
    out = {}
    for stage, body in re.findall(r"--- (\w+) ---\n(.*?)(?=\n--- |\Z)", text, re.S):
        m = re.search(r"Omnibus:\s+(.+?)\s+\w+=[-\d.]+, p=[\d.]+, eta2=([\d.]+)", body)
        if m:
            out[stage] = (m.group(1).strip(), float(m.group(2)))
    return out


def build_blocks():
    combined = pd.read_csv(COMBINED_CSV)
    bouts = pd.read_csv(BOUTS_CSV).set_index("metric")
    continuity = pd.read_csv(CONTINUITY_CSV).set_index("metric")
    cont_subj = pd.read_csv(CONTINUITY_SUBJ_CSV)
    stage_eta = parse_stage_eta()

    def rows_from_combined(section, eta_lookup):
        rows = []
        for _, r in combined[combined["section"] == section].iterrows():
            var = r["variable"]
            gvals = {g: r[GROUP_TITLES[g]] for g in GROUP_ORDER}
            test, eta2 = eta_lookup(var)
            st = _stats_entry(
                test, _p(r["omnibus_p"]), eta2,
                {pair: _p(r.get(f"p_{pair[0]}_vs_{pair[1]}")) for pair in PAIR_KEYS},
            )
            rows.append((var, gvals, st))
        return rows

    def stage_lookup(var):
        return stage_eta.get(var, (None, np.nan))

    def bout_lookup(var):
        if var not in bouts.index:
            return (None, np.nan)
        row = bouts.loc[var]
        return (row["omnibus_test"], float(row["eta_squared"]))

    arch_rows = rows_from_combined(ARCH_SECTION, stage_lookup)
    bout_rows = rows_from_combined(BOUT_SECTION, bout_lookup)

    cont_rows = []
    for key, label, subj_col, dec in CONTINUITY_ROWS:
        r = continuity.loc[key]
        gvals = {}
        for g in GROUP_ORDER:
            vals = cont_subj.loc[cont_subj["group"] == g, subj_col].dropna()
            gvals[g] = f"{vals.mean():.{dec}f} ± {vals.std(ddof=1):.{dec}f}"
        st = _stats_entry(
            r["omnibus_test"], _p(r["omnibus_p"]), float(r["eta_squared"]),
            {pair: _p(r.get(f"p_{pair[0]}_vs_{pair[1]}")) for pair in PAIR_KEYS},
        )
        cont_rows.append((label, gvals, st))

    n_by_group = {g: int(continuity.iloc[0][f"{g}_n"]) for g in GROUP_ORDER}

    blocks = [
        (SECTION_DISPLAY[ARCH_SECTION], arch_rows),
        (SECTION_DISPLAY[CONT_SECTION], cont_rows),
        (SECTION_DISPLAY[BOUT_SECTION], bout_rows),
    ]
    return blocks, n_by_group


def write_csv(blocks, out_path: Path) -> None:
    # The rendered table uses the short band labels; the CSV keeps the fully
    # qualified section names so the units survive outside the caption.
    full_name = {short: full for full, short in SECTION_DISPLAY.items()}
    rows = []
    for section, brows in blocks:
        for var_label, gvals, st in brows:
            row = {"section": full_name.get(section, section), "variable": var_label}
            for g in GROUP_ORDER:
                row[GROUP_TITLES[g]] = gvals[g]
            row["omnibus_test"] = st["test_full"] or ""
            row["omnibus_p"] = round(st["p"], 4) if st["p"] is not None else ""
            row["eta_squared"] = ("" if st["eta2"] is None or not np.isfinite(st["eta2"])
                                  else round(st["eta2"], 4))
            for (g1, g2) in PAIR_KEYS:
                pr = st["pairs"].get((g1, g2))
                row[f"p_{g1}_vs_{g2}"] = round(pr["p"], 4) if pr else ""
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def main() -> None:
    blocks, n_by_group = build_blocks()

    png = OUT_DIR / "table2_sleep_architecture.png"
    csv = OUT_DIR / "table2_sleep_architecture.csv"

    cdt.render(blocks, n_by_group, n_by_group, png,
               fontsize=20, fig_width=17.0, show_test_and_eta=True,
               row_scale=3.3)
    write_csv(blocks, csv)

    n_rows = sum(len(r) for _, r in blocks)
    print("n per group: " + ", ".join(f"{GROUP_TITLES[g]}={n_by_group[g]}"
                                      for g in GROUP_ORDER))
    print(f"{n_rows} rows in {len(blocks)} sections")
    print(f"Saved: {png}")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
