"""
Combined sleep-architecture + N2-bout-properties table (3-group comparison).

Merges the two standalone tables into ONE figure with a shared 8-column layout:

    Variable | Young | Elderly | MCI | Omnibus p | Y vs E | Y vs MCI | E vs MCI

Two section bands:
  - "Sleep architecture (% of recording)"  -> 5 stage rows (Wake/N1/N2/N3/REM)
  - "N2 bout properties (>= 300 s)"          -> 5 bout-metric rows

All per-group values are shown as mean +/- SD. Stats (omnibus + pairwise) are
computed by REUSING the exact pipelines in sleep_stage_pies.py and
n2_bouts_table.py, so numbers match the standalone tables. The standalone
PNGs/CSVs are left untouched; this writes new filenames.

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/combined_demographics_table.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the existing modules (same dir on sys.path when run as python code/<file>).
# Before the two imports below: n2_bouts_table puts code/utils on sys.path, at
# which point "utils" resolves to utils.py and "utils.config" stops importing.
from utils.config import group_label

import sleep_stage_pies as ssp
import n2_bouts_table as nbt


OUTPUT_DIR = Path("results/demographics_V3")
GROUP_ORDER = ["YA", "HE", "MCI"]
GROUP_TITLES = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}
PAIR_KEYS = [("YA", "HE"), ("YA", "MCI"), ("HE", "MCI")]
PAIR_HEADERS = ["Y vs E", "Y vs aMCI", "E vs aMCI"]

# colors
BAND_COLOR = "#DCE6F1"      # section-band fill
HEADER_COLOR = "#E5E5E5"
SIG_COLOR = "#EAF4EA"       # significant-cell highlight


def _format_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def _pair(stats_entry: dict, g1: str, g2: str) -> dict | None:
    pairs = stats_entry["pairs"]
    return pairs.get((g1, g2)) or pairs.get((g2, g1))


def build_rows():
    """Return (section_blocks, n_by_group_stage, n_by_group_bout).

    Each block: (band_label, [ (variable_label, {group: 'mean±sd'}, stats_entry) ]).
    """
    # ---- Sleep architecture (reuse sleep_stage_pies) ----
    df_stage = ssp.load_subjects_table()
    rec_pct = ssp.per_subject_recording_pct(df_stage)
    stage_means = ssp.group_means(rec_pct)        # has 'n' + per-stage mean
    stage_stats = ssp.run_stats(rec_pct)
    n_stage = {g: int(stage_means.loc[g, "n"]) for g in GROUP_ORDER}

    stage_rows = []
    for stage in ssp.STAGES:
        gvals = {}
        for g in GROUP_ORDER:
            v = rec_pct.loc[rec_pct["group"] == g, stage].dropna()
            gvals[g] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) else "—"
        stage_rows.append((stage, gvals, stage_stats[stage]))

    # ---- N2 bout properties (reuse n2_bouts_table) ----
    df_bout = nbt.load_subjects()
    per_subj = pd.DataFrame([nbt.per_subject_metrics(r) for _, r in df_bout.iterrows()])
    bout_stats = {key: nbt.run_metric_stats(per_subj, key) for key, _, _ in nbt.METRICS}
    n_bout = {g: int(per_subj.loc[per_subj["group"] == g, "bout_count"].notna().sum())
              for g in GROUP_ORDER}

    bout_rows = []
    for key, label, dec in nbt.METRICS:
        gvals = {}
        for g in GROUP_ORDER:
            v = per_subj.loc[per_subj["group"] == g, key].dropna()
            gvals[g] = f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f}" if len(v) > 1 else (
                f"{v.mean():.{dec}f}" if len(v) == 1 else "—")
        bout_rows.append((label, gvals, bout_stats[key]))

    blocks = [
        ("Sleep architecture (% of recording)", stage_rows),
        (f"N2 bout properties (≥ {int(nbt.MIN_BOUT_SEC)} s)", bout_rows),
    ]
    return blocks, n_stage, n_bout


def _format_eta(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}".lstrip("0") if abs(value) < 1 else f"{value:.3f}"


def render(blocks, n_stage, n_bout, out_path: Path, fontsize: int = 16,
           fig_width: float = 12.0, show_test_and_eta: bool = False,
           title: str | None = None, footer: str | None = None,
           row_scale: float = 2.4) -> None:
    """Render the comparison table to a PNG.

    The keyword arguments default to the values this module has always used, so
    the standalone demographics_V3 table is reproducible unchanged. Table 2
    (code/make_table2_sleep.py) passes larger type and asks for the extra
    ``Test`` and eta-squared columns, which read ``st['test']`` and
    ``st['eta2']`` from each row's stats entry.
    """
    # GROUP_TITLES stays the CSV/lookup key; group_label() is what the table
    # prints, so the manuscript's aMCI label never reaches the data columns.
    header = ["",
              *(f"{group_label(GROUP_TITLES[g])}\n(n={n_stage[g]})" for g in GROUP_ORDER)]
    if show_test_and_eta:
        header += ["Test", "Omnibus p", "η²"]
    else:
        header += ["Omnibus p"]
    header += PAIR_HEADERS
    ncol = len(header)

    cell_text = [header]
    cell_colors = [[HEADER_COLOR] * ncol]
    bold_cells: list[tuple[int, int]] = []
    band_info: list[tuple[int, str]] = []   # (row_index, label)
    stage_cells: list[int] = []             # row indices whose col-0 is a stage swatch

    for band_label, rows in blocks:
        # Section band: leave the cell text BLANK so it doesn't inflate the
        # auto-sized column-0 width; the label is drawn afterwards as overflow.
        cell_text.append([""] * ncol)
        cell_colors.append([BAND_COLOR] * ncol)
        band_info.append((len(cell_text) - 1, band_label))

        for var_label, gvals, st in rows:
            row_vals = [var_label, *(gvals[g] for g in GROUP_ORDER)]
            if show_test_and_eta:
                row_vals.append(st.get("test") or "—")
            # column index of the omnibus p cell, used for the bold highlight
            p_col = len(row_vals)
            row_vals.append(_format_p(st["p"]))
            if show_test_and_eta:
                row_vals.append(_format_eta(st.get("eta2")))
            for (g1, g2) in PAIR_KEYS:
                pr = _pair(st, g1, g2)
                row_vals.append(_format_p(pr["p"]) if pr else "—")
            cell_text.append(row_vals)
            cell_colors.append(["white"] * ncol)

            ridx = len(cell_text) - 1
            # color the stage-name cell with the pie's stage swatch
            if var_label in ssp.STAGE_COLORS:
                cell_colors[ridx][0] = ssp.STAGE_COLORS[var_label]
                stage_cells.append(ridx)
            # bold/highlight significant omnibus + pairwise
            if np.isfinite(st["p"]) and st["p"] < 0.05:
                bold_cells.append((ridx, p_col))
            first_pair_col = ncol - len(PAIR_KEYS)
            for off, (g1, g2) in enumerate(PAIR_KEYS):
                pr = _pair(st, g1, g2)
                if pr is not None and pr["sig"]:
                    bold_cells.append((ridx, first_pair_col + off))

    nrows = len(cell_text)
    band_rows = [r for r, _ in band_info]
    extra_h = (0.5 if title else 0.0) + (0.5 if footer else 0.0)
    row_h = 0.55 * (row_scale / 2.4)   # keep the default figure height unchanged
    fig = plt.figure(figsize=(fig_width, row_h * nrows + 0.5 + extra_h))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    # Size each column to fit its content (tight; no manual-width gaps).
    table.auto_set_column_width(col=list(range(ncol)))
    table.scale(1.0, row_scale)
    # Taller header row so the two-line "Young\n(n=X)" labels fit inside it.
    header_h = table[(0, 0)].get_height()
    for j in range(ncol):
        table[(0, j)].set_height(header_h * 1.9)

    # left-align variable column
    for i in range(nrows):
        table[(i, 0)].get_text().set_horizontalalignment("left")
    # header bold
    for j in range(ncol):
        table[(0, j)].get_text().set_fontweight("bold")
    # significant cells bold + tinted
    for (i, j) in bold_cells:
        table[(i, j)].get_text().set_fontweight("bold")
        table[(i, j)].set_facecolor(SIG_COLOR)

    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDDDDD")
    # band rows: hide internal gridlines so the band reads as one clean stripe
    for r in band_rows:
        for j in range(ncol):
            table[(r, j)].set_edgecolor(BAND_COLOR)
    # stage-name cells: white bold text on the colored swatch (matches pies)
    for r in stage_cells:
        t = table[(r, 0)].get_text()
        t.set_color("white")
        t.set_fontweight("bold")
    # draw the band labels as overflow text (clip off) so a long label can
    # extend past column 0 without forcing that column wide.
    for r, label in band_info:
        t = table[(r, 0)].get_text()
        t.set_text(label)
        t.set_fontweight("bold")
        t.set_horizontalalignment("left")
        t.set_clip_on(False)

    # Title and explanatory footer are omitted by default: those details live in
    # the thesis text / Methods, and the embedded version wants a bare table.
    # A standalone table (Table 2) passes them in.
    if title:
        ax.set_title(title, fontsize=fontsize + 3, fontweight="bold", pad=16)
    if footer:
        ax.text(0.0, -0.02, footer, transform=ax.transAxes, fontsize=fontsize - 4,
                va="top", ha="left", color="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(blocks, out_path: Path) -> None:
    rows = []
    for band_label, brows in blocks:
        for var_label, gvals, st in brows:
            row = {"section": band_label, "variable": var_label}
            for g in GROUP_ORDER:
                row[GROUP_TITLES[g]] = gvals[g]
            row["omnibus_test"] = st["test"]
            row["omnibus_p"] = round(st["p"], 4) if np.isfinite(st["p"]) else ""
            for (g1, g2) in PAIR_KEYS:
                pr = _pair(st, g1, g2)
                row[f"p_{g1}_vs_{g2}"] = round(pr["p"], 4) if pr else ""
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    blocks, n_stage, n_bout = build_rows()

    png = OUTPUT_DIR / "combined_sleep_n2_table.png"
    csv = OUTPUT_DIR / "combined_sleep_n2_table.csv"
    render(blocks, n_stage, n_bout, png)
    write_csv(blocks, csv)

    print(f"Stage-section n: " + ", ".join(f"{g}={n_stage[g]}" for g in GROUP_ORDER))
    print(f"Bout-section  n: " + ", ".join(f"{g}={n_bout[g]}" for g in GROUP_ORDER))
    print(f"\nSaved: {png}")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
