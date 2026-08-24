"""Regenerate Figure S2 (ISFS scalars vs MoCA) with the paper styling.

Answers C592 ("Graphics and fonts - can't read"). Two things were wrong with
the V3 render: the type was far too small once the grid is shrunk to page
width, and the group colours were the OPPOSITE of every other figure — the
exploratory palette draws elderly blue and MCI red, while Figures 3 to 5 use
elderly red and MCI blue.

Nothing is recomputed. The correlations and the per-subject scalars are read
back from results/moca_correlation_V3/, so the Google Sheet is never touched
and every number is identical to V3 by construction.

Output goes to moca_correlation_V4 because the filename already exists in V3.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/replot_s2_moca.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from moca_correlation import SCALAR_PANELS, plot_grid

V3_DIR = Path("results/moca_correlation_V3")
OUT_DIR = Path("results/moca_correlation_V4")


def main() -> None:
    subj_df = pd.read_csv(V3_DIR / "subject_scalars.csv")
    summary = pd.read_csv(V3_DIR / "correlation_summary.csv")

    counts = subj_df["group"].value_counts()
    print(f"Reusing V3 values from {V3_DIR} (no recomputation).")
    print(f"  pooled n = {len(subj_df)}: "
          + ", ".join(f"{g}={n}" for g, n in counts.items()))
    for _, r in summary.iterrows():
        print(f"  {r['scalar']:<16} r={r['pearson_r']:+.3f} p={r['pearson_p']:.3f}   "
              f"rho={r['spearman_r']:+.3f} p={r['spearman_p']:.3f}   N={int(r['n'])}")

    missing = {c for c, _, _ in SCALAR_PANELS} - set(subj_df.columns)
    if missing:
        raise RuntimeError(f"subject_scalars.csv is missing columns: {sorted(missing)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "moca_correlations_grid.png"
    plot_grid(subj_df, summary, fig_path, paper_style=True)
    print(f"\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
