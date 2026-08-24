"""
C390 — extended sleep statistics: WASO, sleep-onset latency, REM latency.

Yuval asked (comment C390) that the sleep-architecture panel be extended with
sleep efficiency, WASO and REM sleep latency. Sleep efficiency already exists in
the subjects sheet (sleep_efficiency_pct); the other metrics have never been
extracted. This script derives them per subject and compares them across groups
with the same normality-gated scheme used in
results/demographics_V3/sleep_stage_stats.txt.

Hypnogram source
----------------
Each subject's *_cleaned_annotations.txt, which is the normalized representation
of the scoring file (ISO_data/scoring/ has inconsistent naming, letter-vs-int
codes and mixed 1 Hz / 30 s resolutions across the three groups). Annotation
onsets and durations are second-resolution, not 30 s multiples, so the hypnogram
is built at 1 Hz.

Unscorable epochs
-----------------
Per notes/tst_waso_unknown_handling.md, UNKNOWN (-1 in the scoring file) is a
third category — not sleep, not wake, not recorded. It is excluded from both the
numerator and the denominator of sleep efficiency and is never folded into WASO.
yasa's own WASO/SOL/latencies already honour this (WASO counts only == 0 inside
SPT); yasa's SE does not, since it divides by the full hypnogram length, so a
policy SE is computed alongside it.

BAD / BAD_EPOCH / BAD_ACQ_SKIP annotations are deliberately ignored here: in
*_cleaned_annotations.txt the BAD_ACQ_SKIP label is overloaded with hand-drawn
artifact marks from the step1 notebook, so it cannot be used to find real
recording gaps.

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/sleep_statistics_extended.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gspread
import numpy as np
import pandas as pd
import yasa
from google.oauth2.service_account import Credentials

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code" / "utils"))

from subject_summary import parse_annotations  # noqa: E402
from n2_bouts_table import annotation_path_for, run_metric_stats  # noqa: E402


SPREADSHEET_ID = "1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw"
SHEET_TAB = "subjects"
SA_KEY_PATH = Path("C:/Users/Shaked/.gcp/refined-spirit-494512-e8-059d94c0eb34.json")

OUTPUT_DIR = Path("results/demographics_V4")

GROUP_ORDER = ["YA", "HE", "MCI"]
GROUP_TITLES = {"YA": "Young", "HE": "Elderly", "MCI": "MCI"}

# Both stage-label dialects present in *_cleaned_annotations.txt.
STAGE_CODES = {
    "Wake": 0, "WAKE": 0,
    "N1": 1, "NREM1": 1,
    "N2": 2, "NREM2": 2,
    "N3": 3, "NREM3": 3,
    "REM": 4,
    "UNKNOWN": -1, "Unknown": -1,
}
UNSCORABLE = -1

# Metrics that get the full omnibus + post-hoc treatment.
METRICS = [
    ("waso_min",            "WASO (min)",                  1),
    ("sol_min",             "Sleep onset latency (min)",   1),
    ("rem_latency_min",     "REM latency (min)",           1),
    ("sleep_efficiency_pct", "Sleep efficiency (%)",       1),
]


def load_subjects() -> pd.DataFrame:
    creds = Credentials.from_service_account_file(
        str(SA_KEY_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    client = gspread.authorize(creds)
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TAB)
    df = pd.DataFrame(ws.get_all_records())
    needed = ["subject_id", "group", "group_dir",
              "recording_sec", "tst_sec", "wake_sec", "sleep_efficiency_pct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet missing columns: {missing}")
    for c in needed[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[needed].copy()


def build_hypnogram(annot_path: Path) -> tuple[np.ndarray, int, int]:
    """Build a 1 Hz integer hypnogram from an annotations file.

    Returns (hypno, overlap_sec, gap_sec).
      hypno       : yasa-coded array, -1 for unscorable seconds
      overlap_sec : seconds written by more than one stage annotation
      gap_sec     : seconds covered by no stage/UNKNOWN annotation at all
                    (folded into the unscorable category)
    """
    anns = parse_annotations(str(annot_path))

    unknown_labels = sorted({
        desc for _, _, desc in anns
        if desc not in STAGE_CODES and not desc.upper().startswith("BAD")
    })
    if unknown_labels:
        raise RuntimeError(
            f"{annot_path.name}: unrecognised annotation label(s) {unknown_labels}. "
            "Add them to STAGE_CODES or confirm they should be ignored."
        )

    segments = sorted(
        (onset, dur, STAGE_CODES[desc])
        for onset, dur, desc in anns if desc in STAGE_CODES
    )
    if not segments:
        raise RuntimeError(f"{annot_path.name}: no sleep-stage annotations found.")

    end_sec = max(onset + dur for onset, dur, _ in segments)
    n_sec = int(round(end_sec))

    hypno = np.full(n_sec, UNSCORABLE, dtype=int)
    written = np.zeros(n_sec, dtype=bool)
    overlap_sec = 0

    # Last-write-wins on onset-sorted annotations.
    for onset, dur, code in segments:
        i0 = max(0, int(round(onset)))
        i1 = min(n_sec, int(round(onset + dur)))
        if i1 <= i0:
            continue
        overlap_sec += int(written[i0:i1].sum())
        hypno[i0:i1] = code
        written[i0:i1] = True

    gap_sec = int((~written).sum())
    return hypno, overlap_sec, gap_sec


def per_subject_stats(row) -> dict:
    annot = annotation_path_for(row["subject_id"], row["group_dir"])
    if annot is None:
        raise RuntimeError(f"No annotations file for {row['subject_id']}")

    hypno, overlap_sec, gap_sec = build_hypnogram(Path(annot))
    st = yasa.sleep_statistics(hypno, sf_hyp=1.0)

    tib_min = st["TIB"]
    unscorable_min = float((hypno == UNSCORABLE).sum()) / 60.0
    scorable_min = tib_min - unscorable_min

    # Policy SE: unscorable seconds out of the denominator as well as the
    # numerator (yasa's own SE divides by the full hypnogram length).
    se_policy = 100.0 * st["TST"] / scorable_min if scorable_min > 0 else float("nan")

    lat_rem_rec = st["Lat_REM"]
    rem_from_onset = lat_rem_rec - st["SOL"] if np.isfinite(lat_rem_rec) else float("nan")

    return {
        "subject_id": row["subject_id"],
        "group": row["group"],
        # --- compared metrics ---
        "waso_min": st["WASO"],
        "sol_min": st["SOL"],
        "rem_latency_min": rem_from_onset,          # AASM: from sleep onset
        "sleep_efficiency_pct": row["sleep_efficiency_pct"],   # sheet, authoritative
        # --- supporting / QC ---
        "rem_latency_from_rec_start_min": lat_rem_rec,
        "sleep_efficiency_policy_pct": se_policy,
        "tst_min": st["TST"],
        "tib_min": tib_min,
        "spt_min": st["SPT"],
        "scorable_min": scorable_min,
        "unscorable_min": unscorable_min,
        "sme_pct": st["SME"],
        "n1_min": st["N1"], "n2_min": st["N2"],
        "n3_min": st["N3"], "rem_min": st["REM"],
        "qc_overlap_sec": overlap_sec,
        "qc_gap_sec": gap_sec,
        # --- sheet values for cross-checking ---
        "sheet_tst_min": row["tst_sec"] / 60.0 if np.isfinite(row["tst_sec"]) else np.nan,
        "sheet_recording_min": (row["recording_sec"] / 60.0
                                if np.isfinite(row["recording_sec"]) else np.nan),
    }


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def write_stats_report(per_subj: pd.DataFrame, stats: dict, path: Path) -> None:
    lines = []
    lines.append("Extended sleep statistics — 3-group comparison")
    lines.append("=" * 60)
    lines.append("Per-subject values from a 1 Hz hypnogram rebuilt from "
                 "*_cleaned_annotations.txt.")
    lines.append("UNKNOWN epochs are treated as unscorable: excluded from the numerator")
    lines.append("AND the denominator, never counted as wake "
                 "(notes/tst_waso_unknown_handling.md).")
    lines.append("WASO / SOL / REM latency from yasa.sleep_statistics; REM latency is")
    lines.append("measured from sleep onset (AASM), i.e. yasa Lat_REM - SOL.")
    lines.append("Sleep efficiency is the subjects sheet column sleep_efficiency_pct "
                 "(not recomputed).")
    lines.append("Pipeline: Shapiro-Wilk -> One-Way ANOVA or Kruskal-Wallis "
                 "-> Tukey HSD or Dunn (Holm) if omnibus p<0.05.")
    lines.append("")
    lines.append("N per group: " + ", ".join(
        f"{g}={int((per_subj['group'] == g).sum())}" for g in GROUP_ORDER))
    lines.append("")

    for key, label, dec in METRICS:
        s = stats[key]
        lines.append(f"--- {label} ---")
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            lines.append(f"  {GROUP_TITLES[g]:<7} (n={len(vals)}): "
                         f"{vals.mean():.{dec}f} ± {vals.std(ddof=1):.{dec}f}  "
                         f"(median={vals.median():.{dec}f})")
        test_name = "One-Way ANOVA" if s["test"] == "ANOVA" else s["test"]
        lines.append("  Shapiro p: " + ", ".join(
            f"{g}={s['shapiro_p'][g]:.3f}" for g in GROUP_ORDER)
            + ("  (all normal)" if s["all_normal"] else "  (non-normal -> KW)"))
        lines.append(f"  Omnibus:   {test_name}  "
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


def build_summary_table(per_subj: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Tidy one-row-per-metric table, same column shape as n2_bouts_table.csv."""
    rows = []
    for key, label, dec in METRICS:
        s = stats[key]
        row = {"metric": label}
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            row[f"{g}_n"] = int(len(vals))
            row[f"{g}_mean"] = round(vals.mean(), dec + 1) if len(vals) else ""
            row[f"{g}_std"] = round(vals.std(ddof=1), dec + 1) if len(vals) > 1 else ""
        row["omnibus_test"] = "One-Way ANOVA" if s["test"] == "ANOVA" else s["test"]
        row["omnibus_stat_label"] = s["stat_label"]
        row["omnibus_stat"] = round(s["stat_value"], 3)
        row["omnibus_p"] = round(s["p"], 4) if np.isfinite(s["p"]) else ""
        row["eta_squared"] = round(s["eta2"], 3) if np.isfinite(s["eta2"]) else ""
        row["posthoc_test"] = s["posthoc_name"] or ""
        for (g1, g2) in [("YA", "HE"), ("YA", "MCI"), ("HE", "MCI")]:
            pd_ = s["pairs"].get((g1, g2)) or s["pairs"].get((g2, g1))
            row[f"p_{g1}_vs_{g2}"] = round(pd_["p"], 4) if pd_ else ""
        rows.append(row)
    return pd.DataFrame(rows)


def print_validation(per_subj: pd.DataFrame) -> None:
    print("\n--- Validation ---")

    d = per_subj.copy()
    d["tst_diff_min"] = d["tst_min"] - d["sheet_tst_min"]
    d["se_diff"] = d["sleep_efficiency_policy_pct"] - d["sleep_efficiency_pct"]

    worst_tst = d.reindex(d["tst_diff_min"].abs().sort_values(ascending=False).index).head(5)
    print("TST (hypnogram) vs sheet tst_sec — 5 largest absolute differences (min):")
    for _, r in worst_tst.iterrows():
        print(f"  {r['subject_id']:<8} {r['group']:<4} hypno={r['tst_min']:7.1f}  "
              f"sheet={r['sheet_tst_min']:7.1f}  diff={r['tst_diff_min']:+6.1f}")

    worst_se = d.reindex(d["se_diff"].abs().sort_values(ascending=False).index).head(5)
    print("\nPolicy SE vs sheet sleep_efficiency_pct — 5 largest differences (pct pts):")
    for _, r in worst_se.iterrows():
        print(f"  {r['subject_id']:<8} {r['group']:<4} "
              f"policy={r['sleep_efficiency_policy_pct']:6.1f}  "
              f"sheet={r['sleep_efficiency_pct']:6.1f}  diff={r['se_diff']:+6.1f}  "
              f"(unscorable={r['unscorable_min']:.1f} min)")

    unscorable = d.loc[d["unscorable_min"] > 0].sort_values("unscorable_min", ascending=False)
    print(f"\nSubjects with unscorable (UNKNOWN) time: {len(unscorable)}")
    for _, r in unscorable.head(8).iterrows():
        print(f"  {r['subject_id']:<8} {r['group']:<4} {r['unscorable_min']:6.1f} min")

    qc = d.loc[(d["qc_overlap_sec"] > 5) | (d["qc_gap_sec"] > 5)]
    print(f"\nQC — subjects with >5 s of annotation overlap or uncovered gap: {len(qc)}")
    for _, r in qc.iterrows():
        print(f"  {r['subject_id']:<8} {r['group']:<4} "
              f"overlap={int(r['qc_overlap_sec'])} s  gap={int(r['qc_gap_sec'])} s")

    at36 = d.loc[d["subject_id"] == "AT36"]
    if len(at36):
        r = at36.iloc[0]
        print("\nAT36 anchor (expected from notes/tst_waso_unknown_handling.md: "
              "TST 343.5, WASO 74.0, SOL 84.0, unscorable 100.5, policy SE 80.4):")
        print(f"  TST={r['tst_min']:.1f}  WASO={r['waso_min']:.1f}  SOL={r['sol_min']:.1f}  "
              f"unscorable={r['unscorable_min']:.1f}  "
              f"policy SE={r['sleep_efficiency_policy_pct']:.1f}  "
              f"SPT={r['spt_min']:.1f}")


def main() -> None:
    df = load_subjects()
    print(f"Loaded {len(df)} subjects from sheet: "
          + ", ".join(f"{g}={int((df['group'] == g).sum())}" for g in GROUP_ORDER))

    per_subj = pd.DataFrame([per_subject_stats(r) for _, r in df.iterrows()])

    stats = {key: run_metric_stats(per_subj, key) for key, _, _ in METRICS}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subj_path = OUTPUT_DIR / "sleep_statistics_per_subject.csv"
    per_subj.to_csv(per_subj_path, index=False)

    stats_path = OUTPUT_DIR / "sleep_statistics_stats.txt"
    write_stats_report(per_subj, stats, stats_path)

    table_path = OUTPUT_DIR / "sleep_statistics_table.csv"
    build_summary_table(per_subj, stats).to_csv(table_path, index=False)

    print("\nGroup means (mean ± SD):")
    for key, label, dec in METRICS:
        s = stats[key]
        parts = [f"{label}:"]
        for g in GROUP_ORDER:
            vals = per_subj.loc[per_subj["group"] == g, key].dropna()
            parts.append(f"{g}={vals.mean():.{dec}f}±{vals.std(ddof=1):.{dec}f}")
        test_name = "One-Way ANOVA" if s["test"] == "ANOVA" else s["test"]
        parts.append(f"[{test_name} p={_format_p(s['p'])}]")
        print("  " + "  ".join(parts))
        for (g1, g2), pd_ in s["pairs"].items():
            tag = "*" if pd_["sig"] else "ns"
            print(f"      {g1} vs {g2}: p={pd_['p']:.4f} ({tag})")

    print_validation(per_subj)

    print(f"\nSaved: {per_subj_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
