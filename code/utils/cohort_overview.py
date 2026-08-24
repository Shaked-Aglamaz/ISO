"""
Cohort overview: per-subject demographics + sleep + cleaning metrics.

Reads:
  - {BASE_DIR}/{group}/{subject}/{subject}_cleaned_annotations.txt (or _annotations.txt)
  - {BASE_DIR}/{group}/{subject}/{subject}_bad_channels.txt
  - overview/.demographics_cache.json (staged by Claude via google-sheets MCP)
  - notes/notes.txt (best-effort exclusion reasons)

Writes:
  - overview/subjects.csv          one row per active subject
  - overview/excluded.csv          one row per excluded subject
  - overview/group_summary.csv     descriptive stats per group
  - overview/_payload.json         all three tables, for Claude to push to Sheets

Usage:
  PYTHONIOENCODING=utf-8 python code/utils/cohort_overview.py [--group GROUP]
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))  # so `config` and `subject_summary` resolve as siblings

from config import (
    BASE_DIR,
    FACE_ELECTRODES,
    NECK_ELECTRODES,
    EAR_ELECTRODES,
)
from subject_summary import parse_annotations as _parse_annotations_raw

STAGE_ALIASES = {
    "WAKE": "Wake", "Wake": "Wake", "wake": "Wake",
    "N1": "NREM1", "NREM1": "NREM1",
    "N2": "NREM2", "NREM2": "NREM2",
    "N3": "NREM3", "NREM3": "NREM3",
    "REM": "REM", "rem": "REM",
}


def parse_annotations(path):
    """Wrap subject_summary.parse_annotations and normalize stage labels."""
    out = []
    for onset, dur, desc in _parse_annotations_raw(path):
        out.append((onset, dur, STAGE_ALIASES.get(desc, desc)))
    return out

GROUP_LABELS = OrderedDict([
    ("control_clean", "YA"),
    ("elderly_control_clean", "HE"),
    ("MCI_clean", "MCI"),
])
EXCLUDED_DIRS = {"a_excluded", "excluded", "excluded_AD"}
NON_SUBJECT_DIRS = EXCLUDED_DIRS | {"a_the_rest", "dashboards"}

SLEEP_STAGES = {"Wake", "NREM1", "NREM2", "NREM3", "REM"}
N_SCALP_CHANNELS = 176  # post-exclusion scalp channels including VREF

OVERVIEW_DIR = os.path.join(REPO_ROOT, "overview")
DEMO_CACHE = os.path.join(OVERVIEW_DIR, ".demographics_cache.json")
EXCLUSION_CACHE = os.path.join(OVERVIEW_DIR, ".exclusion_reasons_cache.json")
NOTES_PATH = os.path.join(REPO_ROOT, "notes", "notes.txt")

ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def canonical_key(subject_id: str):
    """Return (prefix_upper, int_suffix) for subject-id matching, or None if not parseable."""
    if not subject_id:
        return None
    m = ID_RE.match(subject_id.strip())
    if not m:
        return (subject_id.strip().upper(), None)
    return (m.group(1).upper(), int(m.group(2)))


def stage_durations(annotations):
    out = {s: 0.0 for s in SLEEP_STAGES}
    for _, dur, desc in annotations:
        if desc in out:
            out[desc] += dur
    return out


def annotation_span(annotations):
    if not annotations:
        return 0.0
    onsets = [o for o, _, _ in annotations]
    ends = [o + d for o, d, _ in annotations]
    return max(ends) - min(onsets)


def bad_epoch_overlap_with_n2(annotations):
    n2 = sorted([(o, o + d) for o, d, desc in annotations if desc == "NREM2"])
    bad = [(o, o + d) for o, d, desc in annotations if desc.startswith("BAD")]
    if not n2 or not bad:
        return 0.0
    total = 0.0
    for bs, be in bad:
        for ns, ne in n2:
            if ne <= bs:
                continue
            if ns >= be:
                break
            total += min(be, ne) - max(bs, ns)
    return total


def bad_epoch_total(annotations):
    return sum(d for _, d, desc in annotations if desc.startswith("BAD"))


def n2_bouts_ge_300(annotations, min_dur=300.0):
    """Contiguous NREM2 bouts (treats sub-second gaps as the same bout) of duration >= min_dur."""
    n2 = sorted([(o, d) for o, d, desc in annotations if desc == "NREM2"])
    bouts = []
    for onset, dur in n2:
        if bouts and abs(onset - (bouts[-1][0] + bouts[-1][1])) < 1.0:
            bouts[-1] = (bouts[-1][0], onset + dur - bouts[-1][0])
        else:
            bouts.append((onset, dur))
    return sum(1 for _, d in bouts if d >= min_dur)


def load_demographics():
    if not os.path.exists(DEMO_CACHE):
        raise SystemExit(
            f"Demographics cache not found at {DEMO_CACHE}.\n"
            "Re-run after Claude stages it via the google-sheets MCP."
        )
    with open(DEMO_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    layout = cache["_layout"]
    rows = cache["rows"]

    records = []
    for group_key, group_label in [("young", "YA"), ("elderly", "HE"), ("mci", "MCI")]:
        cols = layout[group_key]
        for r in rows:
            sid = r[cols["id"]] if cols["id"] < len(r) else ""
            if not sid or not sid.strip():
                continue
            rec = {
                "demo_id": sid.strip(),
                "demo_group": group_label,
                "age": _to_float(r[cols["age"]] if cols["age"] < len(r) else ""),
                "sex": _clean_str(r[cols["sex"]] if cols["sex"] < len(r) else ""),
                "moca": _to_float(r[cols["moca"]] if "moca" in cols and cols["moca"] < len(r) else ""),
                "comment": _clean_str(r[cols["comment"]] if "comment" in cols and cols["comment"] < len(r) else ""),
            }
            records.append(rec)
    df = pd.DataFrame(records)
    df["_key"] = df["demo_id"].apply(canonical_key)
    return df


def _to_float(s):
    s = (s or "").strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_str(s):
    s = (s or "").strip()
    return s if s else ""


def list_active_subjects():
    """Yield (group_dir, group_label, subject_id, subject_path) for every active subject."""
    for group_dir, label in GROUP_LABELS.items():
        root = os.path.join(BASE_DIR, group_dir)
        if not os.path.isdir(root):
            print(f"WARNING: group dir not found: {root}")
            continue
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if name in NON_SUBJECT_DIRS:
                continue
            yield group_dir, label, name, full


def list_excluded_subjects():
    for group_dir, label in GROUP_LABELS.items():
        for excl in EXCLUDED_DIRS:
            excl_path = os.path.join(BASE_DIR, group_dir, excl)
            if not os.path.isdir(excl_path):
                continue
            for name in sorted(os.listdir(excl_path)):
                full = os.path.join(excl_path, name)
                if not os.path.isdir(full):
                    continue
                yield group_dir, label, excl, name


def load_exclusion_reasons():
    """Returns dict mapping canonical_key -> reason string."""
    out = {}
    if not os.path.exists(EXCLUSION_CACHE):
        print(f"WARNING: exclusion-reasons cache not found at {EXCLUSION_CACHE} — reasons will be blank")
        return out
    with open(EXCLUSION_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    for entry in cache.get("reasons", []):
        sid = entry.get("id", "").strip()
        reason = entry.get("reason", "").strip()
        if not sid:
            continue
        key = canonical_key(sid)
        if key:
            out[key] = reason
    return out


def find_annotations_file(subject_dir, subject_id):
    cleaned = os.path.join(subject_dir, f"{subject_id}_cleaned_annotations.txt")
    if os.path.exists(cleaned):
        return cleaned, "cleaned"
    raw = os.path.join(subject_dir, f"{subject_id}_annotations.txt")
    if os.path.exists(raw):
        return raw, "raw"
    return None, "missing"


def find_bad_channels_file(subject_dir, subject_id):
    p = os.path.join(subject_dir, f"{subject_id}_bad_channels.txt")
    return p if os.path.exists(p) else None


def read_bad_channels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def per_subject_record(group_dir, group_label, subject_id, subject_dir, demo_df):
    rec = OrderedDict()
    rec["subject_id"] = subject_id
    rec["group"] = group_label
    rec["group_dir"] = group_dir

    key = canonical_key(subject_id)
    rec["match_key"] = f"{key[0]}:{key[1]}" if key else ""

    # Demographics by tuple match
    demo_match = demo_df[demo_df["_key"] == key] if key else demo_df.iloc[0:0]
    if len(demo_match) >= 1:
        d = demo_match.iloc[0]
        rec["age"] = d["age"]
        rec["sex"] = d["sex"]
        rec["moca"] = d["moca"]
        rec["demo_comment"] = d["comment"]
    else:
        rec["age"] = np.nan
        rec["sex"] = ""
        rec["moca"] = np.nan
        rec["demo_comment"] = ""

    # Annotations / sleep stats
    annot_path, annot_src = find_annotations_file(subject_dir, subject_id)
    rec["annotations_source"] = annot_src
    if annot_path is not None:
        annotations = parse_annotations(annot_path)
        durs = stage_durations(annotations)
        rec_secs = annotation_span(annotations)
        wake_sec = durs["Wake"]
        tst_sec = durs["NREM1"] + durs["NREM2"] + durs["NREM3"] + durs["REM"]
        n2_sec = durs["NREM2"]

        rec["recording_sec"] = round(rec_secs, 1)
        rec["tst_sec"] = round(tst_sec, 1)
        rec["wake_sec"] = round(wake_sec, 1)
        rec["pct_n1"] = _pct(durs["NREM1"], tst_sec)
        rec["pct_n2"] = _pct(durs["NREM2"], tst_sec)
        rec["pct_n3"] = _pct(durs["NREM3"], tst_sec)
        rec["pct_rem"] = _pct(durs["REM"], tst_sec)
        rec["pct_wake"] = _pct(wake_sec, rec_secs)
        rec["sleep_efficiency_pct"] = _pct(tst_sec, rec_secs)
        rec["n2_sec"] = round(n2_sec, 1)
        rec["n2_bouts_ge_300"] = n2_bouts_ge_300(annotations)

        bad_total = bad_epoch_total(annotations)
        bad_n2 = bad_epoch_overlap_with_n2(annotations)
        rec["bad_epoch_sec_total"] = round(bad_total, 1)
        rec["bad_epoch_sec_in_n2"] = round(bad_n2, 1)
        rec["pct_bad_epochs_in_n2"] = _pct(bad_n2, n2_sec)
    else:
        for k in ["recording_sec", "tst_sec", "wake_sec", "pct_n1", "pct_n2",
                  "pct_n3", "pct_rem", "pct_wake", "sleep_efficiency_pct",
                  "n2_sec", "n2_bouts_ge_300", "bad_epoch_sec_total",
                  "bad_epoch_sec_in_n2", "pct_bad_epochs_in_n2"]:
            rec[k] = np.nan

    # Bad channels
    bc_path = find_bad_channels_file(subject_dir, subject_id)
    if bc_path is not None:
        bad_chs = read_bad_channels(bc_path)
        rec["has_bad_channels_file"] = True
        rec["n_bad_channels"] = len(bad_chs)
        rec["pct_bad_channels"] = _pct(len(bad_chs), N_SCALP_CHANNELS)
    else:
        rec["has_bad_channels_file"] = False
        rec["n_bad_channels"] = np.nan
        rec["pct_bad_channels"] = np.nan

    return rec


def _pct(num, denom):
    if denom is None or denom == 0 or pd.isna(denom):
        return np.nan
    return round(100.0 * num / denom, 2)


def build_subjects_df(demo_df):
    rows = [per_subject_record(gd, gl, sid, sp, demo_df)
            for (gd, gl, sid, sp) in list_active_subjects()]
    df = pd.DataFrame(rows)
    return df


def build_excluded_df(reasons_map):
    rows = []
    for group_dir, group_label, excl_dir, sid in list_excluded_subjects():
        key = canonical_key(sid)
        rows.append({
            "subject_id": sid,
            "group": group_label,
            "group_dir": group_dir,
            "excluded_dir": excl_dir,
            "reason": reasons_map.get(key, ""),
        })
    return pd.DataFrame(rows)


def build_group_summary(subjects_df):
    numeric_cols = [
        "age", "moca", "recording_sec", "tst_sec", "wake_sec",
        "pct_n1", "pct_n2", "pct_n3", "pct_rem", "pct_wake",
        "sleep_efficiency_pct", "n2_sec", "n2_bouts_ge_300",
        "n_bad_channels", "pct_bad_channels",
        "bad_epoch_sec_total", "bad_epoch_sec_in_n2", "pct_bad_epochs_in_n2",
    ]
    rows = []
    for label in GROUP_LABELS.values():
        sub = subjects_df[subjects_df["group"] == label]
        for col in numeric_cols:
            vals = sub[col].dropna()
            rows.append({
                "group": label,
                "metric": col,
                "n_non_null": int(vals.shape[0]),
                "mean": round(vals.mean(), 3) if len(vals) else np.nan,
                "sd": round(vals.std(ddof=1), 3) if len(vals) > 1 else np.nan,
                "median": round(vals.median(), 3) if len(vals) else np.nan,
                "min": round(vals.min(), 3) if len(vals) else np.nan,
                "max": round(vals.max(), 3) if len(vals) else np.nan,
            })
        # Sex breakdown
        n_total = len(sub)
        n_male = int((sub["sex"].str.upper() == "M").sum())
        n_female = int((sub["sex"].str.upper() == "F").sum())
        n_missing = n_total - n_male - n_female
        rows.append({
            "group": label,
            "metric": "sex_n_male",
            "n_non_null": n_total,
            "mean": n_male,
            "sd": round(100.0 * n_male / n_total, 1) if n_total else np.nan,
            "median": np.nan, "min": np.nan, "max": np.nan,
        })
        rows.append({
            "group": label,
            "metric": "sex_n_female",
            "n_non_null": n_total,
            "mean": n_female,
            "sd": round(100.0 * n_female / n_total, 1) if n_total else np.nan,
            "median": np.nan, "min": np.nan, "max": np.nan,
        })
        rows.append({
            "group": label,
            "metric": "sex_n_missing",
            "n_non_null": n_total,
            "mean": n_missing,
            "sd": round(100.0 * n_missing / n_total, 1) if n_total else np.nan,
            "median": np.nan, "min": np.nan, "max": np.nan,
        })
    return pd.DataFrame(rows)


def report_unmatched(subjects_df, demo_df):
    matched_keys = set(subjects_df["match_key"]) - {""}
    demo_df = demo_df.copy()
    demo_df["_keystr"] = demo_df["_key"].apply(lambda k: f"{k[0]}:{k[1]}" if k else "")
    sheet_keys = set(demo_df["_keystr"]) - {""}

    folders_no_demo = subjects_df[subjects_df["age"].isna()]["subject_id"].tolist()
    sheet_only = sheet_keys - matched_keys
    sheet_only_ids = demo_df[demo_df["_keystr"].isin(sheet_only)]["demo_id"].tolist()

    print(f"\n=== Subject ID join report ===")
    print(f"Active folders:  {len(subjects_df)}")
    print(f"Demo rows:       {len(demo_df)}")
    print(f"Matched:         {len(matched_keys & sheet_keys)}")
    print(f"Folders missing demographics ({len(folders_no_demo)}): {folders_no_demo}")
    print(f"Sheet IDs with no folder ({len(sheet_only_ids)}):     {sheet_only_ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="Restrict to one group label (YA/HE/MCI)")
    args = ap.parse_args()

    os.makedirs(OVERVIEW_DIR, exist_ok=True)
    demo_df = load_demographics()
    reasons_map = load_exclusion_reasons()

    subjects_df = build_subjects_df(demo_df)
    if args.group:
        subjects_df = subjects_df[subjects_df["group"] == args.group].reset_index(drop=True)
    excluded_df = build_excluded_df(reasons_map)
    group_summary_df = build_group_summary(subjects_df)

    report_unmatched(subjects_df, demo_df)

    # Drop helper columns from on-disk outputs (keep in memory for debugging only)
    subjects_out = subjects_df.drop(columns=[c for c in ["match_key"] if c in subjects_df.columns])

    subjects_out.to_csv(os.path.join(OVERVIEW_DIR, "subjects.csv"), index=False)
    excluded_df.to_csv(os.path.join(OVERVIEW_DIR, "excluded.csv"), index=False)
    group_summary_df.to_csv(os.path.join(OVERVIEW_DIR, "group_summary.csv"), index=False)

    payload = {
        "subjects": _df_to_payload(subjects_out),
        "excluded": _df_to_payload(excluded_df),
        "group_summary": _df_to_payload(group_summary_df),
    }
    with open(os.path.join(OVERVIEW_DIR, "_payload.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nWrote: {OVERVIEW_DIR}/subjects.csv ({len(subjects_out)} rows)")
    print(f"Wrote: {OVERVIEW_DIR}/excluded.csv ({len(excluded_df)} rows)")
    print(f"Wrote: {OVERVIEW_DIR}/group_summary.csv ({len(group_summary_df)} rows)")
    print(f"Wrote: {OVERVIEW_DIR}/_payload.json (for MCP push)")
    print(f"\nN_SCALP_CHANNELS denominator = {N_SCALP_CHANNELS} (post-exclusion scalp channels including VREF)")


def _df_to_payload(df):
    """Convert DataFrame to list-of-rows with NaN -> empty string for Sheets."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype.kind in "fc":
            df_clean[col] = df_clean[col].astype(object).where(df_clean[col].notna(), "")
        else:
            df_clean[col] = df_clean[col].fillna("")
    return {
        "header": list(df_clean.columns),
        "rows": df_clean.values.tolist(),
    }


if __name__ == "__main__":
    main()
