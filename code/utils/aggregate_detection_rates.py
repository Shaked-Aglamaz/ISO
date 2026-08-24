"""Aggregate per-group ISFS detection rates and bout counts from sigma_fix results.

Reads each subject's *_all_channels_summary.csv header and reports, per group:
  - subjects processed, ISFS-present (>=20% channels) vs ISFS-absent (<20%)
  - detection rate (% of valid channels) mean/SD/range across included subjects
  - clean NREM2 bout count mean/SD/range

Read-only. Run from repo root with PYTHONIOENCODING=utf-8.
"""
import os
import re
import statistics as st

BASE = r"I:/Shaked/ISO/results"
GROUPS = {
    "Young":   "sigma_fix_YA",
    "Elderly": "sigma_fix_HE",
    "MCI":     "sigma_fix_MCI",
}
# Cohort comes from the data dirs (same source-of-truth as step4/5/6): the active
# subjects are the direct child folders here, intersected with having ISFS results
# in sigma_fix. Excluded subjects live in nested a_excluded/excluded subdirs, so
# they are never picked up.
DATA_BASE = r"I:/Shaked/ISO_data"
DATA_DIRS = {
    "Young":   "control_clean",
    "Elderly": "elderly_control_clean",
    "MCI":     "MCI_clean",
}
INCLUSION_PCT = 20.0  # subjects with < this % of channels detected are ISFS-absent

det_re = re.compile(r"Channels with ISFS detected:\s*(\d+)\s*\(([\d.]+)%\)")
valid_re = re.compile(r"Total valid channels analyzed:\s*(\d+)")
bouts_re = re.compile(r"Number of bouts:\s*(\d+)")


def parse_subject(path):
    det_pct = det_n = valid = bouts = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = det_re.search(line)
            if m:
                det_n, det_pct = int(m.group(1)), float(m.group(2))
            m = valid_re.search(line)
            if m:
                valid = int(m.group(1))
            m = bouts_re.search(line)
            if m:
                bouts = int(m.group(1))
    return det_pct, det_n, valid, bouts


def fmt(vals):
    return (f"mean {st.mean(vals):.1f} +/- {st.pstdev(vals):.1f} SD, "
            f"range {min(vals):.1f}-{max(vals):.1f}")


for gname, gdir in GROUPS.items():
    root = os.path.join(BASE, gdir)
    data_root = os.path.join(DATA_BASE, DATA_DIRS[gname])
    subjects = []
    for subj in sorted(os.listdir(data_root)):
        if subj == "dashboards" or not os.path.isdir(os.path.join(data_root, subj)):
            continue
        sumcsv = os.path.join(root, subj, f"{subj}_all_channels_summary.csv")
        if not os.path.exists(sumcsv):
            continue  # active subject without ISFS results (or a non-subject folder)
        det_pct, det_n, valid, bouts = parse_subject(sumcsv)
        if det_pct is None:
            print(f"  [WARN] no detection header: {sumcsv}")
            continue
        subjects.append((subj, det_pct, det_n, valid, bouts))

    present = [s for s in subjects if s[1] >= INCLUSION_PCT]
    absent = [s for s in subjects if s[1] < INCLUSION_PCT]

    print("=" * 70)
    print(f"{gname}  ({gdir})")
    print("-" * 70)
    print(f"  subjects processed : {len(subjects)}")
    print(f"  ISFS-present (>={INCLUSION_PCT:.0f}%) : {len(present)}")
    print(f"  ISFS-absent  (<{INCLUSION_PCT:.0f}%)  : {len(absent)}"
          + (f"  -> {[s[0] for s in absent]}" if absent else ""))
    if present:
        det = [s[1] for s in present]
        bouts = [s[4] for s in present if s[4] is not None]
        print(f"  detection rate (% of valid ch) : {fmt(det)}")
        if bouts:
            print(f"  bout count                     : {fmt(bouts)}  "
                  f"(total {sum(bouts)})")
print("=" * 70)
