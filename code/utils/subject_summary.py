"""
Generate an MD summary for all subjects in a_the_rest (all groups).
For each subject: N2%, number of N2 bouts >= 300s, number of bad channels.

Usage:
    python code/utils/subject_summary.py
"""

import ast
import os
import numpy as np

BASE_DIR = "I:/Shaked/ISO_data"
ELLA_DIR = "F:/gennadiy/ella_processed"

GROUPS = {
    "control_clean": f"{BASE_DIR}/control_clean/a_the_rest",
    "elderly_control_clean": f"{BASE_DIR}/elderly_control_clean/a_the_rest",
    "MCI_clean": f"{BASE_DIR}/MCI_clean/a_the_rest",
}

SLEEP_STAGES = {"Wake", "NREM1", "NREM2", "NREM3", "REM"}
STAGE_TO_NUM = {"Wake": 0, "NREM1": 1, "NREM2": 2, "NREM3": 3, "REM": 4}


def parse_annotations(annot_path):
    """Read MNE annotations file, return list of (onset, duration, description)."""
    annotations = []
    with open(annot_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            annotations.append((float(parts[0]), float(parts[1]), parts[2]))
    return annotations


def get_n2_stats(annot_path):
    """Return (n2_percent, n_bouts_ge_300s) from annotations file.
    N2 bouts are contiguous NREM2 segments (ignoring BAD annotations)."""
    annotations = parse_annotations(annot_path)

    sleep_annots = [(o, d, desc) for o, d, desc in annotations if desc in SLEEP_STAGES]
    if not sleep_annots:
        return None, None

    total_sleep_time = sum(d for _, d, _ in sleep_annots)
    n2_time = sum(d for _, d, desc in sleep_annots if desc == "NREM2")
    n2_pct = n2_time / total_sleep_time * 100 if total_sleep_time > 0 else 0

    # Find contiguous N2 bouts by merging adjacent NREM2 annotations
    n2_annots = sorted(
        [(o, d) for o, d, desc in sleep_annots if desc == "NREM2"], key=lambda x: x[0]
    )
    bouts = []
    for onset, dur in n2_annots:
        if bouts and abs(onset - (bouts[-1][0] + bouts[-1][1])) < 1.0:
            # Extend previous bout
            bouts[-1] = (bouts[-1][0], onset + dur - bouts[-1][0])
        else:
            bouts.append((onset, dur))

    n_bouts_ge_300 = sum(1 for _, d in bouts if d >= 300)

    return n2_pct, n_bouts_ge_300


def get_bad_channels_count(sub, sub_dir):
    """Get bad channel count from bad_channels.txt or pipeline.log."""
    # Try bad_channels.txt in subject dir first
    bc_path = os.path.join(sub_dir, f"{sub}_bad_channels.txt")
    if os.path.exists(bc_path):
        with open(bc_path, "r") as f:
            channels = [line.strip() for line in f if line.strip()]
        return len(channels), "txt"

    # Try pipeline.log from ella_processed
    pipeline_path = f"{ELLA_DIR}/{sub}/pipeline.log"
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "r") as f:
            content = f.read()
        if "Interpolated channels:" in content:
            for line in content.split("\n")[::-1]:
                if "Interpolated channels:" in line:
                    list_str = line.split("Interpolated channels:")[-1].strip()
                    try:
                        channels = ast.literal_eval(list_str)
                        return len(channels), "log"
                    except Exception:
                        pass

    return None, None


def main():
    rows = []

    for group_name, group_dir in GROUPS.items():
        if not os.path.exists(group_dir):
            continue
        subjects = sorted(
            [d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))]
        )
        for sub in subjects:
            sub_dir = os.path.join(group_dir, sub)
            annot_path = os.path.join(sub_dir, f"{sub}_annotations.txt")

            if not os.path.exists(annot_path):
                rows.append((group_name, sub, None, None, None, None))
                continue

            n2_pct, n_bouts = get_n2_stats(annot_path)
            n_bad, source = get_bad_channels_count(sub, sub_dir)
            rows.append((group_name, sub, n2_pct, n_bouts, n_bad, source))

    # Write MD file
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "subject_summary.md")
    output_path = os.path.normpath(output_path)

    with open(output_path, "w") as f:
        for group_name in GROUPS:
            group_rows = [r for r in rows if r[0] == group_name]
            if not group_rows:
                continue

            # Build cell values first to compute column widths
            headers = ["Subject", "N2 %", "N2 bouts >= 300s", "Bad channels", "Source"]
            table_rows = []
            for _, sub, n2_pct, n_bouts, n_bad, source in group_rows:
                table_rows.append([
                    sub,
                    f"{n2_pct:.1f}%" if n2_pct is not None else "N/A",
                    str(n_bouts) if n_bouts is not None else "N/A",
                    str(n_bad) if n_bad is not None else "N/A",
                    source if source else "-",
                ])

            # Compute max width per column
            widths = [max(len(h), max(len(r[i]) for r in table_rows)) for i, h in enumerate(headers)]

            def fmt_row(cells):
                return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

            f.write(f"## {group_name}\n\n")
            f.write(fmt_row(headers) + "\n")
            f.write("|" + "|".join("-" * (w + 2) for w in widths) + "|\n")
            for r in table_rows:
                f.write(fmt_row(r) + "\n")
            f.write("\n")

    print(f"Summary written to {output_path}")

    # Also print to console
    for group_name in GROUPS:
        group_rows = [r for r in rows if r[0] == group_name]
        if not group_rows:
            continue
        print(f"\n## {group_name}")
        print(f"{'Sub':>10} | {'N2%':>6} | {'N2>=300s':>8} | {'BadCh':>5} | Src")
        print("-" * 50)
        for _, sub, n2_pct, n_bouts, n_bad, source in group_rows:
            n2_str = f"{n2_pct:.1f}%" if n2_pct is not None else "N/A"
            bouts_str = str(n_bouts) if n_bouts is not None else "N/A"
            bad_str = str(n_bad) if n_bad is not None else "N/A"
            src_str = source if source else "-"
            print(f"{sub:>10} | {n2_str:>6} | {bouts_str:>8} | {bad_str:>5} | {src_str}")


if __name__ == "__main__":
    main()
