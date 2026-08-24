"""One-off: verify per-second hypnogram EL3029.txt agrees with the stage
annotations in EL3029_annotations.txt (ignoring BAD_EPOCH entries)."""
import numpy as np

HYP = r"I:/Shaked/ISO_data/scoring/young_control/EL3029.txt"
ANN = r"I:/Shaked/ISO_data/control_clean/a_excluded/EL3029/EL3029_annotations.txt"

STAGE_CODE = {"Wake": 0, "NREM1": 1, "NREM2": 2, "NREM3": 3, "REM": 4}

# --- hypnogram: epoch_idx -> code (1 s per epoch) ---
hyp = {}
with open(HYP) as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # support both "idx code" and single-column "code" formats
        if len(parts) == 2:
            hyp[int(parts[0])] = int(parts[1])
        else:
            hyp[i] = int(parts[0])
n_hyp = len(hyp)
print(f"Hypnogram epochs: {n_hyp} (idx {min(hyp)}..{max(hyp)})")

# --- annotations: build expected per-second stage array ---
intervals = []
with open(ANN) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        onset, dur, desc = line.split(",")
        if desc in STAGE_CODE:
            intervals.append((float(onset), float(dur), STAGE_CODE[desc]))

# total span
end = max(o + d for o, d, _ in intervals)
print(f"Annotation stage intervals: {len(intervals)}, span 0..{end:.0f}s")

# expected[second] (second s covers [s, s+1)); hyp epoch idx (1-based) -> second s = idx-1
expected = np.full(int(round(end)), -1, dtype=int)
for o, d, code in intervals:
    s0, s1 = int(round(o)), int(round(o + d))
    expected[s0:s1] = code

# --- compare ---
mismatches = []
uncovered = 0
for idx, code in hyp.items():
    sec = idx - 1  # 1-based epoch -> 0-based second
    if sec >= len(expected):
        continue  # beyond annotation span
    exp = expected[sec]
    if exp == -1:
        uncovered += 1
        continue
    if exp != code:
        mismatches.append((idx, sec, code, exp))

print(f"\nSeconds with no stage annotation (gap): {uncovered}")
print(f"Mismatches (stage disagreements): {len(mismatches)}")
inv = {v: k for k, v in STAGE_CODE.items()}
for idx, sec, code, exp in mismatches[:50]:
    print(f"  epoch {idx} (t={sec}s): hyp={inv.get(code,code)}  ann={inv.get(exp,exp)}")
if len(mismatches) > 50:
    print(f"  ... and {len(mismatches)-50} more")
