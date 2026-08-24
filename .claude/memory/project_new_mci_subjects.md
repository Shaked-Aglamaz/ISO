---
name: New MCI subjects from Zurich
description: Pipeline for new MCI subjects arriving as chunked FIF files on G:\Shaked_MCI_raw — concat, step0, auto-cleaning, now fully processed
type: project
originSessionId: aa491c21-6c1d-4a0f-9787-8dbd24523a46
---
New MCI subjects arrive as 25-channel chunk FIF files (8 chunks each, 176 total channels) on `G:\Shaked_MCI_raw\{SUB}\`.

**Concat pipeline**: `code/utils/concat_chunks.py` merges chunks into `{SUB}_176-head-ch_resample250_raw.fif`.

**Hypno naming**: Subject folder names have extra zeros (e.g., SM0017) but hypno files in `I:\Shaked\ISO_data\scoring\MCI\` use shorter names (e.g., SM17_hypno.txt). KS5 and SC5 stay as-is. All hypnos are 1-sec epoch, numeric format.

**Step0 adjustments**: Input is pre-resampled FIF (not MFF), face/neck/ear electrodes already excluded. Step0 only does notch+bandpass filtering, sleep scoring, and saving. No annotation files for these subjects. Output filename: `{SUB}_176-head-ch_resample250_filtered_scored_raw.fif`.

**Status as of 2026-04-17 — all subjects fully processed:**
- KS5, SM0016, SM0018, SM0019, SM0020, SM09: step0 → auto-cleaning → step2 → main_loop. Moved to main `MCI_clean/` dir.
- SC5: fully processed but excluded (23.3% ISFS detection). Moved to `MCI_clean/excluded/`.
- SM0017: excluded — only 1 clean bout >= 300s. In `MCI_clean/excluded/`.
- SM004: excluded. In `MCI_clean/excluded/`.
- SM006: concatenated but **no hypnogram available yet** — cannot run step0.

**Why:** These subjects were recorded on another computer with large MFF files that couldn't be copied directly, so they were resampled to 250 Hz and split into 25-channel chunks for transfer.

**How to apply:** When new subjects appear in G:\Shaked_MCI_raw, run concat_chunks.py first, then check for hypno in scoring/MCI/ before running step0, then auto-cleaning, then step2, then main_loop. Always use `PYTHONIOENCODING=utf-8` when running scripts.
