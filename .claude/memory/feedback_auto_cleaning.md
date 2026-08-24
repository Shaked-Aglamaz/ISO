---
name: Auto-cleaning workflow — prefer conservative epochs
description: User manually reviews auto-cleaning output in step1 notebook, prefers narrow-but-accurate epochs over aggressive extension
type: feedback
originSessionId: b633163c-4101-49ba-84cf-b94af6d238ed
---
For `step1_auto_cleaning.py`, user prefers **conservative epoch boundaries** that can be manually extended, rather than aggressive auto-extension that over-shoots and has to be trimmed back.

**Why:** User always reviews epochs manually in `step1_manual_bad_channels.ipynb` after auto-cleaning (saves manual edits with `_manual.txt` suffix: `{sub}_cleaned_annotations_manual.txt`, `{sub}_bad_channels_manual.txt`). Extending auto-detected bad epochs in the Qt browser is easier than identifying and shrinking over-extended ones.

**How to apply:** When tuning epoch detection thresholds, err toward tighter boundaries. Primary detection (GFP z>10 + channel-spread check) is accurate and rarely has false positives. Boundary trimming is kept; boundary extension was tried but over-shot and was reverted. The notebook loads auto outputs directly, so user can extend annotations in MNE's browser.

**Channel detection v2 caveats:** Sometimes over-flags for subjects with many borderline channels (e.g., SC5 with 59 bad channels). User is less strict about channel review — slight over-flagging is acceptable since interpolation handles the extras. Key rule that helps most: `time_flagged` alone (>15% outlier windows) is sufficient to mark as crazy.
