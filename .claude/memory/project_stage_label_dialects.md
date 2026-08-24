---
name: Annotation stage-label dialects
description: cleaned_annotations.txt files use two different stage-label naming schemes across subjects — must normalize before counting
type: project
originSessionId: db1f20fb-54ce-40fd-ac84-00d68d271f7d
---
Per-subject `{subject}_cleaned_annotations.txt` files use one of two label dialects for sleep stages:

- **Long form** (most subjects): `Wake`, `NREM1`, `NREM2`, `NREM3`, `REM`
- **Short form** (~15 elderly subjects: AT36, BA11, CH53, CH66, EB61, EF58, IS74, LOH91, LV93, ME5, RSH71, RY42, SC59, SL44, TZ7): `WAKE`, `N1`, `N2`, `N3`, `REM`

**Why:** Different scoring/conversion pipelines were used over time; never harmonized on disk.

**How to apply:** Before computing %N1/N2/N3/REM, sleep efficiency, n2-bouts, etc., normalize labels — `WAKE→Wake`, `N1→NREM1`, `N2→NREM2`, `N3→NREM3`. Without normalization those subjects look REM-only with TST<5000s. `code/utils/subject_summary.py` (and anything importing its `parse_annotations`/`get_n2_stats`) hits this bug; `code/utils/cohort_overview.py` wraps `parse_annotations` with a STAGE_ALIASES map and is the reference implementation.

BAD-annotation labels also vary: `BAD`, `BAD_EPOCH`, `BAD_ACQ_SKIP`. Match by `desc.startswith("BAD")` rather than equality.
