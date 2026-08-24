---
name: project_methods_revision_yuval
description: "Methods pass on Yuval's review DONE 2026-08-15 (C344/C345/C371/C395/C412) — plus three traps: the 30% bad-epoch criterion is aspirational, SM07 pins the 210-min rule, and n2_bouts_ge_300 doesn't split around BAD"
metadata: 
  node_type: memory
  type: project
  originSessionId: 98f7363a-0941-45c2-b541-544d8166489a
  modified: 2026-08-15T08:16:48.165Z
---

**Methods-only pass on Yuval's review, applied 2026-08-15** to the Doc ("Shaked's Thesis V2") and
mirrored to `thesis/chapters/03_methods.md` (now v6). Full before/after record:
**`thesis/reviews/methods_edits_before_after.md`** — read that rather than re-deriving.

Done: C344 (3.1 split into healthy-controls / MCI / apnea / MoCA-and-exclusions, per site),
C345 (numeric criteria + Table 1 row labels), C371 (210 min justified structurally), C395 (average
reference over good channels only — no re-run), C412 (overview paragraph out of 3.2), plus his
"identical setups" and "Sydney scoring, same?" notes and two stylistic tracked edits.
See [[project_yuval_review]] for the wider triage and [[reference_manuscript_gdoc]] for Doc mechanics.

**Three traps worth not rediscovering:**

1. **The `> 30 % of N2 time` bad-epoch criterion is aspirational, not measured.** All four
   bad-epoch exclusions currently sit *below* it (SM004 13.9 %, HR72 4.6 %, EL3042 4.0 %, YS2 no
   file) because their artifact marking was abandoned once they were written off, while the worst
   *retained* subject (el3007) is at 25.8 %. The user chose 30 % on the basis that it will hold once
   that marking is finished. **Anyone recomputing the percentages from the folders before then will
   find they contradict the stated criterion.** Bad channels are different — that boundary is real
   (retained max 17.61 %, exclusions ≥ 20.45 %; EG5 at 17.61 % is the lone exception).
2. **The 210-min TST rule cannot be collapsed into "clean bouts < 3".** SM07 has 4 clean bouts
   (32.1 min analysed), above the floor, and 32.1 min sits inside the retained range (min 22.8 min),
   so no bout- or duration-based criterion excludes it. Dropping the TST rule re-includes SM07 →
   MCI 30 → 31 → full V10/V11 re-run. The other three (SM0017, MCI13, HG78) had < 3 bouts anyway.
3. **`cohort_overview.n2_bouts_ge_300()` does NOT split N2 around BAD annotations**, so its bout
   counts are upper bounds, not the pipeline definition. Reimplement the split-merge-filter from
   `code/new_iso/mult_chan.py` when bout counts matter; doing so reproduces the ‡ values in
   `thesis/low_bout_and_excluded_n2_table.md` exactly.

Also: **`overview/subjects.csv` was deleted 2026-08-15** — it was a stale `cohort_overview.py`
output holding the pre-June 34/38/31 cohort and listing HG78 as retained. Nothing read it
(`sleep_stage_pies.py` only names it in a docstring). `overview/group_summary.csv` is stale for the
same reason. Regenerate by re-running `code/utils/cohort_overview.py` if ever needed.

**Sharon et al. 2025 (`thesis/references/slow_waves_MOCA_Sharon_2025.pdf`) is the source for the Tel
Aviv protocol** and now supplies Methods' recruitment, diagnosis, apnea screening (AHI ≥ 15),
hardware (NetAmps 300, Cz, 1000 Hz, < 50 kΩ) and scoring montage (F3/F4, C3/C4, O1/O2 vs
contralateral mastoid + EOG + submental EMG). Two things it does **not** contain: any named MCI
diagnostic criterion (no Petersen, no NIA-AA — so we cite none either, by user decision), and **any
young adults** — its controls are 52–85, so the old "the Tel Aviv participants were drawn from the
same cohort" sentence was wrong for our 35 young controls and has been narrowed to the older groups.

**Still `[TO SUPPLY]` in 3.1** (user fills in a later session): young-cohort recruitment/criteria;
Sydney healthy-older-adult recruitment/criteria; Sydney MCI recruitment, criteria and diagnosing
clinician. Also open: the two `#REF` citations Yuval asked for (N2 restriction "following previous
work"; the 300 s bout rule "following zurich procedures?").

**No citation exists for the 210-min floor** — four web searches found nothing citable. The insomnia
phase-3 trial threshold is real but is disorder-eligibility screening running the opposite
direction; don't cite it. Methods justifies 210 min structurally (> two full NREM–REM cycles at
90–100 min) with no reference.
