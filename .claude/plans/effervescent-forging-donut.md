# Methods revision — Yuval's review (C344, C345, C371, C395, C412 + inline notes)

## Context

Yuval Nir returned a review of the thesis (17 comments, 14 inline `##` notes, and 364 tracked
in-place edits), triaged in `thesis/reviews/yuval_review_triage.md`. This session handles **only the
Methods items he explicitly raised** — Doc sections 3.1–3.7 and `thesis/chapters/03_methods.md`.
Everything else in the triage (figures, Intro/Discussion expansion, references, aMCI relabelling,
⚑ items from our own internal check) stays out of scope.

The manuscript is the Google Doc **"Shaked's Thesis V2"**
(`1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`), edited in place; every prose change is mirrored
into `thesis/chapters/03_methods.md`. No local copy of the Doc is ever made.

Outcome: Methods 3.1–3.2 that state recruitment and diagnosis per group and per site, give real
numbers instead of "too many", justify the 210 min cut against the literature, correct Yuval's
mistaken average-reference concern, address apnea, and stop mixing Methods with Results.

**Missing facts are not guessed.** Anywhere information is unavailable, the text carries a literal
`[TO SUPPLY]` marker; the user fills these in a later session. This applies to all the participants
questions still open.

**Nothing is written to the Doc until the user approves the edits.** Every before/after is drafted
into `thesis/reviews/methods_edits_before_after.md` first, with enough surrounding context to read
it without opening the Doc. The user reads that file; only then does the Doc get touched.

---

## What the exploration established

| Finding | Consequence |
|---|---|
| **No coded subject-level cutoff exists** for "too many bad channels/epochs" — it was a case-by-case judgement recorded as free text in the subjects sheet. Only the *detection* thresholds are hard-coded. | C345 cannot quote an algorithm constant; the numbers have to be derived from the data (Step 2). |
| Retained cohort: bad channels 0–17.6 % of 176; rejected N2 time 0–25.8 %. Excluded subjects carry hand-typed channel counts (36–61 ch = 20.5–34.7 %) and **no** epoch numbers at all. | The bad-epoch boundary must be computed, not looked up. |
| **Sharon 2025 has no young adults** (controls 52–85, aMCI, AD). Our 35 young controls are a separate TASMC study. | 3.1's "The Tel Aviv participants were drawn from the same cohort detailed in our previous work" is wrong for the young group and must be narrowed to the two older groups; the young group's recruitment becomes `[TO SUPPLY]`. |
| Sharon 2025 gives the full Tel Aviv protocol: referral route, diagnosing unit, exclusion list, AHI ≥ 15 rule, NetAmps 300 / Cz / 1000 Hz / < 50 kΩ, and the exact scoring montage. | C344, the apnea note, the setup claim and the scoring note are answerable for Tel Aviv. |
| **Sydney is undocumented** in the repo — no recruitment, no diagnosis, no equipment; site is inferred from the subject-ID string. | `[TO SUPPLY]`. |
| User confirmed the two centres used **identical setups**, and that **all participants had AHI ≤ 15** following the Sharon 2025 protocol. | Both are stated as fact; the basis is recorded in the before/after file. |
| **Table 1 in the Doc is a native 10×4 table object**, not an image (`table:body:0` at index 8426). | Yuval's demand that the exclusion *row labels* carry the numbers is editable directly, is in scope, and is approved. |
| User confirmed **all subjects were sleep-scored the same way, as in Sharon 2025**. | The full AASM montage can be stated for both sites and all three groups without hedging. |
| Triage §3 only *summarises* his 364 tracked edits; the purely stylistic ones are written out nowhere. | They have to be read out of the .docx itself (Step 1). |

---

## Step 1 — Extract Yuval's tracked in-place edits for the Methods paragraphs

Triage §3 marks 5 of his paragraphs "identical" in V2 and 11 "lightly edited" — those sentences
survive in V2, so his wording still fits and should be applied. But §3 lists none of them
individually, so they must come from the source.

`python-docx` is not installed. Extract with `zipfile` + `xml.etree` over `word/document.xml` inside
`thesis/reviews/Shaked's Thesis_YN.docx`, walking `w:ins` (his insertions), `w:del` / `w:delText`
(his deletions), and reconstructing each paragraph's before → after. Restrict the output to the
**Methods** paragraphs — this session touches nothing else.

Known trap (from the earlier diff work): he also deleted at least one passage with Track Changes
**off**, so a `w:ins`/`w:del` walk alone can miss losses. Only a paragraph-level comparison against
the current Doc text catches those; note anything suspicious rather than assuming the walk is
complete.

Script goes in the scratchpad, not the repo.

## Step 2 — Compute the real percentages for the excluded subjects (feeds C345)

**Reuse `code/utils/cohort_overview.py`** rather than reimplementing anything —
`list_excluded_subjects()`, `find_annotations_file()`, `find_bad_channels_file()`,
`read_bad_channels()`, `parse_annotations()`, `bad_epoch_overlap_with_n2()`, `stage_durations()`,
`n2_bouts_ge_300()`, `N_SCALP_CHANNELS = 176`. Its `build_excluded_df()` deliberately records only
id/group/dir/reason, which is why these metrics are missing today.

For every subject in the three excluded folders: bad-channel count and % of 176; bad-epoch seconds
overlapping N2 and % of N2 time; clean N2 bouts ≥ 300 s; TST; the recorded reason; and a
`files_present` flag — some excluded subjects will lack `_bad_channels.txt` or
`_cleaned_annotations.txt`, and that coverage gap gets **reported, not silently dropped**. Print the
retained-cohort ranges from `overview/subjects.csv` alongside so any boundary is visible.

Also verify the three TST-excluded subjects' bout counts (HG78 and MCI13 expected at 2 bouts,
SM07 at 206.1 min) — this underpins the C371 argument.

Run with `PYTHONIOENCODING=utf-8`. **The numbers go to the user, who decides the C345 wording** —
both for the Methods sentence and for Table 1's row labels.

## Step 3 — Find a citation for the 210 min criterion (C371)

We treated 210 min as a literature baseline in an earlier session but never recorded a reference,
and `library.bib` has none. Search for sleep-research papers applying a ~210 min / 3.5 h minimum
total sleep time as an inclusion criterion, verify the paper actually states it, and cite it so the
justification reads as "we applied the conventional floor" rather than an invented cut. Add the
entry to `thesis/references/library.bib` in the existing format.

If no solid citation is found, say so rather than citing something approximate — the fallback is the
structural argument (≥ 3.5 h guarantees at least two full NREM–REM cycles) plus the empirical point
that the criterion was additive, not decisive.

## Step 4 — Draft every edit into `thesis/reviews/methods_edits_before_after.md`

One entry per change, each with: the section, a sentence or two of surrounding context so the
location is recognisable without searching the Doc, the **exact current text**, the **exact proposed
text**, and a one-line reason naming the comment it answers. Follows the format already used by
`thesis/reviews/flavio_edits_before_after.md`.

Entries to cover:

1. **C344** — 3.1 split into per-group, per-site paragraphs (healthy controls; MCI patients; apnea
   screening; MoCA + technical exclusions). MCI paragraph written *the way Sharon 2025 writes it*
   (referral route, diagnosing unit, clinical diagnosis) with **no named criterion asserted** —
   Sharon cites none. Young-controls and all Sydney details → `[TO SUPPLY]`.
2. **C345** — the "too many…" phrase replaced with numeric criteria as % of electrodes (of 176) and
   % of **N2** sleep time, per Step 2 and the user's decision.
3. **C345 (Table 1)** — the three exclusion row labels rewritten to carry the criterion
   (user-approved).
4. **C371** — the 210 min justification, with the Step 3 citation.
5. **Sydney scoring** — scoring stated as identical at both sites, and the AASM montage spelled out:
   F3/F4, C3/C4 and O1/O2 against the contralateral mastoid, with EOG and submental EMG, 30 s
   epochs, verified against the Pz spectrogram with the hypnogram overlaid. User confirmed all
   subjects were scored this way, as in Sharon 2025 — no `[VERIFY]` marker needed.
6. **Identical setups** — same setup at both sites, with the Sharon 2025 hardware (256-ch EGI, Cz
   reference, NetAmps 300, 1000 Hz, < 50 kΩ). Basis recorded: the user's confirmation.
7. **C395** — average reference computed over good channels only; bad channels excluded from the
   average and interpolated afterwards. Pipeline unchanged, nothing re-run.
8. **C412** — the Figure 1 overview paragraph deleted from 3.2. The Figure 1 image and caption stay
   put; moving them to Results is another session's job.
9. **Yuval's stylistic tracked edits** from Step 1, listed individually.

**→ Stop here. The user reads the file before anything touches the Doc.**

## Step 5 — Apply, then mirror

Doc first, then the identical wording into `thesis/chapters/03_methods.md` with a `v6 (2026-08-14)`
changelog line matching the existing v4/v5 convention.

Mechanics: prefer `findAndReplace` (index-independent) for text swaps; `getTableStructure` +
cell-level edits for Table 1; `findElement` + `deleteRange(start, textEnd + 1)` for the C412
paragraph deletion, done **last** with freshly fetched indices so no earlier edit shifts them.
Inserting after a heading needs `applyParagraphStyle{NORMAL_TEXT}` to avoid inheriting the heading
style. Step 3's citation is the only new reference; if it is added, check whether the superscript
numbering downstream of it shifts.

---

## Files touched

| File | Change |
|---|---|
| `thesis/reviews/methods_edits_before_after.md` | **new** — every before/after, written before any Doc edit |
| Google Doc `1YpXrDGFlzRk…` §3.1–3.2 + Table 1 | the approved edits, in place |
| `thesis/chapters/03_methods.md` | identical mirror + `v6` changelog line |
| `thesis/references/library.bib` | one entry, only if Step 3 finds a solid citation |
| scratchpad scripts | Steps 1–2, not added to the repo |

`thesis/figure_manifest.md` needs no edit — no caption changes in this pass.

## Verification

1. Re-read Doc §3.1–3.2 and Table 1, and confirm they are word-identical to `03_methods.md` and to
   the approved before/after file.
2. Confirm the deleted Figure 1 paragraph left no empty paragraph behind, and that the Figure 1
   image and caption are still present.
3. Confirm citation superscripts still run in order and that nothing shifted.
4. Confirm the italic runs in the 3.4 Gaussian formula are untouched.
5. Re-check every number in prose against its source: percentages against Step 2's table, 216.5 min
   and the bout counts against `thesis/low_bout_and_excluded_n2_table.md`, hardware and protocol
   wording against the Sharon 2025 extraction.
6. Grep `03_methods.md` for `[TO SUPPLY]` and `[VERIFY]` and list what remains outstanding, so the
   later fill-in session has a work list.

## Explicitly not done

Everything ⚑ in the triage; the aMCI relabelling pass; moving the Figure 1 asset to Results; the
figure/font work (C487, C502, C591, C592); Table 2; C429's ANCOVA; ethics/consent/funding statements
(S4). None are among the items scoped for this session.
