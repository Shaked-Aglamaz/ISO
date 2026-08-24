# Discussion revision — Yuval's review (C557, C558, C571, two new paragraphs, bandwidth correction)

## Context

Yuval returned his review on 2026-08-11 (17 comments, 364 tracked insertions, 9-item email list).
The Methods and Results passes are already applied (`thesis/reviews/methods_edits_before_after.md`,
`results_edits_before_after.md`). **This pass closes the Discussion.**

Five things are open, all from `thesis/reviews/yuval_review_triage.md`:

| Item | What he said | Where it lands |
|---|---|---|
| **C557** | The cholinergic/REM passage is "out of context… a much more relevant connection could be literature on LC degeneration (NoaR can share our review) and changes in NREM sleep (for example Omer's paper on slow wave activity)" | Discussion ¶4 (rewrite) |
| **C558** | "I don't see how a more diffuse IFSF hotspot connects to thalamocortical mechanisms… it's really unclear what you think is going on" | Discussion ¶4, final sentence |
| **C571** | "Not only earlier. Some aMCI will never develop AD they could develop other neurodegenerative disorders" | Discussion ¶6, "perhaps because MCI is an earlier disease stage" |
| **His `##` note** | Two missing paragraphs: (1) broader young/old/MCI sleep changes beyond ISFS, incl. **CAP**; (2) what is known about ISFS irrespective of aging/MCI — same methods? same parameters? | Two new paragraphs |
| **C429 consequence** | — (not his words; a *consequence* of his C429, flagged in triage §8.2.1) | ¶2 and ¶7 — the Discussion currently contradicts the revised Results on bandwidth |

Plus his surviving in-place tracked edits inside the Discussion, extracted from
`thesis/reviews/Shaked's Thesis_YN.docx` (paras 159–166).

**Intended outcome:** a Discussion that answers every Discussion item he raised, no longer contradicts
Results 4.1/4.2 on bandwidth, and carries the citations those arguments actually need.

---

## Decisions taken

1. **Workflow — review file first.** Draft `thesis/reviews/discussion_edits_before_after.md` with every
   before/after block; **stop and wait for approval**; only then touch the Doc and mirror to
   `thesis/chapters/05_discussion.md`. (Standing rule; same as the Methods and Results passes.)
2. **ISO naming.** State the ISFS/ISO equivalence once. Use "ISO" only when describing studies that use
   that term; **our own work is always ISFS, never ISO.**
3. **Citations — only where the argument needs them.** No padding to hit 50. The Intro pass follows;
   the ≥50 count gets checked once all parts are done. Expected here: **~22–26 new**, Doc 30 → ~52–56.
4. **⚠ C558 is deferred — you asked to be reminded.** The review file will open with this decision,
   both options drafted in full, recommendation = make the mechanism explicit. **Nothing is applied
   until you pick.**

---

## Scope

Only what Yuval explicitly asked. Items marked ⚑ in the triage stay out. Things spotted but **not
touched** are listed in §7 below for you to rule on.

---

## The work

### 1. `05_discussion.md` ¶4 — C557 rewrite (LC replaces the ACh/REM lead)

**Current** (Doc: "The neuromodulatory systems that pace and generate spindles are a likely place…")
runs `schmitz2018cholinergic` → `andre2025remslowing` — exactly the ACh/REM material he called out.

**Replacement argument**, built from `thesis/references/new_refs_annotated.md` §J:

- The LC clocks the infra-slow sigma rhythm, and **the direction matters**: NE peaks coincide with
  micro-arousals, NE troughs with spindles — LC infra-slow activity is *anti-correlated* with sigma
  power. The current text never says this. → `osorioforero2025gatekeeper`, `kjaerby2022norepinephrine`
  (already cited in Intro).
- The LC is the **earliest** site of AD-type tau (`braak2011stages`) and loses more neurons than the
  cholinergic nucleus basalis (`zarow2003neuronalloss` — this is what licenses leading with LC).
- It lands in humans: LC integrity tracks memory and sleep continuity in older adults
  (`dahl2019rostrallc` / `vanegroo2021awakenings`), and LC abnormality **in aMCI specifically**
  separates future progressors (`galgani2023locuscoeruleus`).
- Weakened noradrenergic infra-slow fluctuations occur in neurodegenerative disease
  (`luthi2025microarousals` — load-bearing).
- **ACh is reframed, not deleted:** `kjaerby2026neuromodulators` shows ACh oscillates infra-slowly
  *during NREM under LC control*, so `schmitz2018cholinergic` survives as one arm of a system
  oscillating on this thesis's own timescale, with `andre2025remslowing` demoted to a brief parallel.
- **The NREM half he named:** `sharon2025slowwaves` (Omer's paper) is currently cited *only* as cohort
  provenance in Methods 3.1. It gets used substantively — same lab, same cohort, a slow-wave measure
  that **did** track prodromal AD where our ISFS did not separate aMCI.
- One clause noting the LC literature defines sigma as 10–16 Hz against our 13–16 Hz.

**Why this is the right chain:** the structure that paces the rhythm is the structure that degenerates
earliest — which predicts the effect shows up in *healthy aging* rather than at the MCI stage. That is
the thesis's actual result, so C557 and the AGING-not-MCI framing reinforce each other.

### 2. ¶4 final sentence — C558 ⚠ YOUR DECISION

Anchor: *"…a faster and spatially more diffuse ISFS points to a thalamocortical and neuromodulatory
infrastructure whose temporal and spatial organization coarsens with age…"*

- **Option A (recommended) — make it explicit.** The LC chain above supplies the missing step, plus
  the spatial half: the young-adult hotspot sits where fast-spindle generators are densest, so its
  flattening means the rhythm's modulation depth is no longer regionally differentiated. State which
  step is inferred rather than measured.
- **Option B — drop the claim.** Delete the thalamocortical assertion and its echo in the closing
  paragraph; keep only what the data support.
- **Option C — explicit, framed as a testable prediction.**

Both A and B will be drafted in full in the review file so you can read the actual sentences.

### 3. ¶6 — C571 rewrite

Replace *"perhaps because MCI is an earlier disease stage"* with the aetiological-mixture reading,
which does **not** assume the AD continuum he rejects: a group recruited by *cognitive* criteria is
aetiologically mixed, so an AD-specific ISFS effect would be diluted.

Numbers ready in the §G dossier — `mitchell2009progression` (most MCI never converts; annual conversion
9.6% specialist / 4.9% community; conversion is not only to AD), `malekahmadi2016reversion` (~24% of
aMCI reverts; 14% clinic vs 31% community — *ours is clinic-recruited*), `jicha2006neuropathologic`
(29% of aMCI converters had non-AD primary pathology at autopsy, and neither demographics nor cognitive
scores predicted which), `ferman2013nonamnestic` (the fair counter-evidence: subtype *does* predict).

Position: **"aMCI is enriched for AD without being equivalent to it."**
Same citations also fix the currently-uncited heterogeneity claim in ¶7.

### 4. Two new paragraphs

Recommended placement — after ¶6 (Grollero), before Limitations:

**NEW ¶ — ISFS outside aging and MCI** (his ask 2). Three clinical populations, two research
communities, two methods, effects in opposite directions:
- `dimitriades2025schizophrenia` — the strongest single point: reduced ISFS strength **specifically
  over central-parietal electrodes**, no correlation with clinical characteristics. Structurally the
  same result as ours (weakened central-parietal hotspot + null MoCA). Band is 10–16 Hz, note it.
- `sun2026longcovid` — ISO power **elevated** in ME/CFS (slow sigma 11–13 Hz, not our fast band).
- `liu2026autism` — the rhythm is measurable in early childhood; **do not overstate**, the group
  difference did not survive correction.
- **The method split is the real answer to his question.** Only the Zurich/Bristol lineage fits a
  Gaussian, so *peak frequency and bandwidth have essentially no comparison literature outside it*;
  strength is the only parameter with cross-study currency, and even it varies (AUC / peak amplitude /
  relative band power). This strengthens rather than weakens the thesis.
- Direction is inconsistent across conditions — reduced in schizophrenia and aging, elevated in ME/CFS
  — which is exactly the inverted-U `luthi2025microarousals` proposes. Gives the paragraph a
  conclusion rather than a list.
- **The ISFS/ISO equivalence sentence goes here.**

**NEW ¶ — broader sleep changes beyond ISFS, incl. CAP** (his ask 1). Spindles irrespective of ISFS
(reuse `mander2017aging`, `champetier2023spindlememory`, `helfrich2018uncoupled`, `gorgoni2016parietal`,
`zhang2022alzheimerreview` — all already cited), then CAP: `terzano2001cap` (what it is),
`parrino2012cap` (the marker of NREM instability; age/disease), `maestri2015capmci` (**engage with
directly** — same contrast as ours, an infra-slow NREM-instability measure, and it *did* find an
MCI difference where our ISFS did not; n = 11/group, different construct — a genuine contrast, not a
contradiction), `zheng2026capdementia` (CAP predicts incident dementia better than conventional sleep
parameters), `silvani2026cycles` (the bridge — already places CAP and infra-slow sigma in one
framework), `parrino2025phasic` (the CAP authorities themselves link CAP periodicity to LC infra-slow
activity). Optionally `drozario2020objectivesleep` for placing the null MCI result.

### 5. Bandwidth — the C429 consequence (triage §8.2.1)

The Discussion currently contradicts Results 4.1/4.2. Two fixes:

- **¶2:** *"A tendency toward a broader spectral peak pointed the same way but did not reach
  significance"* → bandwidth was numerically broader, but tracks how much N2 each participant
  contributed (r = 0.386; covariate p = 0.0002, partial η² = 0.128 — larger than the group effect) and
  the group difference goes 0.061 → **0.206** with the covariate. Not an age effect.
- **¶7 limitations:** currently lists bandwidth and the ROI null as "two effects that fell short of
  significance". Bandwidth moves out of that pairing into a **methodological caveat about the bandwidth
  estimator** (longer recordings → wider fitted peaks; a spectral-resolution/averaging property, not
  biology — `c429_c390_results.md` §1.3). The ROI null stays as the sub-threshold effect.

### 6. His surviving tracked edits inside the Discussion

Extracted from paras 159–166. Most of his opening-paragraph rewrite is **superseded** — V2 already
restructured into "First… Second…", already dropped "negative result", already relabelled aMCI. What
still lands verbatim:

| Para | Edit | Status |
|---|---|---|
| 160 → ¶3 | "thalamocortical **machinery** that organize**s**" → "**mechanisms** that organize" | apply |
| 160 → ¶3 | "In young adults the ISFS" → "In young adults**,** the ISFS" | apply |
| 163 → ¶6 | "the same **family** of ISFS features" → "the same **collection** of ISFS features" | apply |
| 159 → ¶1/¶2 | his "First… Second…" restructure, aMCI relabel, "dominant factor is age" | already in V2 |
| 162 → ¶5 | "In our hands…" / "While…" restructure | V2 rewrote; offer his phrasing, recommend keeping V2's |
| 164 → ¶7 | no run edits — only the paragraph split that carries his `##` note | nothing to apply |

Do **not** re-add the clauses V2 deliberately removed on Flavio's review (the "sensitive to which
subjects were included" trend hedge; "its insensitivity to MCI is itself informative about what the
rhythm indexes").

**Note on his placement signal:** he inserted the `##` note *after* the Limitations paragraph. The
review file will flag this — recommendation is still before Limitations.

### 7. Reference renumbering — the fragile part

The Doc's reference list is an auto-numbered Docs list ordered by **first appearance**; the in-text
markers are literal Unicode superscript glyphs that must be retyped by hand.

- **Refs 1–26 are frozen** — all first appear in the Abstract, Intro, Methods, Results, or Discussion
  ¶3, which sits before every new citation. (Verified against the live Doc.)
- **Refs 27–30 will shift**: `schmitz2018cholinergic` (27), `andre2025remslowing` (28),
  `niethard2023spindleaging` (29), `grollero2026iso` (30). Each shifts by a different amount depending
  on how many new citations land before it inside the rewritten ¶4.
- **Procedure:** build the complete final ordering table *first*, in the review file, before any edit.
  Then renumber existing markers **descending** (highest first) so no two ever collide, anchoring each
  `findAndReplace` on surrounding words rather than the bare glyph. Insert new markers and list entries
  last. Re-check list order afterwards.

### 8. Docs API mechanics

- `findAndReplace` cannot match across a paragraph mark → to delete a paragraph use `findElement` then
  `deleteRange(start, textEnd + 1)`.
- Inserting after a heading inherits the heading style → follow with
  `applyParagraphStyle{namedStyleType: NORMAL_TEXT}`.
- Superscript citation markers carry **no** formatting — replacements can span them freely.
- The doc uses **straight** apostrophes, not curly.
- Inserting a reference entry after one ending in a URL: insert at the *start of the following*
  paragraph, or the new text inherits the hyperlink run.
- Never download or export the Doc.

---

## Spotted but NOT done — your call

Per the scope rule, these are flagged rather than acted on:

1. **¶8 cites nothing** for "the infra-slow hemodynamic fluctuations that share its timescale".
   `fultz2019coupled` is in the library and fits exactly. Not a Yuval ask.
2. **¶7's heterogeneity claim is uncited** — but the C571 citations fix it as a side effect, so this
   one *is* covered.
3. **Stale reference metadata** (our own B1/S2 list, not his): `dimitriades2024isfs` is now *Sci Rep*
   2026, `andre2025remslowing` is now *Mol Psychiatry* 2026 (title + authors changed too),
   `sharon2025slowwaves` is missing volume/issue/pages (21(5):e70247).
4. **`tallonbaudry1997gamma`** is uncited anywhere despite Methods 3.4 describing a Gabor–Morlet
   wavelet at 4 cycles.
5. **Typo** in `thesis/figure_manifest.md:60` — "Dimitriadesographics" (find/replace artefact).
6. **Triage §8.2.5**, unresolved on purpose: ¶7 calls the cohort "clinically and etiologically
   heterogeneous" right beside a blanket "aMCI" label.

---

## Files

| File | Change |
|---|---|
| `thesis/reviews/discussion_edits_before_after.md` | **new** — the deliverable of this pass |
| Google Doc `1YpXrDGFlzRk…` Discussion + reference list | edited in place, after approval |
| `thesis/chapters/05_discussion.md` | mirrored, `[@key]` form, + a v4 changelog note |
| `thesis/reviews/yuval_review_triage.md` | STATUS table + §8.2 marked done |

Read-only inputs: `new_refs_annotated.md` §§G/H/I/J, `library.bib`, `c429_c390_results.md`,
`Shaked's Thesis_YN.docx`. Nothing is committed; nothing is deleted.

---

## Verification

1. **Before approval** — the review file itself is the check: every entry carries section, surrounding
   context, exact current text, exact proposed text, and the comment it answers.
2. **Reference integrity** — after applying, re-read the Doc and confirm (a) the highest superscript
   equals the number of entries in the list, (b) no number appears twice, (c) every new entry is in
   first-appearance order, (d) the AMA format matches its neighbours.
3. **Contradiction check** — grep the Doc and `05_discussion.md` for "tendency", "fell short",
   "broader spectral peak"; confirm no surviving sentence presents bandwidth as an age effect. Confirm
   ¶2/¶7 agree with Results 4.1 and 4.2 and with the Figure 3 caption.
4. **Terminology check** — grep for "ISO" in both; confirm every occurrence describes *other people's*
   studies, never ours, and that the equivalence sentence exists exactly once.
5. **Mirror check** — diff the Doc's Discussion text against `05_discussion.md` with citation markers
   stripped; they must be identical (this is the check that caught his silent Results 4.7 deletion).
6. **Answer check** — walk C557, C558, C571 and both `##` sub-asks and point at the sentence that
   answers each.
7. **Count** — report the new reference total so the Intro pass knows how far from 50 it starts.
