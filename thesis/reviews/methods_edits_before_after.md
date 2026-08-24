# Methods edits — before / after, for review before anything touches the Doc

**Target:** Google Doc "Shaked's Thesis V2" (`1YpXrDGFlzRk…`), sections 3.1–3.2 and Table 1.
Mirror: `thesis/chapters/03_methods.md`. Drafted 2026-08-14. **APPLIED to the Doc and mirrored to
the chapter on 2026-08-15**, after your decisions on §A and §B. Decisions taken are recorded inline
below; the before/after text is what actually went in.

Scope = only the Methods items Yuval explicitly raised: **C344, C345, C371, C395, C412**, his inline
notes on apnea / Sydney scoring / "mixing Methods and Results" / "identical setups", plus his
stylistic tracked edits inside Methods.

Read order: **§A and §B are decisions I need from you.** §C is the proposed text. §D is his tracked
edits. §E is the list of gaps left as `[TO SUPPLY]`. **§F is a later fill-in (2026-08-17) closing
§E item 1.**

---

## §A — DECISION 1: the numbers for C345

Computed from the subject folders as they stand now, using `code/utils/cohort_overview.py`'s own
metric functions. Sanity check: the retained-cohort means come out at 6.30 / 1.43 / 3.37 % channels
and 4.13 / 3.32 / 1.80 % epochs — matching the figures already in Methods 3.2 (6.3 / 1.4 / 3.4 and
4.1 / 3.3 / 1.8), so the computation is consistent with what is published.

*(Note: `overview/subjects.csv` is stale — it still holds the pre-June cohort of 34/38/31 and lists
HG78 as retained. Everything below is recomputed, not read from it.)*

### A1. Bad channels — a clean boundary does exist

| | worst retained | | | first excluded | |
|---|---|---|---|---|---|
| MCI01 | **17.61 %** (31/176) | | EG5 | 17.61 % (31/176) | ← the one that muddies it |
| KS5 | 14.20 % (25) | | EL3045 | **20.45 %** (36) | |
| EL3025 / EL3034 / EL3036 | 13.07 % (23) | | SC5 | 23.86 % (42) | |
| EL3044 | 10.80 % (19) | | MG4 | 32.95 % (58) | count from the sheet; folder has no files |
| | | | SM03 | 33.52 % (59) | ditto |
| | | | SM12 | 34.66 % (61) | ditto |

Everything retained sits at or below 17.6 %; every bad-channel exclusion except EG5 sits at or above
20.5 %. **EG5 is the exception** — identical to MCI01 at 17.61 %, and its recorded reason was a
combined judgement ("not enough N2 + a lot of bad channels"; it has exactly 3 clean bouts, right on
the floor).

**DECISION: `> 20 % of electrodes (> 35 of 176)`.** Applied in Methods 3.1 and in Table 1's row.
EG5, at 17.61 %, is the one exclusion this wording does not describe; its recorded reason was a
combined judgement and it sits exactly on the three-bout floor as well.

### A2. Bad epochs — there is no boundary, and it runs backwards

| retained | % of N2 time rejected | | excluded for "too many bad epochs" | % of N2 time rejected |
|---|---|---|---|---|
| el3007 | **25.80 %** | | SM004 (MCI) — "nothing to work with, all bad epochs" | 13.91 % |
| SL44 | 16.60 % | | HR72 (HE) — "recording, anxiety, I stopped midway" | 4.62 % |
| CH66 | 15.71 % | | EL3042 (YA) — "tons of sweat" | 4.00 % † |
| IS74 | 12.35 % | | YS2 (MCI) — "all is thick" | no annotation file at all |
| YF58 | 11.78 % | | | |

† EL3042's folder holds only **raw** annotations — artifact marking was never completed on it.

**Every subject excluded for bad epochs has a lower rejected fraction than the worst subject we
kept.** The reason is mechanical, not statistical: these recordings were written off *before* the
artifact marking was finished, so their bad-epoch fraction under-represents how bad they were. A
"% of sleep time" threshold cannot be reconstructed from this, and any number we quote would be
contradicted by el3007 sitting in the cohort at 25.8 %.

**DECISION: `> 30 % of N2 time`.** Applied in Methods 3.1 and in Table 1's row. The threshold sits
above every retained subject (el3007, the worst, is at 25.80 %), so it is consistent with the cohort
as it stands; the four bad-epoch exclusions would be expected to clear it once their artifact
marking is completed, which is why their current figures are low.

**Consequence to be aware of:** as of today the four excluded recordings sit *below* the stated
threshold, because their cleaning was abandoned once they were written off. If anyone recomputes
those percentages from the folders before the marking is finished, the numbers will not support the
criterion. Finishing the marking on SM004, HR72, EL3042 and YS2 would close that gap.

### A3. What this means for Table 1's rows

Yuval's tracked rewrite of the row labels, and what each can honestly become:

| his label | can it be filled? | proposed |
|---|---|---|
| `Clean N2 bouts < X% of data` | no — the criterion is a **count**, not a percentage | `Clean N2 bouts < 3` ✔ applied |
| `Bad channels > Y% of electrodes` | yes | `Bad channels > 20% of electrodes` ✔ applied |
| `Bad epochs > Z% of sleep time` | yes, at 30 % (per A2) | `Bad epochs > 30% of N2 time` ✔ applied |
| `Total sleep time < 210 min` | already numeric | his wording verbatim ✔ applied |

Also applied: the `MoCA (where available)` header → `MoCA`, and the Young MoCA cell `—` → `N/A`.

---

## §B — DECISION 2: the 210 min citation (C371)

You asked me to find the paper we used as the literature baseline. **Four searches turned up
nothing citable.** What exists:

- Insomnia phase-3 trial protocols do use a 210 min TST screening threshold, but as an *eligibility*
  criterion for the disorder, not as a data-adequacy floor for spectral analysis — and it runs in
  the opposite direction from ours. Citing it would invite a correction.
- The structural anchor is solid and textbook: adult sleep runs in NREM–REM cycles of ~90–100 min
  (first cycle 70–100 min), so 210 min guarantees more than two complete cycles.
- Our own working note (`thesis/low_bout_and_excluded_n2_table.md:3–9`) calls it "a commonly cited
  PSG/spectral-study cutoff" but cites nothing.

**DECISION: keep the 210 min criterion, justified structurally, with no citation.** Applied in
Methods 3.1.

You asked whether the criterion could instead be dropped and everything collapsed under
"clean bouts < 3", so the exclusion reads as insufficient data for our pipeline. **It cannot, without
changing the cohort.** SM07 has **4 clean bouts (32.1 min analysed)** — above the three-bout floor —
so dropping the TST rule puts it back in, taking MCI from 30 to 31 and forcing a full re-run of
every V10/V11 statistic and figure. A duration-based variant does not rescue it either: 32.1 min
sits inside the retained range (retained minimum 22.8 min; six retained subjects are below SM07).
The 210 min rule is the only criterion that excludes SM07, so it is doing independent work.

**The empirical half of the answer is strong anyway** — worth keeping for the reply letter to Yuval
rather than the Methods text — and it answers his actual question
("why is this a problem if a big portion is clean N2"):

| removed on TST | TST | clean N2 bouts (pipeline definition) |
|---|---|---|
| SM0017 | 41.0 min | 1 |
| MCI13 | 96.5 min | 2 |
| HG78 | 162.5 min | 2 |
| SM07 | 206.1 min | 4 ← the only one the rule removed on its own |

Three of the four had fewer than three clean bouts anyway and were excluded regardless. Only SM07,
at 206 min, was removed by the TST rule alone, and it was removed to apply the rule uniformly. The
lowest retained TST is 216.5 min.

---

## §C — Proposed text, section by section

### C1 · 3.1 Participants, paragraph 1 — "identical setups"

*Location: the opening sentence of 3.1, immediately under the heading.*
*Answers: his tracked insertion, which you confirmed.*

**BEFORE**
> Data were pooled from two high-density EEG sleep studies: one at the Tel Aviv Sourasky Medical Center (TASMC), Tel Aviv, Israel, and one at the CIRUS Centre for Sleep and Chronobiology, Woolcock Institute of Medical Research, Macquarie University, Sydney, Australia. Three groups were analyzed: young healthy controls (n = 35; …

**AFTER**
> Data were pooled from two centers, both employing identical high-density EEG setups: one at the Tel Aviv Sourasky Medical Center (TASMC), Tel Aviv, Israel, and one at the CIRUS Centre for Sleep and Chronobiology, Woolcock Institute of Medical Research, Macquarie University, Sydney, Australia. Three groups were analyzed: young healthy controls (n = 35; …

*(His literal insertion reads "two centers both employing identical setups of high-density EEG sleep
studies", which is hard to parse. Same claim, smoothed. Say the word if you want it verbatim.)*
The rest of the paragraph — the three groups with n, age and sex — is untouched.

### C2 · 3.1 — split into healthy controls / MCI patients (C344)

*Location: the second paragraph of 3.1, the one beginning "The Tel Aviv participants were drawn from
the same cohort…". It currently mixes both groups, the MoCA numbers and the exclusion list into one
block; Yuval's comment sits on it.*
*Answers: C344, and his apnea note.*

**BEFORE** (one paragraph)
> The Tel Aviv participants were drawn from the same cohort detailed in our previous work¹⁹; there, MCI patients had been referred for subjective cognitive complaints to general practitioners, neurologists, geriatricians, and psychiatrists at the TASMC clinic. Montreal Cognitive Assessment (MoCA) scores were obtained during their stay in the sleep laboratory and were available for all of the Tel Aviv participants in the two older groups, but not for the Sydney participants (older adults 27.2 ± 2.6, n = 30; MCI 21.5 ± 4.3, n = 14). Beyond the diagnostic group criteria, recordings were excluded on technical grounds: too many bad channels, too many artifactual epochs, too few clean N2 bouts (fewer than three; see Section 3.3), or a total sleep time under 210 min. The per-group exclusion counts are reported in Table 1, which also summarizes the group demographics.

**AFTER** (four paragraphs)

> **[healthy controls]** Healthy participants were recruited from the community. The healthy older adults recorded in Tel Aviv belong to the cohort described in our previous work¹⁹; community volunteers whose Montreal Cognitive Assessment (MoCA) score fell below 26 were not analyzed. The young healthy controls were recorded in Tel Aviv in a separate study, `[TO SUPPLY: recruitment route and inclusion/exclusion criteria for the young cohort]`. At the Sydney site, healthy older adults were recruited `[TO SUPPLY: recruitment route and inclusion/exclusion criteria]`. Exclusion criteria applied to all participants were any history of sleep apnea, psychiatric or neurological disorder, stroke, or head injury; current alcohol or drug misuse; trans-meridian travel in the week before the recording; and use of hypnotics.

> **[MCI patients]** Patients with MCI recorded in Tel Aviv were recruited through the Memory and Attention Disorders Center of the Neurology Department at TASMC: individuals presenting with subjective cognitive complaints were referred to the clinic's general practitioners, neurologists, geriatricians, and psychiatrists, and those diagnosed with amnestic MCI were invited to take part in the sleep study¹⁹. The diagnosis was made clinically at the Cognitive Neurology Unit. Patients who had progressed to a clinical diagnosis of Alzheimer's disease were not included in the present analysis. At the Sydney site, patients with MCI were recruited and diagnosed `[TO SUPPLY: recruitment route, diagnostic criteria, and diagnosing clinician]`.

> **[apnea]** Sleep-disordered breathing was assessed in all participants. Overnight respiratory monitoring — pulse oximetry, nasal airflow, and thoracic and abdominal effort belts — was reviewed offline by a sleep clinician certified by the American Academy of Sleep Medicine (AASM); the apnea–hypopnea index (AHI) was derived where the recording permitted, and a binary clinical judgment was made otherwise. Participants with moderate or severe obstructive sleep apnea (AHI ≥ 15), central sleep apnea, or hypoventilation were excluded, so that every participant analyzed here had an AHI of 15 or below¹⁹.

> **[MoCA + technical exclusions]** MoCA scores were obtained during the participants' stay in the sleep laboratory and were available for all of the Tel Aviv participants in the two older groups, but not for the Sydney participants (older adults 27.2 ± 2.6, n = 30; MCI 21.5 ± 4.3, n = 14). Beyond the diagnostic group criteria, recordings were excluded on technical grounds: ‹§A1 wording›, ‹§A2 wording›, fewer than three clean N2 bouts (see Section 3.3), or a total sleep time below 210 min. ‹§B justification›. The per-group exclusion counts are reported in Table 1, which also summarizes the group demographics.

**Notes on what I did *not* write:**
- **No named diagnostic criterion.** Sharon et al. (2025) cites none — not Petersen, not NIA-AA. You
  said to write it the way that paper writes it, so the text gives the referral route and the
  diagnosing unit and stops there.
- **"amnestic MCI" appears only for the Tel Aviv patients**, which is what Sharon 2025 supports. The
  Sydney half has no recorded subtype, so no cohort-wide "aMCI" relabelling is done here.
- **The AHI wording.** Sharon 2025 states the rule twice with different signs — "AHI > 15" in the
  participants section, "AHI ≥ 15" in the clinical-assessment section. I used **≥ 15**, which matches
  your "all is AHI ≤ 15".
- The "same cohort detailed in our previous work" claim is now **narrowed to the older adults and the
  MCI patients**, because Sharon 2025 has no young participants (its controls are 52–85). As written
  today the sentence covers all Tel Aviv participants, which is wrong.

### C3 · 3.2 — recording setup and scoring

*Location: the first paragraph of 3.2, directly under the heading.*
*Answers: his "identical setups" insertion and his inline note "#what about Sydney scoring, same? #".*

**BEFORE**
> High-density polysomnography (PSG) was recorded throughout the night using a 256-channel recording system (Electrical Geodesics, Inc. [EGI]). Sleep was scored in 30 s epochs according to American Academy of Sleep Medicine (AASM) guidelines, using the sleep module of the Visbrain Python package²⁰ within the SleepEEGpy platform²¹. The ISFS analysis was restricted to N2 sleep.

**AFTER**
> High-density polysomnography (PSG) was recorded throughout the night at both sites using the same setup: a 256-channel system (Electrical Geodesics, Inc. [EGI]) referenced to Cz, amplified with a NetAmps 300 amplifier and digitized at 1000 Hz, with electrode impedances kept below 50 kΩ. Sleep was scored identically at both sites, in 30 s epochs and according to American Academy of Sleep Medicine (AASM) guidelines, using the sleep module of the Visbrain Python package²⁰ within the SleepEEGpy platform²¹. Scoring was performed on frontal (F3/F4), central (C3/C4), and occipital (O1/O2) derivations referenced to the contralateral mastoid, together with the electrooculogram and the submental electromyogram, and was verified against the spectrogram of the Pz electrode with the hypnogram overlaid. The ISFS analysis was restricted to N2 sleep.

Hardware and scoring montage both come from Sharon et al. (2025); extending them to Sydney and to
the young cohort rests on your confirmation that the setups and the scoring were identical.

### C4 · 3.2 — face/neck/ear electrodes (his stylistic edit)

**BEFORE**
> Electrodes overlying the face, neck, and ears, which do not sit over cortex and are prone to muscle and movement artifact, were excluded from analysis, leaving 176 channels for the analysis pipeline.

**AFTER**
> Electrodes overlying the face, neck, and ears, which are not positioned over the scalp and are more prone to muscle and movement artifact, were excluded from analysis, leaving 176 channels for the analysis pipeline.

### C5 · 3.2 — average reference (C395)

*Location: the last sentence of the "Bad channels and artifactual epochs were rejected…" paragraph,
where his C395 comment is anchored.*
*Answers: C395. **No pipeline change and no re-run** — he is mistaken, and MNE's
`set_eeg_reference('average')` excludes `info['bads']` from the average (verified empirically on
MNE 1.6.1). This clause simply says so.*

**BEFORE**
> Data were then re-referenced to the common average reference, and channels rejected as bad were interpolated from neighboring channels.

**AFTER**
> Data were then re-referenced to the common average reference. The average was computed over the good channels only, excluding channels marked as bad, and those channels were subsequently interpolated from their neighbors.

### C6 · 3.2 — delete the Figure 1 overview paragraph (C412)

*Location: the standalone one-sentence paragraph between the scoring paragraph and the Figure 1
image.*
*Answers: C412 and his inline note "you are mixing Methods and Results".*

**BEFORE**
> An overview of the recorded sleep across the three groups, including whole-night hypnospectrograms, the distribution of sleep stages, and N2 bout properties, is shown in Figure 1.

**AFTER** — deleted entirely.

The Figure 1 image and its caption **stay where they are**; moving them to Results is a separate
session. Worth carrying over when that happens: Yuval also *rewrote* this sentence, and his version
is better —

> …including whole-night hypnograms (time-course of sleep stage dynamics) superimposed with EEG spectrograms (time-frequency dynamics), the distribution of sleep stages, and N2 bout properties…

### C7 · Table 1 — exclusion row labels

Table 1 is a native table object in the Doc, so these are direct cell edits.

| row | before | after |
|---|---|---|
| 3 | `MoCA (where available)` | `MoCA` *(his edit)* |
| 3, Young cell | `—` | `N/A` *(his edit)* |
| 6 | `Not enough clean bouts` | `Clean N2 bouts < 3` |
| 7 | `Too many bad channels` | ‹§A1 decision› |
| 8 | `Too many bad epochs` | ‹§A2 decision› |
| 9 | `TST < 210 min` | `Total sleep time < 210 min` *(his edit)* |

Counts in the table are unchanged (10 / 4 / 12, total 26).

---

## §D — His tracked Methods edits: applied, superseded, or flagged

Extracted from `thesis/reviews/Shaked's Thesis_YN.docx` by walking `w:ins` / `w:del` — 16 changed
paragraphs inside Methods.

**Applied** (above): "identical setups" → C1; face/neck/ear rewording → C4; the four Table 1 row
labels and the MoCA cells → C7.

**Superseded — V2 already does it, no action:**

| his edit | V2 already reads |
|---|---|
| "256-channel Electrical Geodesics, Inc. (EGI) system" in 3.4 | EGI is introduced at first mention in 3.2 |
| "Fast Fourier Transform (FFT)" spelled out in 3.4 | "via the fast Fourier transform (FFT)" |
| "for each clean **N2** bout" in 3.4 | the passage was rewritten; it reads "For each clean bout in each channel" — I can add "N2" if you want it |

**Flagged — his, but outside this session's eight items. Not done, not forgotten:**

- **3.2**: he wants a citation on "The ISFS analysis was restricted to N2 sleep **following previous
  work #REF**" — one of the three `#REF` placeholders in the triage.
- **3.3**: "#is this following zurich procedures? If yes then mention and cite#" on the 300 s bout
  rule.

Both are citation requests rather than Methods prose, and they belong with the reference pass.

**Checked for silent deletions:** he removed at least one passage elsewhere with Track Changes off,
so I compared every unchanged Methods paragraph in his copy against V2. Nothing is missing from
Methods — the only differences are the Flavio-era rewrites already in V2.

---

## §E — `[TO SUPPLY]` register

Everything left blank, for the later fill-in session:

1. ~~**Young controls** — which TASMC study, recruitment route, inclusion/exclusion criteria~~
   **CLOSED 2026-08-17, see §F.** The AHI sub-question is closed too (§F2: the young cohort was not
   screened, and the paragraph now says so). Still open: the separate TASMC young-adult study is not
   named or cited.
2. **Sydney healthy older adults** — recruitment route, inclusion/exclusion criteria.
3. **Sydney MCI patients** — recruitment route, diagnostic criteria, who made the diagnosis.

Not marked `[TO SUPPLY]` but still open, listed so nothing is lost:

4. The two `#REF` citations in §D.

*(A named MCI diagnostic criterion was considered and dropped: Sharon 2025 states none, so neither
do we.)*

---

## §F — young-cohort recruitment fill-in (2026-08-17) — **APPLIED to the Doc**

Gap 1 of the §E register, supplied by the user: the young adults were recruited at TASMC, and
healthy participants reported no history of neurological, psychiatric, or sleep disorders.
One sentence changed, in §3.1 paragraph 2. Applied to the Doc with a single `findAndReplace`
(1 occurrence) and mirrored to `thesis/chapters/03_methods.md`.

**BEFORE**

> The young healthy controls were recorded in Tel Aviv in a separate study, [TO SUPPLY: recruitment
> route and inclusion/exclusion criteria for the young cohort].

**AFTER**

> The young healthy controls were recruited at TASMC in a separate study and reported no history of
> neurological, psychiatric, or sleep disorders.

Notes:

- The health-history clause deliberately overlaps the exclusion sentence two lines later ("any
  history of sleep apnea, psychiatric or neurological disorder…"). It was kept because it adds what
  that sentence does not say — that for the young cohort the screen was **by self-report** — and
  makes the young cohort's criteria explicit rather than leaving them to be inferred from a list
  introduced as applying "to all participants". The alternative considered and not taken was the
  recruitment half alone.
- Verified after the edit: `findElement "TO SUPPLY"` returns **2** hits (was 3), both Sydney; the new
  sentence returns exactly 1 hit; single spaces at both joins, no formatting drift.

### F2 — apnea screen scoped to the older groups (same day, also APPLIED)

The user's follow-up: apnea is relevant to, and was screened in, the older groups only — it should
not be attached to the young cohort. Two sentences in §3.1 paragraph 3, wording chosen by the user:

| | before | after |
|---|---|---|
| opening | Sleep-disordered breathing was assessed in **all participants**. | Sleep-disordered breathing was assessed in **the two older groups**. |
| closing | …so that **every participant** analyzed here had an AHI of 15 or below | …so that **every older participant** analyzed here had an AHI of 15 or below |

The middle sentence (respiratory montage, AASM-certified scorer, AHI derivation) is unchanged and now
reads as describing the older-group screen only. This closes the AHI sub-question left open in §E
item 1, and answers Yuval's apnea comment more precisely than the previous wording did.

**Still unverified:** whether the **Sydney** older adults and aMCI patients were screened the same
way. "The two older groups" asserts they were; the cited source (Sharon 2025) covers TASMC only. Was
raised with the user and not answered — if Sydney did not screen, the sentence should become "the
older participants recorded in Tel Aviv".

Paragraph 2's exclusion sentence still lists "any history of sleep apnea" among criteria "applied to
all participants". Deliberately left: that is self-reported history, not the overnight screen, and it
stays true for the young cohort under the sentence added in §F above.
