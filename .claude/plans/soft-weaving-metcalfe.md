# Final pre-send pass — Shaked's Thesis V2 — approved edit set

## Context

Yuval's review is closed; this is the last pass before sending, inside a one-hour ceiling.
I read the full Doc (237 paragraphs, 80,021 chars) and swept memory + `thesis/reviews/` for
carried-over items. Your decisions on my findings are folded in below: **#1 stays as-is**
(placeholders, Yuval is aware), **#2 is dropped** (Dimitriades keeps its preprint citation),
and **#4 turned out to be a false alarm** — see below. Everything else is an edit.

Seven edits total, all in the Doc, all string-exact.

---

## Answers to your two questions

**#4 — "so what is the fix???" → There is no fix. The citation is correct.**

Ref 56 is titled *"…microstructures in long COVID"*, which is why it looked wrong next to a
sentence about ME/CFS. But the study has three arms — long COVID (28), **ME/CFS (19)**,
controls (28) — and the elevated slow-sigma ISO power is specifically the **ME/CFS** result.
`thesis/references/library_status.md:152` records this: *"Reports elevated ISO power in the
slow sigma band (11–13 Hz) in ME/CFS."* The sentence already says "slow-sigma band", and §5.4
already notes "The sigma bands differ as well", so the 11–13 Hz vs our 13–16 Hz mismatch is
disclosed too. **No change. Nothing to fix.**

**#3 — confirmed a real error, and here is why.**

Source is `chapters/05_discussion.md:33`, citing `[@zhang2022alzheimerreview; @gorgoni2016parietal]`
= refs 26, 27. Ref 26 is Zhang's meta-analysis of polysomnography in **established AD** — it
proposes no early marker of anything. The "candidate early markers" half of the claim is
carried by refs 36–37 (Taillard, Liu), which is exactly how the identical claim is cited in
§2.4 and §5.5. Ref 27 (Gorgoni) correctly supports the parietal fast-spindle half and stays.
Ref 26 remains cited in §2.3, so dropping it here orphans nothing.

---

## The seven edits

### E1 — §5.3 citation numbers (finding #3) · ¶152

**Find:** `as candidate early markers of incipient cognitive impairment²⁶⁻²⁷.`
**Replace:** `as candidate early markers of incipient cognitive impairment²⁷,³⁶⁻³⁷.`

Ascending order and the bare-comma style match house usage elsewhere (`¹⁵,¹⁶` in §4.3,
`¹¹⁻¹³,⁴⁷` in §5.2).

### E2 — remove the duplicated sentence in §5.3 (finding #5) · ¶153

The paragraph states Grollero's peak-amplitude result, then restates it five sentences later.
Only the restatement goes; the sentence after it carries new information and is kept, with
`that result` → `their result` so the referent survives the cut.

**Find:**
> They diverge on whether cognitive impairment leaves a mark beyond age. Their effect was on the amplitude of the spectral peak, which was reduced in patients with Alzheimer's disease relative to age-matched controls. Peak amplitude was not among the parameters measured here, so that result has no direct counterpart in our data;

**Replace:**
> They diverge on whether cognitive impairment leaves a mark beyond age. Peak amplitude was not among the parameters measured here, so their result has no direct counterpart in our data;

### E3 — "rather" thinning (finding #6) · ¶43, ¶78, ¶148, ¶152

18 occurrences. I am touching only the four densest spots, taking it to **12**. I am
deliberately **not** touching the Abstract's closing "…reshaped by aging … rather than by
amnestic mild cognitive impairment", because it mirrors the title.

| # | ¶ | Find | Replace |
|---|---|---|---|
| a | 43 | `contributed rather than age.` | `contributed, not age.` |
| b | 78 | `typical of adults — rather than a fragment of one,` | `typical of adults — and not a fragment of one,` |
| c | 148 | `so what was lost was regional focus rather than overall strength.` | `so what was lost was regional focus, not overall strength.` |
| d | 152 | `a marker of chronological aging rather than of cognitive status, and the data temper, rather than support, the use` | `a marker of chronological aging, not of cognitive status, and the data temper rather than support the use` |
| e | 152 | `tracks how far the brain has aged rather than whether cognition has begun to decline.` | `tracks how far the brain has aged, not whether cognition has begun to decline.` |

That takes ¶152 from three "rather" to one, and ¶148 and the Abstract from two to one each.

### E4 — remove LC-NE, English abstract (finding #7) · ¶43

**Find:** `changes in locus-coeruleus noradrenergic (LC-NE) activity`
**Replace:** `changes in locus coeruleus noradrenergic activity`

### E5 — remove LC-NE, Hebrew abstract (finding #7) · ¶45

**Find:** `של הלוקוס סרולאוס (LC-NE)`
**Replace:** `של הלוקוס סרולאוס`

### E6 — remove LC-NE, Intro §2.2 (finding #7) · ¶55

**Find:** `the locus coeruleus norepinephrine (LC-NE) system¹¹⁻¹².`
**Replace:** `the locus coeruleus noradrenergic system¹¹⁻¹².`

That is all three occurrences. `LC` stays defined where it is first actually used, in §5.2
("the locus coeruleus (LC) does not merely accompany…") — **no edit needed there**, and
nothing downstream breaks. `norepinephrine` → `noradrenergic` also unifies the Intro with the
Abstract and the whole Discussion, which use the noradrenaline family throughout.

### E7 — EEG expansion in the abstract (finding #8) · ¶43

**Find:** `including high-density (256-channel) EEG in 35 young adults`
**Replace:** `including high-density (256-channel) electroencephalography (EEG) in 35 young adults`

The Hebrew abstract already expands it (`אלקטרואנצפלוגרפיה (EEG)`), so only the English side
needs this. ("EEG" does occur once earlier, in the Acknowledgements, which by convention does
not carry the definition.)

### E8 — Table 2 caption (finding #9) · ¶116

**Find:** `a bout is a stretch of NREM2 of at least 300 s`
**Replace:** `a bout is a stretch of N2 of at least 300 s`

Single occurrence of "NREM2" in the document.

---

## Not being done, per your call

- `[TO SUPPLY: …]` ×2 in Methods §3.1 — kept, Yuval is aware.
- Dimitriades ref 16 stays as bioRxiv 2024, and the four in-text "(2024)" mentions stay.
- Everything in my LOW tier (Fig S1 `±1σ` notation, Abstract sigma-band wording, ethics
  statements, two-site confound, Sydney apnea screening) — untouched.
- No broad humanizer rewrite. E3 is the whole of it.

---

## Execution

1. Apply E1–E8 with `mcp__google-docs__findAndReplace`, one call each, longest/most-specific
   strings first. Every Find string above I have verified is unique in the document.
2. Do **not** touch anything else in the Doc.

## Verification

- `findElement "LC-NE"` → 0 hits (was 3)
- `findElement "NREM2"` → 0 hits (was 1)
- `findElement "²⁶⁻²⁷"` → 0 hits; `"²⁷,³⁶⁻³⁷"` → 1 hit
- Count of "rather" → 12 (was 18)
- `findElement "TO SUPPLY"` → still 2, unchanged and intentional
- `listComments` → still 0
- Re-read ¶43, 45, 55, 78, 116, 148, 152, 153 and confirm the reference list is still 63
  entries in unbroken first-appearance order

Rollback if anything mis-fires: each edit is a single string swap, reversible by running the
replacement backwards.
