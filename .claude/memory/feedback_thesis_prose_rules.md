---
name: feedback_thesis_prose_rules
description: "Thesis-prose rules: only 'the ROI' (never 'extended'), highest-V figures, N1/N2/N3 naming, abbreviate-after-first-use, no em dashes — plus the sentence-level style he enforces line by line (no throat-clearing openers, reason→outcome, name the parameter, no unearned scope)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d7e0bf61-82d0-4afc-bb1a-ad690a607b82
  modified: 2026-08-16T21:24:44.050Z
---

Two rules for writing thesis prose:

**1. To the reader there is only "the ROI" — never write the word "extended", and never compare a Core ROI (clarified 2026-05-31).**

**Why:** The Core-vs-Extended trade-off (documented in [[project_roi_choice]]) was an internal design decision. We use the larger (36-channel) version, but the reader should never see the word "extended" or learn that a smaller variant existed. To them it is just "the ROI": a pre-defined central-parietal electrode set taken from the Dimitriades (2024) young-adult AUC hotspot, mapped from 128 to 256 channels.

**How to apply:** In Methods/Results/Discussion and all figure captions, call it "the ROI" or "the pre-defined central-parietal ROI". Do NOT write "extended ROI", "core ROI", "we also tried a smaller variant", or any sensitivity-analysis framing. Result filenames still contain `extended` (e.g. `..._extended_ROI_normalized_auc.png`) but that label is internal only and must not surface in prose. Only mention the Core variant if the user or a reviewer asks directly.

**2. Thesis figures: always select from the highest V-numbered subfolder.**

**Why:** Multiple V-numbered analysis output folders exist under `results/` (e.g. `three_groups_V2`, `three_groups_V5`). The latest version is canonical. This complements [[feedback_versioned_output_dirs]] which governs *saving* to V-dirs; this rule governs *consuming* them for thesis figures.

**How to apply:** Before adding any figure to `thesis/figure_manifest.md` or `thesis/figures/`, scan the parent dir for V-numbered siblings and pick the highest. If two analyses appear redundant across V-numbers, flag it for user review rather than silently choosing.

**3. Sleep-stage naming = N1/N2/N3 + REM (decided 2026-06-13).** Use the short AASM stage labels (N1/N2/N3) throughout prose and captions; reserve "NREM" for the general non-REM category only. Spell out in full at first use (Intro opening: "Non-rapid eye movement (NREM) sleep … its second stage (N2)"). The whole paper was swept `NREM2→N2` (0 NREM2 remain). When the Abstract is drafted, it precedes the Intro, so its first use of N2 must carry the full term too. Figures/Table 1 already use N2/N3/REM.

**4. IMRAD Methods/Results split — no duplicated numbers (decided 2026-06-13).** Methods states *procedures and decision criteria*; Results states *outcomes (counts, rates, the actual findings)*. A given number lives in exactly ONE place. Applied: bout counts, ISFS detection rates, and the "N2 abundant / not a confound" claim were stripped from Methods and kept only in Results; Methods retains the extraction procedure, the ≥20%-channel inclusion criterion, and "analysis restricted to N2". Apply this when drafting Discussion/Abstract too.

**5. Abbreviations: define once, then always abbreviate (instruction 2026-08-16).** The user's words:
*"you used 'amnestic MCI' and 'Alzheimer's disease' a lot of times instead of aMCI and AD — define
those once and then use the shortened term."* Define at the term's genuine **first use in the body**,
then never spell it out again. **Downstream consequence to handle in the same pass:** a definition
added in an earlier chapter makes the later one a duplicate, so strip it — the 2026-08-16 Intro pass
forced Methods 3.1 to drop `(aMCI; …)` and Methods 3.2 to drop `polysomnography (PSG)`. The Abstract
is exempt and keeps its own definitions, because abstracts are standalone.

**6. No em dashes — the whole document is now at zero (completed 2026-08-19).** Use colons, commas or
a sentence break instead. The Methods exception is gone: the last two pairs were removed pre-send, one
to parentheses (*"Overnight respiratory monitoring (pulse oximetry, nasal airflow, and thoracic and
abdominal effort belts) was reviewed offline…"*) and one by splitting the sentence, which also killed a
doubled "so that" in §3.1. **The single remaining `—` is inside Ju et al.'s published reference title
(*"Sleep and Alzheimer disease pathology—a bidirectional relationship"*) and must stay** — never
"correct" a `—` that sits in a citation title. **En dashes are a different character and are all
correct** (80 of them: `13–16 Hz`, `NREM–REM`, `Gabor–Morlet`, `Shapiro–Wilk`, `apnea–hypopnea`), so
grep for U+2014 specifically and do not touch U+2013. He asks for this check by name ("make sure we
dont have em dashed"), so audit new prose for `—` before applying anything.

**7. Cite where the argument needs it; never pad to a reference count.** Instruction 2026-08-16:
*"add where they are needed, don't overflow — we don't have a problem of amount."* Yuval's ≥50-refs
demand (email #9) is met by the Intro and Discussion expansions naturally; do not force extra
citations into a section to reach a number.

**8. Section length is set by content, not a word target.** Instruction 2026-08-16: *"the word count
is not that important, mainly by the amount of text which makes sense for each one of the topics he
requested."* Cover every requested topic properly and stop; do not pad to a page count. For
calibration against what this lab expects, see [[reference_yael_gat_thesis]].

**9. Sentence-level style, enforced line by line (2026-08-16, three rounds on Discussion §4.3).** The
recurring complaints, in his words: *"i want sentences with simple structure, 1 → 2 → 3 → 4, not all
these complex structures."*

**Why:** he reads every sentence and rejects anything that sounds like it was generated rather than
argued. Three revisions of one paragraph were spent on this, so applying it up front saves rounds.

**How to apply:**
- **No throat-clearing openers.** "These mechanisms have a specific consequence for what was measured
  here", "The change in speed is a separate finding" — a paragraph opens on its first real claim.
- **Reason → outcome, then pattern → conclusion.** Evidence first, inference second, both within and
  across sentences. He flags the reverse order on sight.
- **State a pattern once.** Do not restate it after drawing the conclusion; end on the conclusion.
- **No colon-plus-explanation** ("produces precisely this effect: …") and no "Consequently",
  "Meanwhile", "precisely this".
- **Name the parameter, always.** "peak frequency", never "speed"/"acceleration"; "strength" is always
  paired with "(AUC)".
- **No coinages.** If a concept was defined earlier, reuse those exact words ("fragile and offline
  substates"), never a paraphrase like "offline window".
- **Every mechanistic-sounding phrase must be cashable.** "where the driving signal exerts its
  strongest influence" and "the source rather than its cortical targets" were both cut as unsupported;
  replacements had to come from an actual result (whole-scalp effect with no significant cluster) or an
  existing citation.
- **No unearned scope.** A claim about something the thesis never set up (sleep fragmentation) is cut,
  not softened — even when the data would support it, if the reader has not met it yet.
- **Check claims for a citation before he does.** He asks "can we support this — verify me"; find the
  existing reference (it is usually already in the Intro) rather than dropping the claim.

**10. Never point at another section by number, and never argue with the reviewer inside the text
(2026-08-17, Discussion pass).** Three separate complaints, all the same instinct: the prose should read
as a thesis, not as a reply.

- **No "Section 2.3" / "as set out in Section 2.x" in prose.** His words: *"i didnt like the reference
  'Section 2.3 noted that...' can we refer in words to the location (or just say 'as mentioned before')"*.
  Use "as noted earlier", or nothing at all — a citation usually carries the claim on its own. Rejected
  in four places in one pass.
- **No self-justification.** Cut anything that explains *why the thesis is written this way*: "which is
  the reason to begin an account of these data with noradrenaline rather than with acetylcholine", "It
  is worth stating plainly that", "These studies also answer, indirectly, whether the field measures the
  same thing in the same way". His words: *"no need to justify ourselves inside the thesis text, we
  shouldn't treat yuvals comments inside the text in a defensive way"*. State the fact; the answer to
  the comment is that the fact is now there.
- **Never open a paragraph with a question.** "What would that mean concretely for…?", "Where does this
  leave the ISFS among…?" — both cut on sight. Related to the throat-clearing rule in §9, but he flags
  the question form specifically.

**Why:** a reviewer reads the finished chapter, not the revision history. Anything that betrays the
revision — a cross-reference standing in for an argument, a defence of an editorial choice, a rhetorical
question setting up an answer — reads as scaffolding and gets cut.

**11. When trimming a claim, check the same paragraph for the claim it now contradicts (2026-08-17).**
Narrowing one sentence ("peak amplitude has no direct counterpart in our data") left a sentence four
earlier still asserting the opposite ("Their result **parallels** our central-parietal cluster"). Nobody
would have caught it by reading the diff. After any softening, re-read the whole paragraph.

Related: [[project_thesis]], [[project_roi_choice]], [[feedback_versioned_output_dirs]],
[[project_scientific_story]], [[project_intro_revision_yuval]], [[reference_yael_gat_thesis]],
[[feedback_doc_edits_review_first]]
