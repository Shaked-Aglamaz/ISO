# Rewrite §4.3 of the Discussion mechanism paragraph

## Context

`thesis/reviews/discussion_edits_before_after.md` §4.3 (lines 160–201) holds "Option A" of the
mechanism paragraph answering Yuval's C558. The user is unhappy with both the current draft and their
own alternative. Two rounds of line-level objections, all now folded into the version below:

Round 1 — no "Consequently"; "where the driving signal exerts its strongest influence" unclear;
flattening sentence must run reason → outcome; no colon-plus-explanation constructions; the
"source vs. cortical targets" framing is unsupported; "offline window" is undefined; the
"mechanistic explanation for fragmented sleep" is an overclaim (fragmentation is not set up until
§6.3, which comes later).

Round 2 — drop the empty opener ("These mechanisms have a specific consequence…"); always pair
"strength" with "AUC"; not "swings", just say the hotspot, and cite the fast-spindle topography;
comma-join "What changed was the difference between regions" into its predecessor; "A weaker or less
well-timed driving signal" is too long and unclear; do not re-describe the pattern after stating it —
end on the mechanism; say "peak frequency", never "speed"; drop the second paragraph's opener too;
and the fragile-phase sentence is wrong, since a shorter cycle makes *both* substates recur more often.

Facts verified this session:
- Whole-scalp AUC null: F = 0.82, p = 0.45 (`04_results.md`; `results_edits_before_after.md:186`).
- Central-parietal AUC cluster lost in both older groups (`04_results.md` §4.3).
- Peak frequency: whole-scalp effect, **no significant cluster at any location** (`04_results.md:43`,
  Figure S1) — this is what replaces the "source vs. targets" framing.
- "Fast spindles are maximal over centroparietal cortex [@molle2011fastslow]" is already stated in
  Introduction 2.2 (`02_introduction.md:25`). Re-citing it in the Discussion adds **no new
  reference** and costs no renumbering.
- "Alternating fragile and offline substates" is defined in Introduction 2.2
  (`02_introduction.md:27`) — reuse that exact wording rather than coining "offline window".

## The full replacement text for §4.3

Paragraphs 1 and 2 are replaced. Paragraph 3 (the limits paragraph) is unchanged.

> The ISFS is a modulation of fast-spindle amplitude, so its strength (AUC) at an electrode reflects
> how deeply spindle amplitude rises and falls at that site, not how many spindles occur there. In
> young adults the AUC hotspot was central-parietal, over the region where fast spindles are largest⁽ᵐᵒˡˡᵉ⁾.
> In both older groups that hotspot was gone, while average AUC across the scalp was unchanged, so what
> was lost was regional focus rather than overall strength. That pattern points to a drive that has
> grown less precise rather than weaker.

> The peak-frequency effect appeared across the whole scalp, with no region standing out. A change
> present everywhere is more easily placed in whatever paces the cycle than in the cortex where the
> spindles appear, and in rodents the pacing comes from the LC. On that reading the noradrenergic cycle
> itself runs faster in older adults: fragile and offline substates alternate more often, and the
> noradrenergic peak each cycle carries arrives more often with them.

> This chain is not shown in these data. Neither noradrenergic nor thalamic activity was measured here,
> so the link from LC decline to a flattened scalp hotspot rests on rodent, neuropathological and
> imaging work. What these data can check is the prediction it makes. A mechanism anchored in a
> structure that degenerates early, and in the general population rather than only in patients, predicts
> a change that is complete in healthy older adults and no further advanced in aMCI, which is the
> pattern found here.

`⁽ᵐᵒˡˡᵉ⁾` = the superscript `molle2011fastslow` already carries in the Doc (Introduction 2.2). Existing
reference, re-cited; nothing is added to the list.

### How each objection is discharged

| Objection | Where it went |
|---|---|
| "Consequently" / "Meanwhile" | gone; no connective needed |
| empty opener, ¶1 | deleted; the paragraph opens on the definition |
| empty opener, ¶2 | deleted; opens on the finding |
| "strength" unpaired | "its strength (AUC)", then AUC throughout |
| "swings" / hotspot | "the AUC hotspot was central-parietal" |
| fast-spindle claim uncited | now cited, `molle2011fastslow`, already in the Intro |
| outcome-before-reason | one sentence: hotspot gone + average unchanged → conclusion |
| stranded "What changed was…" | comma-joined into that same sentence |
| "weaker or less well-timed driving signal" | "a drive that has grown less precise, rather than weaker" |
| pattern re-described after the conclusion | the two restating sentences are deleted outright |
| "speed" | "peak frequency" |
| source vs. cortical targets | whole-scalp-with-no-cluster vs. regional, straight from Results 4.3 |
| "offline window" | the Introduction's "fragile and offline substates" |
| fragile phase singled out | both substates recur; the NE point rests on one peak per cycle |
| fragmented sleep | deleted entirely |

Round 3 — ¶1 closer reversed to pattern → conclusion; ¶2 rebuilt so it states its point (the pacing of
the cycle, not the cortex, is what changed, and in rodents that pacing is the LC); ¶3's "Its one
testable prediction, though, is met" replaced by naming the prediction before stating it, and
"That is what we found" folded into the preceding sentence as "which is the pattern found here".
¶3's opener "Two things this argument does not establish" also goes — it announced two items and
delivered one.

### One call left to the user

- ¶2 stops at the noradrenergic peaks. If an arousal consequence is wanted, one sentence can be
  appended: "In rodents those peaks coincide with micro-arousals, so the windows in which sleep is most
  easily interrupted come round more often too." Defensible, but it reopens the topic already cut.

## Change to apply (only after approval of the text above)

Single file: `thesis/reviews/discussion_edits_before_after.md`. Nothing touches the Google Doc or
`thesis/chapters/05_discussion.md` — edits are staged in the review .md and applied only after approval.

1. Replace the quoted §4.3 block (lines ~164–183) with the three paragraphs above, with `⁽ᵐᵒˡˡᵉ⁾`
   rendered as the Doc's actual superscript for `molle2011fastslow`.
2. Rewrite the `▸ Plain version` box (~185–197) to the new six beats: AUC = depth of modulation, not
   spindle count → hotspot is central-parietal where fast spindles are largest → hotspot gone but
   scalp average unchanged, so focus was lost, not strength → a less precise drive fits → peak
   frequency changed everywhere, so it is a timing statement → shorter cycle, NE peaks closer
   together. Delete the fragmented-sleep beat.
3. Rewrite the `*Changed this round:*` note to record the two rounds of objections and their fixes.
4. Add a line noting that `molle2011fastslow` is now cited in the Discussion as well — no new
   reference, so §9's renumbering plan is untouched.

## Knock-on checks

- §0.1 row "§4.3 too hard to follow" and §0.2 decision #1 (C558): refresh so they describe this
  revision, not the previous one.
- §10.1 row H5 maps "pacing signal"/"pacemaker" → "driving signal"/"source of the modulation". The new
  text keeps "drive" but no longer contains "source of the modulation"; trim that half of H5.

## Verification

- `grep -n "Consequently\|offline window\|fragmented sleep\|speed" thesis/reviews/discussion_edits_before_after.md`
  returns nothing inside §4.3.
- Every factual clause maps to a verified number: whole-scalp AUC null (p = 0.45); lost central-parietal
  cluster; peak-frequency effect with no significant cluster; fast-spindle topography cited to an
  existing reference.
- Read §4.1 → §4.2 → §4.3 in sequence and confirm ¶1 still reads as a continuation now that its linking
  opener is gone.
