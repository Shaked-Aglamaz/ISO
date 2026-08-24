# Fill the young-adult recruitment gap in Methods §3.1

## Context

Methods §3.1 states recruitment and diagnosis per site (added for Yuval's C344), but three gaps
were left as literal `[TO SUPPLY]` brackets in the Doc. You have now supplied gap #1: the **young
adults were recruited at TASMC**, and **healthy participants reported no history of neurological,
psychiatric, or sleep disorders**. This edit closes that one bracket. The other two (Sydney older
adults, Sydney aMCI) stay open.

The `[TO SUPPLY]` text is really in the Doc — verified at indices 21229, 21371, 22331 — so this is a
visible placeholder that Yuval can see, not just a local note.

## The edit — §3.1, paragraph 2 ("Healthy participants were recruited from the community…")

Only the third sentence changes. Full paragraph shown so you can read it in context; **bold** marks
what moves.

**BEFORE**

> Healthy participants were recruited from the community. The healthy older adults recorded in Tel
> Aviv belong to the cohort described in our previous work [Sharon 2025]; community volunteers whose
> Montreal Cognitive Assessment (MoCA) score fell below 26 were not analyzed. **The young healthy
> controls were recorded in Tel Aviv in a separate study, [TO SUPPLY: recruitment route and
> inclusion/exclusion criteria for the young cohort].** At the Sydney site, healthy older adults were
> recruited [TO SUPPLY: recruitment route and inclusion/exclusion criteria]. Exclusion criteria
> applied to all participants were any history of sleep apnea, psychiatric or neurological disorder,
> stroke, or head injury; current alcohol or drug misuse; trans-meridian travel in the week before
> the recording; and use of hypnotics.

**AFTER (recommended)**

> Healthy participants were recruited from the community. The healthy older adults recorded in Tel
> Aviv belong to the cohort described in our previous work [Sharon 2025]; community volunteers whose
> Montreal Cognitive Assessment (MoCA) score fell below 26 were not analyzed. **The young healthy
> controls were recruited at TASMC in a separate study and reported no history of neurological,
> psychiatric, or sleep disorders.** At the Sydney site, healthy older adults were recruited [TO
> SUPPLY: recruitment route and inclusion/exclusion criteria]. Exclusion criteria applied to all
> participants were any history of sleep apnea, psychiatric or neurological disorder, stroke, or head
> injury; current alcohol or drug misuse; trans-meridian travel in the week before the recording; and
> use of hypnotics.

### One thing to decide

The new clause partly restates the sentence two lines later — "any history of sleep apnea,
psychiatric or neurological disorder" is already an exclusion criterion for everyone. The version
above keeps it anyway, because it adds something that sentence does not say: for the young cohort the
screen was **by self-report**, and it makes the young cohort's criteria explicit rather than leaving
the reader to infer them from a list introduced as applying "to all participants".

If you would rather not repeat it, the alternative is the recruitment half only:

> **The young healthy controls were recruited at TASMC in a separate study.**

Tell me which; the recommended version is what I will apply unless you say otherwise.

## What stays open

`[TO SUPPLY]` #2 (Sydney healthy older adults) and #3 (Sydney aMCI patients) are untouched and remain
visible in the Doc. Also still unanswered from the §E register: whether the AHI ≤ 15 apnea screen was
applied to the young cohort — §3.1 paragraph 3 currently asserts breathing was assessed in *all*
participants, so if the young study did not screen for apnea that sentence needs a caveat.

## Execution, after you approve

1. **Doc** (`Shaked's Thesis V2`, `1YpXrDGFlzRk…`) — one `findAndReplace` of the old sentence with the
   new one. It is inside a single paragraph, so no paragraph-mark limitation applies, and the run
   carries no italic/superscript formatting, so the replacement inherits plain 12 pt body style.
2. **Mirror** `thesis/chapters/03_methods.md` line 17 with the identical sentence, and add a `v8`
   note at the top of that file recording the fill-in.
3. **Record** the before/after in `thesis/reviews/methods_edits_before_after.md`, and strike item 1
   from its §E `[TO SUPPLY]` register (leaving the AHI sub-question listed as still open).

## Verification

- `findElement` on `"TO SUPPLY"` in the Doc should return **2** hits afterwards, not 3.
- `findElement` on the new sentence should return exactly 1 hit.
- Read back the paragraph range to confirm no formatting drift and no double space at the join.
