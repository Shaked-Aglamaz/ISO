---
name: reference_sun2026_mecfs_not_miscitation
description: "Ref sun2026longcovid looks mis-cited in Discussion 5.4 (title says long COVID, our text says ME/CFS) but is CORRECT — the paper has a separate ME/CFS arm; do not re-flag"
metadata:
  node_type: memory
  type: reference
---

**`sun2026longcovid` is cited correctly in Discussion §5.4. It only looks wrong. Do not "fix" it.**

The sentence reads *"ISO power in the slow-sigma band is elevated in myalgic encephalomyelitis and
chronic fatigue syndrome"* and cites a paper **titled** *"Facility-measured sleep electroencephalographic
microstructures in **long COVID**"* (Sun et al., SLEEP 2026;49(8):zsag090). The title/claim mismatch
reads like a mis-citation on sight — I flagged it CRITICAL in the 2026-08-19 pre-send pass before
checking, and it was a false alarm.

**Why it is right:** the study has three arms — long COVID (28), **ME/CFS (19)**, controls (28) — and the
elevated slow-sigma ISO power is specifically the **ME/CFS** result. Recorded at
`thesis/references/library_status.md:152` and `thesis/references/new_refs_annotated.md:235,253`.

**The real caveat is already disclosed in the prose:** it is *slow* sigma (11–13 Hz, the frontal
slow-spindle band), not our fast-spindle 13–16 Hz. §5.4 says "slow-sigma band" and the next paragraph
says "The sigma bands differ as well", so this is handled — do not add a caveat that is already there.

**General lesson:** before flagging a citation from the title alone, read the entry in
`library_status.md` / `new_refs_annotated.md` — every reference was vetted there with its arms and
measures written out. See also [[reference_isfs_frequency_attribution]] for a case that *was* a genuine
mis-citation, and [[project_isfs_definition]] on ISFS vs ISO naming.

Related: [[reference_dimitriades_citation_status]], [[project_bibliography_expansion]],
[[project_discussion_revision_yuval]]
