---
name: project_isfs_definition
description: "ISFS = Infra-Slow Fluctuations of Sigma Power (NOT \"Frequency Shifts\") — and the SAME phenomenon is published elsewhere as \"infra-slow oscillations (ISO)\" of sigma/spindle power, so literature searches MUST cover both names"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e0bf61-82d0-4afc-bb1a-ad690a607b82
  modified: 2026-08-14T09:45:19.709Z
---

**ISFS = Infra-Slow Fluctuations of Sigma Power.**

It is NOT "Infra-Slow Frequency Shifts" — that misreading appeared in `CLAUDE.md` and elsewhere and should be corrected on sight.

What it actually refers to: the amplitude envelope of the sigma band (13–16 Hz) during NREM2 sleep oscillates at very low frequencies (~0.0075–0.04 Hz). ISFS is the spectral peak detected in the FFT of that envelope (the bout's sigma-power envelope), fit with a Gaussian and characterized by peak frequency, bandwidth, AUC, peak power.

## The rhythm has TWO names in the literature — search both (added 2026-08-14)

Other groups publish the identical phenomenon as the **infra-slow oscillation (ISO)** of sigma or spindle power. Same rhythm, different naming convention. `grollero2026iso`, already in `library.bib`, uses "infraslow oscillations" in its title.

**This has bitten once already.** A bibliography sweep searched only the ISFS naming and concluded that almost no ISFS work existed outside aging/MCI. Re-running on the ISO terminology found three clinical populations (schizophrenia, autism, long COVID/ME-CFS) plus a large mechanistic literature. The conclusion was a search artefact, not a fact about the field.

**Search rule:** cover `ISFS`, `ISO`, **and** both spellings `infraslow` / `infra-slow` — PubMed treats the two spellings as distinct tokens in `[tiab]` queries. Pair them with `sigma`, `spindle`, `NREM`, `"0.02 Hz"`, `NREM substates`, `fragility`.

**Two research communities, two methods — don't assume comparability.** The Zurich/Bristol lineage (Dimitriades/Huber, Grollero) fits a Gaussian to the sigma-envelope spectrum and reports peak frequency / bandwidth / strength — this project's pipeline. The Boston group (Sun, Westover) computes ISO *relative band power* in a fixed 0.005–0.03 Hz window with no peak fit, so it reports no frequency or bandwidth at all. Peak frequency and bandwidth therefore have almost no comparison literature outside the Zurich lineage.

**Why:** the two-name split is invisible from the code and from `CLAUDE.md`, and the existing prose rule ("never write ISO") makes it easy to assume ISO is simply someone else's wrong term rather than the search key for half the literature.

**How to apply:** When writing thesis prose, captions, abstracts, or commenting on results, always expand ISFS as "Infra-Slow Fluctuations of Sigma Power", and **keep writing ISFS, never ISO** ([[project_scientific_story]]). But when *searching* or *reading*, treat ISO as a synonym and search it explicitly. The Discussion should state the ISFS/ISO equivalence once, since Yuval knows this literature under the ISO name. If you find the old "Frequency Shifts" expansion in code/docs, flag it (but don't silently fix `CLAUDE.md` without confirming — it is project documentation the user may want to update themselves).

Related: [[project_thesis]], [[project_central_parietal_roi]], [[project_scientific_story]], [[project_bibliography_expansion]], [[reference_dimitriades_citation_status]]
