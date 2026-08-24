---
name: project_final_pass_sent
description: "Thesis V2 final pre-send pass DONE and SENT to Yuval 2026-08-19; the two Sydney [TO SUPPLY] placeholders and the Dimitriades preprint citation were deliberately left in"
metadata:
  node_type: memory
  type: project
---

**Thesis V2 went to Yuval on 2026-08-19.** The final pre-send QA pass is closed. No before/after file
was written for this one — the edits were applied straight to [[reference_manuscript_gdoc]] after
per-item approval in chat, so this note is the only record.

**Two things ship KNOWINGLY unfinished — do not re-flag them as defects:**
- **The two `[TO SUPPLY: …]` markers in Methods §3.1** (Sydney recruitment route; Sydney aMCI
  recruitment + diagnostic criteria + diagnosing clinician). His words: *"keep the place holder. yuval
  is aware."* They are still in the sent version.
- **Dimitriades stays cited as the bioRxiv 2024 preprint**, with the four in-text "(2024)" mentions,
  even though it is published — see [[reference_dimitriades_citation_status]]. He declined the upgrade.

**What was fixed in the pass:**
- **A real citation error in §5.3:** the claim *"…proposed N2 sleep-EEG features, and reduced parietal
  fast-spindle activity in particular, as candidate early markers"* cited refs 26,27
  (`zhang2022alzheimerreview` + `gorgoni2016parietal`). Zhang is a PSG meta-analysis of **established
  AD** and proposes no early marker; the early-marker half is carried by refs 36–37 (Taillard, Liu),
  which is how §2.4 and §5.5 cite the same claim. Now `²⁷,³⁶⁻³⁷`. Ref 26 is still cited in §2.3, so
  nothing was orphaned. **Source of the bug: `chapters/05_discussion.md:33` still has the wrong keys.**
- A pure duplication in §5.3 (Grollero's peak-amplitude result stated twice, five sentences apart).
- "rather" 18 → 13, with no paragraph left holding more than one (§5.3 had three).
- `LC-NE` deleted from all three places (English abstract, Hebrew abstract, §2.2) — it was defined
  twice with two different expansions and then never used; `LC` stays defined in §5.2 where it is
  first actually used. `norepinephrine` → `noradrenergic` for family consistency.
- Abstract now expands `electroencephalography (EEG)`; Table 2 caption `NREM2` → `N2`.
- Em dashes 5 → 1, see [[feedback_thesis_prose_rules]] §6.
- Discussion subheadings 5.1–5.7 were the only headings in the document without an explicit bold run,
  so they rendered lighter than 2.x/3.x/4.x. Bolded to match.

**Verified clean at send:** 0 open Doc comments; no `#REF`/`TBD`/`XXX`/`~X`; reference list **63**
entries with first-appearance order unbroken 1→63 and every entry cited at least once; internal
arithmetic reconciles (80.8% fit rate vs the three group rates, 0.0219 Hz vs group means, 1023/104=9.8,
MoCA n=30+14=44, all three sex splits sum to their group n).

**Left open by his call** (all previously logged): no ethics/consent/funding/COI statement; the two-site
confound never tested or mentioned; whether Sydney participants had the same apnea screening §3.1
asserts for "the two older groups"; Fig S1's `±1σ` label (for `a·exp(−((f−b)/c)²)`, `c` is not σ).
No broad humanizer rewrite — reopening prose Yuval had already line-edited was judged the wrong risk.

Related: [[project_front_matter_pass]], [[project_yuval_review]], [[feedback_thesis_prose_rules]],
[[reference_sun2026_mecfs_not_miscitation]], [[feedback_doc_edits_review_first]]
