# Bibliography expansion for the thesis (Yuval review, item #9 + the topic gaps)

## Context

Yuval's review (triaged in `thesis/reviews/yuval_review_triage.md`) asks for **at least 50
references**. The manuscript cites 29; `thesis/references/library.bib` holds 31 entries. The
shortfall is not cosmetic — his margin note *"Put one more general paragraph to begin with –
**for thesis (unnecessary for paper)**"* settles that this is a full Master's thesis, so the
Introduction has to grow from a 782-word ISFS-focused funnel into proper thesis sections
(sleep in general → sleep & memory → sleep & aging → neurodegeneration → AD → MCI/aMCI),
and the Discussion needs two new paragraphs (broader aging/MCI sleep changes incl. **CAP**;
ISFS irrespective of aging/MCI). Those sections cannot be written without the sources first.

Three of his comments are specifically bibliographic:

- **C571** — "Some aMCI will never develop AD… Please read more about this and describe also
  in intro." The current Discussion says "perhaps because MCI is an earlier disease stage",
  which is exactly the framing he rejects. Needs real conversion/reversion/neuropathology
  evidence.
- **C557** — the cholinergic/REM passage in the Discussion is "out of context"; he wants
  **locus coeruleus degeneration** + NREM-sleep changes instead. He offers a lab review via
  Noa R.
- **Discussion inline note** — CAP ("look it up") and other infra-slow phenomena; and whether
  ISFS work outside aging/MCI uses comparable methods and reports comparable parameters.

Outcome: `library.bib` grows to ~55 verified entries, and the prose session gets an annotated
map telling it which reference supports which section, so it can write straight from it.

## Scope

**Owned by this session (the only files that get written):**
- `thesis/references/library.bib`
- `thesis/references/library_status.md`
- `thesis/references/new_refs_annotated.md` (new)

**Explicitly not touched:** `thesis/chapters/*`, any figure or stats script, the Google Doc.
**No commits.** Nothing else in the bibliography gets restructured on my own initiative — if I
find a problem with an existing entry I report it and stop (see "Report, don't fix" below).

## Decisions already taken

| Question | Answer |
|---|---|
| Noa R.'s LC review (C557) | Shaked will get it from Noa. I leave a clearly-marked, **commented-out** placeholder in `library.bib` + a pending row in `library_status.md`, and add published LC-aging primary sources around it so the section is writable now. |
| Non-DOI sources (AASM manual, textbook chapters) | **No.** Journal articles only, every one DOI-verifiable. Stage/PSG/scoring background goes to review articles instead — Markov & Goldman 2006 is the model. |
| ISFS outside aging/MCI | **Strict only** — a paper qualifies only if it actually computes an infra-slow fluctuation of sigma/spindle power. Expect ~2–4 hits; the sparseness itself gets reported as a finding the Discussion can state. |

## Approach

Follow the `literature-review` skill at **scoping-review** rigor: search protocol → search log →
screen → extract → verify. Sources: PubMed, Crossref, Semantic Scholar, publisher pages.
No subagents; searches run inline with WebSearch/WebFetch + the Crossref REST API.

### 1. Candidate search, by gap (target ~24 new entries → ~55 total)

Each bucket maps to a section Yuval named. Counts are targets, not quotas.

| # | Gap (his words) | New refs | Search angles |
|---|---|---|---|
| A | Sleep in general: what it is, stages, functions; PSG + EEG signatures | 4 | normal-sleep neurobiology reviews (Markov & Goldman 2006; Brown et al. *Physiol Rev* 2012); a DOI'd journal statement of the AASM visual scoring rules (Silber et al. *JCSM* 2007) in place of the manual; slow-wave physiology (Steriade / Massimini lineage). Spindles are already covered by `fernandez2020spindles`, `andrillon2011intracranial`, `purcell2017characterizing`, `molle2011fastslow` |
| B | Sleep, learning and memory | 3 | Rasch & Born 2013; Diekelmann & Born 2010; Klinzing/Niethard/Born 2019 |
| C | Sleep and aging | 2 | complement `mander2017aging` + `ohayon2004metaanalysis` with a normal-aging clinical review and a spindle/slow-wave-with-age primary study (Carrier lineage) |
| D | Sleep and neurodegeneration | 3 | bidirectional sleep–pathology reviews (Ju/Lucey/Holtzman); glymphatic clearance (Xie 2013); NREM-tau primary evidence (Lucey *Sci Transl Med* 2019) |
| E | Alzheimer's disease, and sleep in AD | 3 | one AD overview (*Lancet* seminar), one diagnostic-framework paper (NIA-AA), one sleep-in-AD review to sit beside `zhang2022alzheimerreview` |
| F | MCI and aMCI specifically; sleep in MCI | 4 | Petersen's MCI entity paper; NIA-AA MCI criteria (Albert 2011); **Nasreddine 2005 MoCA** — the MoCA is used in Methods and is currently uncited; D'Rozario 2020 objective-sleep-in-MCI meta-analysis |
| G | **aMCI ≠ early AD (C571)** | 4 | reversion-to-normal meta-analysis (Malek-Ahmadi 2016); conversion-rate meta-analysis of inception cohorts (Mitchell & Shiri-Feshki 2009); reverters still at elevated risk (Roberts 2014); **neuropathological heterogeneity** at autopsy — non-AD outcomes incl. hippocampal sclerosis, argyrophilic grain disease, Lewy bodies (Jicha 2006 / Schneider 2009 / Ferman 2013). This bucket must be primary evidence, not commentary |
| H | CAP and other infra-slow phenomena | 3–4 | Terzano 2001 CAP atlas/rules; Parrino 2012 CAP-as-sleep-instability review; CAP in MCI/AD if a real study exists; Vanhatalo 2004 *PNAS* infra-slow cortical oscillation in human sleep; the infra-slow hemodynamic/CSF timescale the Discussion already gestures at |
| I | ISFS outside aging/MCI (**strict**) | 2–4 | infra-slow sigma / NREM-substate work in insomnia, apnea, epilepsy, psychiatry, RBD/PD; plus the current Lüthi-lab LC mechanism papers not yet in the library. Each screened on: does it compute a sigma-envelope infra-slow spectrum, and does it report peak frequency / bandwidth / strength? |
| J | LC degeneration with age & in neurodegeneration + NREM changes (C557) | 3 | LC-and-aging review (Mather & Harley 2016); LC as earliest tau site (Braak 2011); LC integrity imaging as a biomarker (Betts 2019 / Dahl 2019). `sharon2025slowwaves` (Omer's paper, already entry 19) gets re-purposed in the annotated list from cohort-provenance-only to the NREM-slow-wave half of his comment. **+ placeholder for Noa's review** |

Screening rules, applied and logged: peer-reviewed primary work where the claim needs primary
evidence; reviews allowed only for framing and labelled as such; preprints labelled as
preprints; **publication type checked explicitly** on every entry — `niethard2023spindleaging`
already turned out to be a single-author editorial, and that mistake does not get repeated.

### 2. Verification (every new DOI, before import)

A throwaway script in the scratchpad (`.../scratchpad/verify_dois.py` — not written into the
repo) that, for each DOI:

1. `GET https://api.crossref.org/works/{doi}` with a polite `User-Agent` (mailto set).
2. Asserts the DOI **resolves** (HTTP 200, `status: ok`).
3. Compares Crossref metadata against the drafted bib entry — first author surname, year,
   `container-title`, volume/issue/pages, and full title — and flags every mismatch rather than
   silently overwriting.
4. Records `type` / `subtype` (`journal-article` vs `posted-content` vs `book-chapter`) so
   editorials, comments and preprints are caught and labelled.
5. Checks author lists are complete (no bare "et al." reaching the bib).

Run it over the **whole** file, new and old, so existing entries are checked for free.
Crossref reachability was confirmed during planning (`10.1016/j.neuron.2017.02.004` → 200 OK).
Anything Crossref cannot settle (author-list completeness, editorial status) is confirmed
against the publisher page with WebFetch, exactly as the 2026-06-07 batch was.

No reference gets written from memory. If a candidate cannot be verified, it is dropped and the
drop is logged.

### 3. Write the three files

**`library.bib`** — append only, in new `% ====` section banners matching the existing ones
(`% Rodent ISO / infraslow sigma (mechanistic background for human ISFS)` etc.). Per entry, the
existing field style is followed exactly: `@article{key,` with `author / title / journal /
volume / number / pages / year / doi` aligned on `=`, `{Braces}` protecting proper nouns and
acronyms (`{Alzheimer's}`, `{REM}`, `{MCI}`), UTF-8 diacritics preserved, and a `% notes:` line
immediately after each entry saying what it is and where it is cited. Keys follow
`authorYEARkeyword`, lowercase, e.g. `rasch2013memory`, `petersen2004mci`,
`malekahmadi2016reversion`, `terzano2001cap`, `mather2016locuscoeruleus`. Existing entries are
left byte-identical.

**`library_status.md`** — a new section in the established format:
`## Verified additions (2026-08-13, Yuval review — thesis-scale expansion)`, with the same
`| Cite key | Verified against | Status / correction vs seed |` table, one row per new entry
naming the Crossref DOI or publisher page it was checked against and any correction made to the
draft metadata. Plus a short "Screened and dropped" subsection with reasons, and a pending row
for Noa's LC review. Existing sections untouched.

**`thesis/references/new_refs_annotated.md`** (new, the prose session's working document):

- **Section map** — one table per gap A–J: cite key | one-line claim it supports | exact target
  location (`Intro §1.x new subsection` / `Discussion ¶N`) | primary vs review vs preprint.
- **C571 dossier** — a tight paragraph-ready summary of what the aMCI-outcome evidence actually
  shows (reversion rates, annual conversion rates, non-AD pathologies at autopsy), with the
  numbers and which reference carries each, so the Intro and Discussion rewrites are a matter of
  prose rather than re-reading papers.
- **ISFS-elsewhere comparability table** — the direct answer to *"Do people use same methods; do
  they report results in similar aspects"*: study | population/condition | how the envelope was
  obtained | which parameters reported (peak freq / bandwidth / strength) | comparable to ours?
- **Search log** — database, date, query string, filters, hits, kept (skill's reproducibility
  requirement).
- **Coverage count** — refs currently cited vs total in `library.bib`, so the ≥50 claim is
  checkable at a glance.

### 4. Report, don't fix

Findings that fall outside the stated scope get written into a short section at the end of
`new_refs_annotated.md` and raised in my summary — **no edits applied**. Already anticipated
from the triage doc (items B1/S2), to be re-confirmed against Crossref during the sweep:

- `dimitriades2024isfs` — bioRxiv preprint in the bib, but reportedly published in *Sci Rep*,
  18 Jun 2026, `10.1038/s41598-026-58423-z`.
- `andre2025remslowing` — medRxiv preprint in the bib, but reportedly published in
  *Mol Psychiatry*, 12 May 2026, `10.1038/s41380-026-03635-y`.
- `grollero2026iso` — still a preprint; re-checked.
- plus anything else the whole-file DOI sweep turns up.

## Verification

1. `python verify_dois.py thesis/references/library.bib` (scratchpad script) → every entry
   resolves, zero metadata mismatches on the new entries, publication type printed for each.
2. Key uniqueness and duplicate-DOI check across the whole file (same script).
3. Non-ASCII round-trip: re-read `library.bib` and confirm diacritics survived
   (Lázár, Mölle, Lüthi, André, Chételat + any new ones).
4. BibTeX parse: load with a parser in the `eeg_clean` venv (or `bibtexparser` if present) and
   confirm 55-ish entries parse with no syntax errors — the file is compiled by biber per its
   own header comment. If no parser is installed, a brace-balance + `@article{` count check.
5. Cross-check every cite key referenced in `new_refs_annotated.md` exists in `library.bib`.
6. Report the final count: entries in `library.bib` vs the ≥50 target.

## Out of scope / open

- Writing the Intro or Discussion prose (another session owns `thesis/chapters/*`).
- Inserting citations into the Google Doc.
- Noa R.'s LC review — blocked on Shaked; slot reserved.
