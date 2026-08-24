# library.bib build status

Built from PDFs in `I:/Shaked/ISO/thesis/references/`. All metadata was extracted with `pdftotext -l 2` from each PDF (no web lookup needed — title/journal/DOI were visible on pages 1–2 of every paper). Encoding: UTF-8 with diacritics preserved (e.g., André, Bédard, Géraldine, Chételat, Tübingen).

## Per-paper status

| Cite key | PDF | Status |
|---|---|---|
| `dimitriades2024isfs` | ISFS_Development_Dimitriades_2024.pdf | Confirmed. bioRxiv preprint, doi 10.1101/2024.11.06.620875, 14 co-authors captured from page 1. |
| `grollero2026iso` | ISO_sleep_Alzheimer_Grollero_2026.pdf | Confirmed. bioRxiv preprint, doi 10.64898/2026.04.09.717425, 6 authors. Note: post-thesis-analysis. |
| `zhang2022alzheimerreview` | Sleep_Alzheimer_review_Zhang_2022.pdf | Confirmed. Translational Psychiatry 12:136 (2022), doi 10.1038/s41398-022-01897-y, 9 authors. |
| `andre2025remslowing` | REM_slowing_cholinergic_denervation_MCI_André_2025.pdf (+ Supplementary) | Confirmed. medRxiv preprint, doi 10.1101/2025.04.28.25326545, 12 authors. Supplementary PDF is part of same preprint - single bib entry. |
| `schmitz2018cholinergic` | Alzheimer_Degeneration_Topography_Cholinergic_BF_Projections_Schmitz_2018.pdf | Confirmed. Cell Reports 24(1):38–46 (2018), doi 10.1016/j.celrep.2018.06.001, 5 authors. |
| `chen2024spindletiming` | Individualized_temporal_patterns_drive_spindle_timing_Chen_2024.pdf | Confirmed. PNAS 122(2):e2405276121, doi 10.1073/pnas.2405276121. Published 7 Jan 2025 (received 2024 - filename year reflects acceptance year, bib year is 2025 per PNAS). |
| `niethard2023spindleaging` | Aging_impairs_temporal_spindles_clustering_Niethard_2023.pdf | Confirmed. SLEEP 46(5):zsad011 (2023), doi 10.1093/sleep/zsad011. Single-author editorial commentary on Champetier et al. |
| `champetier2023spindlememory` | Age_changes_spindle_memory_consolidation_Champetier_2023.pdf | Confirmed. SLEEP 46(5):zsac282 (2023), doi 10.1093/sleep/zsac282, 15 authors. |
| `sharon2025slowwaves` | slow_waves_MOCA_Sharon_2025.pdf | Confirmed. Alzheimer's & Dementia (2025), doi 10.1002/alz.70247, 10 authors. Source publication for the TASMC dataset; cited in Methods 3.1. |
| (skipped) | YaelG_MSc_thesis.pdf | Intentionally not included per instructions. |

## Entries needing user verification

- **`chen2025spindletiming`**: ✅ RESOLVED (2026-05-28) — cite key renamed to `chen2025spindletiming` and PDF renamed to `Individualized_temporal_patterns_drive_spindle_timing_Chen_2025.pdf` to match the actual PNAS publication year (January 2025).
- **`grollero2026iso`**: ✅ CONFIRMED (2026-05-28) — Shaked confirmed the DOI `https://doi.org/10.64898/2026.04.09.717425`. The unusual `10.64898` prefix is correct for this preprint. No change needed.
- **`niethard2023spindleaging`**: ✅ RESOLVED (2026-08-08) — keep it, but demoted to a secondary cite. Verified against the publisher page: single-author Editorial on Champetier, no original data; its own contribution is a mechanistic argument (synaptic-plasticity windows, slow-oscillation coupling, noradrenergic aging). Removed from the Introduction aging bracket (Champetier alone carries the finding); retained in the Discussion paired with Champetier, where the noradrenergic argument is what is being invoked. Renumbered 15 → 28 in the Google Doc.
- **`andre2025remslowing`**: medRxiv preprint — check periodically for the peer-reviewed version before final submission and update journal/volume/pages.
- **`dimitriades2024isfs`**: bioRxiv preprint — also check for peer-reviewed version before submission.

## Verified additions (2026-06-07)

17 gap-list candidates verified against Crossref / publisher pages and appended to `library.bib`. Each DOI was resolved before import; seed best-guess metadata was corrected where it conflicted. Decisions: gap-list only (no backfill); Methods refs limited to tools actually used in the pipeline.

| Cite key | Verified against | Status / correction vs seed |
|---|---|---|
| `lecci2017infraslow` | Crossref 10.1126/sciadv.1602026 | Confirmed. *Science Advances* 3(2):e1602026 (2017). DOI is `...sciadv.1602026` (do not append article number). |
| `osorioforero2021noradrenergic` | Crossref 10.1016/j.cub.2021.09.041 | Confirmed (corrected). *Current Biology* 31(22):5009–5023 (2021). Seed **omitted author Christiane Devenoges** — added. Publisher pages are 5009–5023.e7; bib uses 5009–5023. |
| `kjaerby2022norepinephrine` | Crossref 10.1038/s41593-022-01102-9 | Confirmed (corrected). *Nat. Neurosci.* 25(8):1059–1070 (2022). Seed "et al." expanded to full 12 authors; issue 8 added; diacritic Björn Sigurdsson. |
| `cardis2021corticoautonomic` | Crossref 10.7554/eLife.65835 | Confirmed (corrected). *eLife* 10:e65835 (2021). Seed **omitted 2nd author Sandro Lecci** — inserted. |
| `fernandez2020spindles` | Crossref 10.1152/physrev.00042.2018 | Confirmed. *Physiological Reviews* 100(2):805–868 (2020). |
| `boutin2020spindleframework` | Crossref 10.1098/rstb.2019.0232 | Confirmed (corrected). *Phil. Trans. R. Soc. B* 375(1799):20190232 (2020). Seed lacked issue 1799. |
| `andrillon2011intracranial` | Crossref 10.1523/JNEUROSCI.2604-11.2011 | Confirmed. *J. Neurosci.* 31(49):17821–17834 (2011). Nir-lab lineage. |
| `purcell2017characterizing` | Crossref 10.1038/ncomms15930 + PMC5490197 | Confirmed (corrected). *Nat. Commun.* 8:15930 (2017). Seed "et al." expanded to full 12 authors (full given names from PMC). 15930 is an article number. |
| `gorgoni2016parietal` | Crossref 10.1155/2016/8376108 | Confirmed (corrected). *Neural Plasticity* 2016:8376108 (2016). Seed 6 authors + "et al." → full 13 authors; apostrophe in D'Atri. |
| `mander2017aging` | Crossref 10.1016/j.neuron.2017.02.004 | Confirmed. *Neuron* 94(1):19–36 (2017). Exact match. |
| `helfrich2018uncoupled` | Crossref 10.1016/j.neuron.2017.11.020 | Confirmed. *Neuron* 97(1):221–230 (2018). Publisher pages 221–230.e4; bib uses 221–230. |
| `taillard2019nonrem` | Crossref 10.3389/fneur.2019.00197 | Confirmed (corrected). *Front. Neurol.* 10:197 (2019). Seed 6 authors + "et al." → full 10 authors; diacritics Hélène, Jean-François. |
| `liu2020spindlebiomarkers` | Crossref 10.1007/s11325-019-01970-9 + PubMed 31786748 | **Flagged & corrected.** Seed had wrong year (2021→**2020**), volume (25→**24**, issue 2), pages (1239–1246→**637–651**), and no DOI. The seed's vol/pages belonged to a *different* paper (10.1007/s11325-020-02214-x, Li et al. on napping/stroke). Cite key changed `liu2021…`→`liu2020spindlebiomarkers` to follow AuthorYEARkeyword. |
| `tallonbaudry1997gamma` | Crossref 10.1523/JNEUROSCI.17-02-00722.1997 | Confirmed. *J. Neurosci.* 17(2):722–734 (1997). Canonical title uses "γ-Band"; bib uses lowercase "gamma-band" (acceptable). |
| `maris2007nonparametric` | Crossref 10.1016/j.jneumeth.2007.03.024 | Confirmed. *J. Neurosci. Methods* 164(1):177–190 (2007). Exact match. |
| `gramfort2013mnepython` | Crossref 10.3389/fnins.2013.00267 + Frontiers page | Confirmed. *Front. Neurosci.* 7:267 (2013). Full author list from publisher (Crossref stores first author only); added middle initial Engemann, D.A.; Hämäläinen diacritic confirmed. |
| `vallat2021yasa` | Crossref 10.7554/eLife.70092 | Confirmed. *eLife* 10:e70092 (2021). Exact match. |

## Verified additions (2026-06-14, drafting session)

Two references added while drafting the Introduction (not from the original gap list). Both DOIs web-verified before import.

| Cite key | Verified against | Status / notes |
|---|---|---|
| `molle2011fastslow` | doi 10.5665/SLEEP.1290 | Confirmed. Mölle, Bergmann, Marshall & Born, *SLEEP* 34(10):1411–1421 (2011). Canonical slow (frontal, ~11–13 Hz) vs fast (centroparietal, ~13–16 Hz) spindle distinction; cited in Intro to justify the 13–16 Hz fast-spindle band. |
| `lazar2019infraslow` | doi 10.1016/j.jneumeth.2018.12.002 | Confirmed. Lázár, Dijk & Lázár, *J. Neurosci. Methods* 316:22–34 (2019). First detailed human demonstration of ISO in spindle/sigma activity (most prominent in fast spindles, centro-parieto-occipital); cited in Intro before Dimitriades 2024. |

### Dropped from the gap list (not imported)
- **#6** (Lecci human follow-up) — not a concrete reference; the human-validation portion is already covered by `lecci2017infraslow`.
- **#19** Cohen (2014) *Analyzing Neural Time Series Data* (textbook) — dropped per user decision: Morlet conventions cited via `tallonbaudry1997gamma`; textbook not used in the pipeline.
- **#20** Delorme & Makeig (2004) EEGLAB — dropped: EEGLAB is not used in this pipeline (analysis is MNE-Python + custom Gabor-Morlet).

## Verified additions (2026-08-13, Yuval review — thesis-scale expansion)

**31 entries added**, taking `library.bib` from 31 to **62** (Yuval's email item #9 asked for ≥50). Scope was fixed in advance to the nine gaps he named; nothing else in the bibliography was restructured. Section-by-section mapping for the prose session is in `new_refs_annotated.md`.

Method: candidate DOIs resolved through `api.crossref.org` and compared field-by-field against the drafted entry (first-author surname, year, journal, volume/issue/pages, full title, author-list completeness, `type`/`subtype`). Pagination for the AMA-published entries, which Crossref records with a first page only, was cross-checked against PubMed. After import, **all 62 entries in the file were re-resolved**: 62/62 resolve, zero year mismatches, zero first-author mismatches, zero duplicate keys, zero duplicate DOIs, braces balanced, all diacritics intact.

Two prior decisions constrain this batch: **journal articles only** (no AASM manual, no textbook chapters — stage/PSG background is carried by `silber2007visualscoring`), and **strict ISFS only** for the "other conditions" gap (see the finding under that heading below).

| Cite key | Verified against | Status / correction vs draft |
|---|---|---|
| `markov2006normalsleep` | Crossref 10.1016/j.psc.2006.09.008 | Confirmed. *Psychiatr. Clin. North Am.* 29(4):841–853 (2006). Review. Shaked's suggested model entry for the general-sleep gap. |
| `silber2007visualscoring` | Crossref 10.5664/jcsm.26814 | Confirmed. *J. Clin. Sleep Med.* 3(2):121–131 (2007). 12 authors, Iber listed last. The AASM visual scoring rules as a citable journal article, standing in for the non-DOI manual. |
| `steriade1993slowoscillation` | Crossref 10.1523/JNEUROSCI.13-08-03252.1993 | Confirmed (corrected). *J. Neurosci.* 13(8):3252–3265 (1993). Crossref returns the title with an HTML entity (`&lt; 1 Hz`); written as `$<$1 {Hz}` for LaTeX. Crossref stores initials only (M/A/F) — expanded to Mircea / Angel / Florin. |
| `rasch2013memory` | Crossref 10.1152/physrev.00032.2012 | Confirmed. *Physiol. Rev.* 93(2):681–766 (2013). Review. |
| `klinzing2019consolidation` | Crossref 10.1038/s41593-019-0467-3 | Confirmed. *Nat. Neurosci.* 22(10):1598–1610 (2019). Review. |
| `li2018normalaging` | Crossref 10.1016/j.jsmc.2017.09.001 | Confirmed. *Sleep Med. Clin.* 13(1):1–11 (2018). Review. |
| `ju2014bidirectional` | Crossref 10.1038/nrneurol.2013.269 | Confirmed. *Nat. Rev. Neurol.* 10(2):115–119 (2014). Review. Em-dash in title written `---`. |
| `xie2013clearance` | Crossref 10.1126/science.1241224 | Confirmed. *Science* 342(6156):373–377 (2013). Full 13 authors; apostrophe in O'Donnell. |
| `lucey2019nremtau` | Crossref 10.1126/scitranslmed.aau6550 | Confirmed. *Sci. Transl. Med.* 11(474):eaau6550 (2019). Crossref title uses an en-dash in "non–rapid"; normalised to a hyphen. |
| `scheltens2021alzheimer` | Crossref 10.1016/S0140-6736(20)32205-4 | Confirmed (corrected). *Lancet* 397(10284):1577–1590 (2021). Crossref stores "Chételat, Gael"; set to **Gaël** to match the correct form and `champetier2023spindlememory`. Periods added to the Teunissen / van der Flier initials. |
| `petersen1999mci` | Crossref 10.1001/archneur.56.3.303 + PubMed 10190820 | Confirmed (corrected). *Arch. Neurol.* 56(3):303–**308** (1999). Crossref records the first page only (`303`); range from PubMed. Title carries the subtitle "Clinical Characterization and Outcome". |
| `albert2011mcicriteria` | Crossref 10.1016/j.jalz.2011.03.008 | Confirmed. *Alzheimer's & Dementia* 7(3):270–279 (2011). Full 14 authors. Non-breaking hyphen in "Aging‐Alzheimer's" written `--`. |
| `nasreddine2005moca` | Crossref 10.1111/j.1532-5415.2005.53221.x | Confirmed. *J. Am. Geriatr. Soc.* 53(4):695–699 (2005). Diacritic Bédirian, Valérie. **Fills a real hole — the MoCA is used throughout and was previously uncited.** |
| `drozario2020objectivesleep` | Crossref 10.1016/j.smrv.2020.101308 | Confirmed. *Sleep Med. Rev.* 52:101308 (2020). Systematic review + meta-analysis; article number, no issue. |
| `mitchell2009progression` | Crossref 10.1111/j.1600-0447.2008.01326.x + PubMed 19236314 | Confirmed (corrected). *Acta Psychiatr. Scand.* 119(4):252–265 (2009). Crossref gives initials and a non-breaking hyphen in "Shiri‐Feshki"; expanded to Alex J. / Mojtaba with an ASCII hyphen. |
| `malekahmadi2016reversion` | Crossref 10.1097/WAD.0000000000000145 + PubMed 26908276 | Confirmed. *Alzheimer Dis. Assoc. Disord.* 30(4):324–330 (2016). Single-author meta-analysis. Note: a PMID guessed during checking (26840545) returned a *different* paper in the same issue (Lee et al., DOI …0143) — the DOI-verified entry is the correct one. |
| `roberts2014reversion` | Crossref 10.1212/WNL.0000000000000055 + PubMed 24353333 | Confirmed. *Neurology* 82(4):317–325 (2014). Primary population-based cohort (Mayo Clinic Study of Aging). |
| `jicha2006neuropathologic` | Crossref 10.1001/archneur.63.5.674 + PubMed 16682537 | Confirmed (corrected). *Arch. Neurol.* 63(5):674–**681** (2006). Crossref records the first page only; range from PubMed. Primary autopsy series. |
| `ferman2013nonamnestic` | Crossref 10.1212/01.wnl.0000436942.55281.47 + PubMed 24212390 | Confirmed. *Neurology* 81(23):2032–2038 (2013). Full 16 authors. Primary longitudinal cohort. |
| `terzano2001cap` | Crossref 10.1016/S1389-9457(01)00149-6 | Confirmed. *Sleep Medicine* 2(6):537–553 (2001). Full 12 authors. The CAP consensus scoring rules. |
| `parrino2012cap` | Crossref 10.1016/j.smrv.2011.02.003 | Confirmed. *Sleep Med. Rev.* 16(1):27–45 (2012). Review. |
| `maestri2015capmci` | Crossref 10.1016/j.sleep.2015.04.027 + PubMed 26298791 | Confirmed. *Sleep Medicine* 16(9):1139–1145 (2015). Primary (11 MCI / 11 AD / 11 controls). |
| `zheng2026capdementia` | Crossref 10.1002/alz.71331 | Confirmed. *Alzheimer's & Dementia* 22(4):e71331 (2026). Primary prospective cohort (MrOS). **A second Crossref record exists for the same work as a conference abstract — excluded, see below.** |
| `silvani2026cycles` | Crossref 10.1016/j.smrv.2026.102351 | Confirmed. *Sleep Med. Rev.* 90:102351 (2026). Review. An SSRN preprint record also exists (10.2139/ssrn.7014458) — the journal version is used. |
| `vanhatalo2004infraslow` | Crossref 10.1073/pnas.0305375101 | Confirmed. *PNAS* 101(14):5053–5057 (2004). Crossref stores initials only; **left as initials rather than inventing given names** (unlike earlier batches, no publisher page was consulted to expand them). |
| `fultz2019coupled` | Crossref 10.1126/science.aax5440 | Confirmed. *Science* 366(6465):628–631 (2019). |
| `dimitriades2025schizophrenia` | Crossref 10.1016/j.schres.2025.09.029 + PubMed 41086780 | Confirmed. *Schizophrenia Research* 285:295–303 (2025). Full 17 authors. **Three Crossref records exist for this work** — the journal article (used), a bioRxiv preprint (10.1101/2025.04.23.650209), and a conference abstract (excluded, see below). |
| `osorioforero2025gatekeeper` | Crossref 10.1038/s41593-024-01822-0 | Confirmed. *Nat. Neurosci.* 28(1):84–96 (2025). En-dash "NREM–REM" written `{NREM}--{REM}`. |
| `mather2016locuscoeruleus` | Crossref 10.1016/j.tics.2016.01.001 | Confirmed. *Trends Cogn. Sci.* 20(3):214–226 (2016). Review. |
| `braak2011stages` | Crossref 10.1097/NEN.0b013e318232a379 | Confirmed. *J. Neuropathol. Exp. Neurol.* 70(11):960–969 (2011). Primary neuropathological series. |
| `dahl2019rostrallc` | Crossref 10.1038/s41562-019-0715-2 | Confirmed. *Nat. Hum. Behav.* 3(11):1203–1214 (2019). Diacritics Düzel, Kühn. Primary human MRI study. |

### Screened and not imported

Publication-type failures — these are the reason the type check was run on every candidate:

| Candidate | Why dropped |
|---|---|
| Petersen et al. (2018) AAN practice guideline, 10.1212/WNL.0000000000004826 | Crossref returns the title as "Practice guideline update summary: Mild cognitive impairment **[RETIRED]**" with **zero authors**. A retired guideline, and not primary evidence. `albert2011mcicriteria` does this job. |
| 10.1016/j.sleep.2025.107031 | *Sleep Medicine* 138 conference abstract of the schizophrenia ISFS study (authors reduced to initials, abstract-range article number). Superseded by the *Schizophrenia Research* article. |
| 10.1016/j.sleep.2025.108276 | *Sleep Medicine* 138 conference abstract of the CAP-dementia study — only 2 of the 8 authors. Superseded by the *Alzheimer's & Dementia* article. |
| 10.1016/j.sleep.2017.11.529 | Lázár, *Sleep Medicine* conference abstract. The peer-reviewed version is already in the library as `lazar2019infraslow`. |

Editorial drops — all verified fine, judged redundant against what the library already holds. Listed so they can be reinstated without re-searching:

| Candidate | Why dropped |
|---|---|
| Betts et al. (2019) *Brain*, 10.1093/brain/awz193 | 42-author consensus review. LC coverage is carried by primary work (`braak2011stages`, `dahl2019rostrallc`) plus one framing review (`mather2016locuscoeruleus`). |
| Schneider et al. (2009) *Ann. Neurol.*, 10.1002/ana.21706 | Makes the same MCI-neuropathology-heterogeneity point as `jicha2006neuropathologic`, which is more directly on Yuval's wording. Best reserve pick if C571 needs a second autopsy citation. |
| Jack et al. (2018) NIA-AA Research Framework, 10.1016/j.jalz.2018.02.018 | Biomarker framework not used in this thesis. |
| Peter-Derex et al. (2015) *Sleep Med. Rev.*, 10.1016/j.smrv.2014.03.007 | Sleep-in-AD already covered by the newer `zhang2022alzheimerreview` meta-analysis. |
| Brown et al. (2012) *Physiol. Rev.*, 10.1152/physrev.00032.2011 | Overlaps `markov2006normalsleep`. |
| Massimini et al. (2004), 10.1523/JNEUROSCI.1318-04.2004 | Slow-wave signature carried by `steriade1993slowoscillation`. |
| Diekelmann & Born (2010), 10.1038/nrn2762 | Superseded by the fuller `rasch2013memory` from the same group. |
| Scullin & Bliwise (2015), 10.1177/1745691614556680 | Overlaps `mander2017aging` + `li2018normalaging`. |
| Carrier et al. (2011), 10.1111/j.1460-9568.2010.07543.x | Overlaps the existing aging block. |
| Petersen (2004) *J. Intern. Med.*, 10.1111/j.1365-2796.2004.01388.x | MCI subtypes carried by `petersen1999mci` + `ferman2013nonamnestic`. |
| Watson (2018) *Front. Syst. Neurosci.*, 10.3389/fnsys.2018.00044 | Single-author perspective; `silvani2026cycles` is newer and more targeted. |
| Turi et al. (2025) *eLife*, 10.7554/eLife.100196 | Rodent serotonergic infra-slow rhythm in dentate gyrus — not sigma-power, so out of scope under the strict-ISFS decision. |
| Qin et al. (2023) *Sleep Med. Rev.*, 10.1016/j.smrv.2022.101734 | Sleep-and-cognition meta-analysis; not needed for any gap Yuval named. |

### ~~Finding: ISFS outside aging and MCI is a near-empty literature~~ — SUPERSEDED 2026-08-14

The original conclusion here was that exactly **one** peer-reviewed study applied the ISFS to a clinical population outside aging/cognitive impairment. **That was a search artefact**, not a fact about the field: every query used the *ISFS* naming. Shaked pointed out that the same phenomenon is widely published as **"infra-slow oscillations (ISO)" of sigma / spindle power**. Re-running the sweep on the ISO terminology found more. See the 2026-08-14 section below.

## Verified additions (2026-08-14, ISO-term re-sweep)

**9 further entries**, taking `library.bib` from 62 to **71**.

Cause: the 2026-08-13 sweep searched only the *ISFS* naming. The Lüthi/Bellesi/Nedergaard lineage and the clinical sleep-EEG literature publish the same phenomenon as **infra-slow oscillation (ISO)** of sigma or spindle power. Re-searching PubMed under the ISO terminology — seven query formulations covering `infraslow`/`infra-slow` × `sigma`/`spindle`/`NREM`, `"0.02 Hz" AND sleep`, `NREM substates`, and `fragility` — returned 74 unique PMIDs, of which these 9 were kept.

All 9 resolved through Crossref before import; after import all **71/71** entries in the file re-resolve, with zero year mismatches, zero first-author mismatches, no duplicate keys or DOIs.

| Cite key | Verified against | Status / notes |
|---|---|---|
| `liu2026autism` | Crossref 10.1111/jsr.70309 + PubMed 41668405 | Confirmed. *J. Sleep Res.* 35(4):e70309 (2026). Primary. Measures ISO relative power (0.005–0.03 Hz) in 26 autistic vs 27 typically developing children. **A second clinical population.** |
| `sun2026longcovid` | Crossref 10.1093/sleep/zsag090 + PubMed 42017829 | Confirmed. *SLEEP* 49(8):zsag090 (2026). Primary. Reports elevated **ISO power in the slow sigma band (11–13 Hz)** in ME/CFS. **A third clinical population, and the most explicit ISO-of-sigma measurement outside aging/MCI.** |
| `picchioni2011infraslow` | Crossref 10.1016/j.brainres.2010.12.035 | Confirmed. *Brain Research* 1374:63–72 (2011). Primary EEG/fMRI. En-dash in "cortical–subcortical" written `--`. |
| `dash2019infraslow` | Crossref 10.1093/sleep/zsz170 + PubMed 31353415 | Confirmed. *Sleep* 42(12):zsz170 (2019). Single-author primary rodent study. Infra-slow fluctuation of **slow-wave** activity — the scope-honesty citation. |
| `parrino2025phasic` | Crossref 10.1016/j.clinph.2025.2111004 + PubMed 40961560 | Confirmed. *Clin. Neurophysiol.* 179:2111004 (2025). Review. Explicitly links CAP periodicity to LC infra-slow oscillatory activity. |
| `osorioforero2022locuscoeruleus` | Crossref 10.3390/ijms23095028 + PubMed 35563419 | Confirmed. *Int. J. Mol. Sci.* 23(9):5028 (2022). Review. LC-during-sleep, Lüthi lab. |
| `luthi2025microarousals` | Crossref 10.1016/j.neuron.2024.12.009 + PubMed 39809276 | Confirmed. *Neuron* 113(4):509–523 (2025). Review. States that weakened noradrenergic infra-slow fluctuations occur in neurodegenerative disease. |
| `hauglund2025vasomotion` | Crossref 10.1016/j.cell.2024.11.027 + PubMed 39788123 | Confirmed. *Cell* 188(3):606–622 (2025). Primary. Publisher pages are 606–622.e17; bib uses 606–622, matching the `helfrich2018uncoupled` precedent. |
| `kjaerby2026neuromodulators` | Crossref 10.1016/j.isci.2025.114554 + PubMed 41630901 | Confirmed. *iScience* 29(2):114554 (2026). Primary. Diacritics Kovács, Schiøler. |

### Screened and not imported (ISO sweep)

| Candidate | Why dropped |
|---|---|
| Edvardsson et al. (2026) *Front. Hum. Neurosci.*, 10.3389/fnhum.2026.1832178 | Title matches on "infra-slow EEG" but it is a neurofeedback *intervention* trial for insomnia (n = 8 analysed, single-case design). No ISO-of-sigma measurement. |
| Turi et al. (2025) *eLife*, 10.7554/eLife.100196 | Re-surfaced under the ISO terminology; still rodent serotonergic infra-slow activity in dentate gyrus, not sigma power. Stays dropped. |
| Watson (2018) *Front. Syst. Neurosci.*, 10.3389/fnsys.2018.00044 | Re-surfaced; still a single-author perspective. `silvani2026cycles` and `parrino2025phasic` cover the ground better. |
| Jacobsen et al. (2026) bioRxiv, Xiang et al. (2023) *Front. Cell. Neurosci.*, Rolle et al. (2025) *Cell Rep.*, Teng et al. (2025) *PNAS*, Miyawaki et al. (2017) *Sleep* | All genuine infra-slow/NREM-substate work, but the LC-noradrenergic mechanism is already carried by six entries (`lecci2017infraslow`, `osorioforero2021noradrenergic`, `osorioforero2022locuscoeruleus`, `osorioforero2025gatekeeper`, `kjaerby2022norepinephrine`, `kjaerby2026neuromodulators`). Held in reserve; Jacobsen is a preprint. |
| Blasiak et al. (2013) pupil ISO; Chrobok et al. (2017, 2019) absence-epilepsy thalamic ISO | Infra-slow oscillations, but not sleep sigma/spindle power. Out of scope. |

### Finding, revised: ISO of sigma/spindle power outside aging and MCI

There are now **three** clinical populations, not one: schizophrenia (`dimitriades2025schizophrenia`), autism (`liu2026autism`), and long COVID / ME-CFS (`sun2026longcovid`). Two distinct research communities are involved and they do **not** use the same method — the Zurich group fits a Gaussian to the sigma-envelope spectrum and reports peak frequency, bandwidth and AUC (this thesis's pipeline), whereas the Boston group (Sun, Westover, shared across `liu2026autism` and `sun2026longcovid`) computes ISO **relative band power** in a fixed 0.005–0.03 Hz window with no peak fit. Directions also differ: strength is *reduced* in schizophrenia and *elevated* in ME/CFS. The comparability table in `new_refs_annotated.md` sets this out, and it is a better answer to Yuval's "do people use same methods; do they report results in similar aspects" than the original near-empty finding.

## Verified additions (2026-08-14, mined from the C557 locus coeruleus review)

**13 further entries**, taking `library.bib` from 71 to **84**. All 84 DOIs resolve; zero year or first-author mismatches; no duplicate keys or DOIs.

Shaked obtained the review Yuval offered in C557: `thesis/references/Nir_etAl_LC_NE_Sleep_Review.docx`. **It is Yuval's own lab review** — Nir, Matosevich, Zelinger, Falach, Kimchy, Regev, *"Beyond Arousal: The Locus Coeruleus–Norepinephrine System as a Multimodal Orchestrator of Sleep Physiology"*. "NoaR" is **Noa Regev**, the last author.

### Decision: the review itself is NOT cited

It was briefly entered as `@unpublished{nir2026locuscoeruleus}` and then **removed by decision (Shaked, 2026-08-14)**. Confirmed against Crossref and PubMed that no journal version and no preprint exists, so there was no DOI to cite and a thesis reference list would have carried a bare "manuscript" entry for the supervisor's own unpublished work.

**Instead it is used as a source**: its 227-reference bibliography was mined for the published primary and review literature the thesis was missing, and those are cited directly. If the review is published before submission, reconsider — it would then be a straightforward `@article`. **Do not re-add it as `@unpublished`.**

The review also independently validates the 2026-08-14 ISO sweep: its own citations include `osorioforero2025gatekeeper` (its ref 162), `kjaerby2026neuromodulators` (ref 216), `hauglund2025vasomotion` (ref 128), `osorioforero2022locuscoeruleus` (ref 141) and `lecci2017infraslow` (ref 114) — all already imported before it arrived.

| Cite key | Verified against | Status / notes |
|---|---|---|
| `bergel2026conserved` | Crossref 10.1038/s41593-025-02159-y | Confirmed. *Nat. Neurosci.* 29(3):543–550. 17 authors; diacritics Sébastien, Chloé. Year: Crossref `published-online` is 2025-12-29 but `published-print` is 2026-03 — **2026** used, matching the print issue and the review's own citation. (review ref 77) |
| `poe2020locuscoeruleus` | Crossref 10.1038/s41583-020-0360-9 | Confirmed. *Nat. Rev. Neurosci.* 21(11):644–659. Review, 14 authors. (ref 50) |
| `matchett2021vulnerability` | Crossref 10.1007/s00401-020-02248-1 | Confirmed. *Acta Neuropathol.* 141(5):631–650 (2021). Review. **The central C557 framing citation.** (ref 154) |
| `theofilas2017stereology` | Crossref 10.1016/j.jalz.2016.06.2362 | Confirmed. *Alzheimer's & Dementia* 13(3):236–246. Year note: the DOI string contains "2016" and Crossref `published-online` is 2016, but `published-print` is **2017** — 2017 used. Primary, 20 authors. (ref 170) |
| `zarow2003neuronalloss` | Crossref 10.1001/archneur.60.3.337 + PubMed | Confirmed (corrected). *Arch. Neurol.* 60(3):337–**341**. Crossref records the first page only, as with the other AMA entries; range from PubMed. Primary. (ref 196) |
| `weinshenker2018noradrenergic` | Crossref 10.1016/j.tins.2018.01.010 | Confirmed. *Trends Neurosci.* 41(4):211–223. Single-author review. (ref 174) |
| `galgani2023locuscoeruleus` | Crossref 10.1111/ene.15556 | Confirmed. *Eur. J. Neurol.* 30(1):32–46 (2023). Primary, 21 authors. Crossref wraps part of the title in `<scp>` markup — stripped. The review cites it pre-pagination as "00, 1–15". **Serves C557 and C571 simultaneously.** (ref 180) |
| `vanegroo2022sleepwake` | Crossref 10.1016/j.smrv.2022.101592 | Confirmed. *Sleep Med. Rev.* 62:101592. Review. (ref 189) |
| `vanegroo2021awakenings` | Crossref 10.1186/s13195-021-00902-8 | Confirmed. *Alzheimer's Res. Ther.* 13(1):159. Primary 7T MRI, 3 authors. (the real paper behind review ref 190 — see the drop list) |
| `mander2016biomarker` | Crossref 10.1016/j.tins.2016.05.002 | Confirmed. *Trends Neurosci.* 39(8):552–566. Review. Distinct from the existing `mander2017aging` (*Neuron*, healthy aging). (ref 185) |
| `shokrikojori2018amyloid` | Crossref 10.1073/pnas.1721694115 | Confirmed. *PNAS* 115(17):4483–4488. Primary human PET, 17 authors. Greek beta in the title written `$\beta$`. (ref 130) |
| `slutsky2024dyshomeostasis` | Crossref 10.1038/s41583-024-00797-y | Confirmed. *Nat. Rev. Neurosci.* 25(4):272–284. Single-author review. Inna Slutsky is also thanked in the review's acknowledgements. (ref 188) |
| `teng2025synchrony` | Crossref 10.1073/pnas.2514202122 | Confirmed. *PNAS* 122(52):e2514202122. Primary. (ref 76) |

### Screened and not imported (review-bibliography mining)

| Candidate | Why dropped |
|---|---|
| `10.1002/alz.085381` — Van Egroo, "Relationships between locus coeruleus microstructural integrity, sleep, and 24-h rest-activity rhythm… 7T MRI" | **Conference abstract.** *Alzheimer's & Dementia* issue **S3**, no pages — an AAIC supplement. Traced to the real published 7T paper instead (`vanegroo2021awakenings`). The third conference-abstract trap caught in this bibliography. |
| Lüthi, Franken, Fulda, Siclari & Van Someren (2023), "Do all norepinephrine surges disrupt sleep?", 10.1038/s41593-023-01313-8 | Two pages in *Nat. Neurosci.* 26(6):955–956 — a News & Views commentary, not primary evidence. Its inverted-U argument is carried more fully by `luthi2025microarousals`. Same reasoning that demoted `niethard2023spindleaging`. |
| Betts et al. (2019) *Brain*, 10.1093/brain/awz193 | Still out. The review does cite it (its ref 194), so it is a legitimate reinstatement candidate if an LC-imaging-methods citation is ever wanted; LC coverage is otherwise complete without it. |
| Van Egroo et al. (2024) *Ann. Neurol.* 95(4):653–664, 10.1002/ana.26880; Koshmanova et al. (2023) *JCI Insight* 8(20), 10.1172/jci.insight.172008 | Both verified and both good, but REM-quality and rest-activity outcomes rather than NREM. Held in reserve — note Yuval objected to REM material in this very comment. |
| ~200 further references in the review | Out of scope: LC cellular/circuit anatomy, optogenetic methods, stress/PTSD, pupillometry, sensory processing. Available in `thesis/references/` if a section ever needs them. |

### Note on `sharon2025slowwaves` (Omer's paper, also named in C557)

Already in the library and confirmed to be the only Omer Sharon paper on slow-wave activity. **The existing entry is incomplete** — it has no volume, issue or pages; Crossref gives **21(5):e70247**. Not changed here, since it is a pre-existing entry. Worth fixing in the same pass as the two preprint upgrades below.

### Flagged, not changed

Found while re-resolving all 62 DOIs. **No edits were applied** — these are existing entries and outside the agreed scope for this session.

- **`dimitriades2024isfs`** — the bib entry is the bioRxiv preprint (10.1101/2024.11.06.620875). A peer-reviewed version now exists: *Scientific Reports* (2026), **10.1038/s41598-026-58423-z**, resolves, 14 authors matching. The published title differs slightly: "The infraslow fluctuation of sigma power during sleep **and its links to** markers of arousal and memory reactivation across development".
- **`andre2025remslowing`** — the bib entry is the medRxiv preprint (10.1101/2025.04.28.25326545). Published version: *Molecular Psychiatry* (2026), **10.1038/s41380-026-03635-y**, resolves. Both the title and the author list changed — now "**Associations between** REM sleep EEG slowing **and** brain cholinergic denervation in aging and Mild Cognitive Impairment", with **14** authors (Olga Fliaguine and Serge Gauthier added versus the 12 in the current entry).
- **`grollero2026iso`** — re-checked, still a preprint (`posted-content`). No change needed.
- **`visbrain` and `sleepeegpy`** — the only two cite keys in the file that do not follow the `authorYEARkeyword` convention. Pre-existing; left alone.

## Lit-review gap list (original suggestions — superseded by the table above)

These were *suggestions only*. Items 1–5, 7–18 are now verified and imported (see table); items 6, 19, 20 were dropped. Retained below for provenance.

### Rodent ISO / infraslow sigma (the rodent literature that grounds the human ISFS work)

1. **Lecci, S., Fernandez, L.M.J., Weber, F.D., Cardis, R., Chatton, J.-Y., Born, J., Lüthi, A.** (2017). *Coordinated infraslow neural and cardiac oscillations mark fragility and offline periods in mammalian sleep.* Science Advances 3:e1602026. — Foundational rodent paper defining 0.02 Hz sigma fluctuation as fragility/offline alternation; cited by every ISFS paper.
2. **Osorio-Forero, A., Cardis, R., Vantomme, G., Guillaume-Gentil, A., Katsioudi, G., Devenoges, C., Fernandez, L.M.J., Lüthi, A.** (2021). *Noradrenergic circuit control of non-REM sleep substates.* Current Biology 31(22):5009–5023. — Demonstrates the locus coeruleus drives infraslow sigma fluctuation in mice; mechanistic basis for human ISFS interpretation.
3. **Kjaerby, C., Andersen, M., Hauglund, N., Untiet, V., Dall, C., Sigurdsson, B., et al.** (2022). *Memory-enhancing properties of sleep depend on the oscillatory amplitude of norepinephrine.* Nature Neuroscience 25:1059–1070. — LC noradrenergic infraslow oscillations and sleep-dependent memory in mice.
4. **Cardis, R., Lecci, S., Fernandez, L.M.J., Osorio-Forero, A., Chu Sin Chung, P., Fulda, S., Decosterd, I., Lüthi, A.** (2021). *Cortico-autonomic local arousals and heightened somatosensory arousability during NREMS of mice in neuropathic pain.* eLife 10:e65835. — Pairs the rodent ISO framework with arousability measures relevant to fragile/protected periods.
5. **Fernandez, L.M.J., Lüthi, A.** (2020). *Sleep spindles: mechanisms and functions.* Physiological Reviews 100(2):805–868. — Comprehensive review of thalamic spindle generation and infraslow dynamics — useful methods citation for spindle biology.

### Human NREM2 sigma envelope / spindle dynamics

6. **Lecci, S. (and Lüthi colleagues)** any direct human follow-up — e.g., the human-validation portion of Lecci et al. 2017 (already in #1) covers EEG humans; check for follow-up by the Lüthi group on adults.
7. **Boutin, A., Doyon, J.** (2020). *A sleep spindle framework for motor memory consolidation.* Philosophical Transactions B 375:20190232. — Source of the 6-s "spindle train" interval used by Champetier et al.; cite for clustering methodology.
8. **Andrillon, T., Nir, Y., Staba, R.J., Ferrarelli, F., Cirelli, C., Tononi, G., Fried, I.** (2011). *Sleep spindles in humans: insights from intracranial EEG and unit recordings.* Journal of Neuroscience 31(49):17821–17834. — Yuval Nir lab paper on intracranial spindle physiology — appropriate lab-lineage citation.
9. **Purcell, S.M., Manoach, D.S., Demanuele, C., Cade, B.E., Mariani, S., Cox, R., et al.** (2017). *Characterizing sleep spindles in 11,630 individuals from the National Sleep Research Resource.* Nature Communications 8:15930. — Large-N normative spindle data; useful reference for "what is a typical adult spindle topography".

### MCI / aging EEG biomarkers

10. **Gorgoni, M., Lauri, G., Truglia, I., Cordone, S., Sarasso, S., Scarpelli, S., et al.** (2016). *Parietal fast sleep spindle density decrease in Alzheimer's disease and amnesic Mild Cognitive Impairment.* Neural Plasticity 2016:8376108. — Direct AD/MCI spindle-density paper most parallel to the thesis comparison.
11. **Mander, B.A., Winer, J.R., Walker, M.P.** (2017). *Sleep and human aging.* Neuron 94(1):19–36. — Standard review on aging-sleep interactions.
12. **Helfrich, R.F., Mander, B.A., Jagust, W.J., Knight, R.T., Walker, M.P.** (2018). *Old brains come uncoupled in sleep: slow wave–spindle synchrony, brain atrophy, and forgetting.* Neuron 97(1):221–230. — Spindle/SO coupling decline with aging; complements the ISFS-with-aging story.
13. **Taillard, J., Sagaspe, P., Berthomier, C., Brandewinder, M., Amieva, H., Dartigues, J.-F., et al.** (2019). *Non-REM sleep characteristics predict early cognitive impairment in an aging population.* Frontiers in Neurology 10:197. — Predictive sleep-EEG features for cognitive decline.
14. **Liu, S., Pan, J., Tang, K., Lei, Q., He, L., Meng, Y., et al.** (2021). *Sleep spindles, K-complexes, limb movements and sleep stage proportions may be biomarkers for amnestic mild cognitive impairment and Alzheimer's disease.* Sleep & Breathing 25:1239–1246. — MCI-specific spindle biomarker work.

### Methods: Morlet wavelet and cluster-based permutation tests

15. **Tallon-Baudry, C., Bertrand, O., Delpuech, C., Pernier, J.** (1997). *Oscillatory γ-band (30–70 Hz) activity induced by a visual search task in humans.* Journal of Neuroscience 17(2):722–734. — Classic reference for Morlet-wavelet time-frequency analysis in EEG.
16. **Maris, E., Oostenveld, R.** (2007). *Nonparametric statistical testing of EEG- and MEG-data.* Journal of Neuroscience Methods 164(1):177–190. — The canonical citation for cluster-based permutation testing — required in your Methods.
17. **Gramfort, A., Luessi, M., Larson, E., Engemann, D.A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., Hämäläinen, M.** (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience 7:267. — Required citation for MNE-Python.
18. **Vallat, R., Walker, M.P.** (2021). *An open-source, high-performance tool for automated sleep staging.* eLife 10:e70092. — Required citation for YASA (used in step2 sigma analysis).
19. **Cohen, M.X.** (2014). *Analyzing Neural Time Series Data: Theory and Practice.* MIT Press. — Standard methods textbook for Morlet-wavelet conventions (number of cycles, frequency resolution).
20. **Delorme, A., Makeig, S.** (2004). *EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics.* Journal of Neuroscience Methods 134(1):9–21. — Useful methods reference even if EEGLAB itself isn't used.

End of list.
