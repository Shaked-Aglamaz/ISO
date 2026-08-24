# Front matter, Abstract, Hebrew abstract and TOC — before / after

Pass run 2026-08-17. Closes the last **OPEN** row of the status table in
`yuval_review_triage.md:20` — *"Writing, remaining (Hebrew abstract, TOC, Abstract additions, title
page, acknowledgements)"* — plus Yuval's email items **#1** (missing TOC) and **#2** (Hebrew
abstract).

Manuscript = the Google Doc **"Shaked's Thesis V2"** (`1YpXrDGFlzRk_MxdBXllLlqTG-caWg1vkTzxwR1boDyY`).
Source of his edits = `thesis/reviews/Shaked's Thesis_YN.docx`, front-matter paragraphs 002–059, all
tracked insertions authored *Yuval Nir*, 2026-08-11T16:47Z. No comments are anchored in the front
matter; it is pure insert/delete.

> ## ✅ APPLIED 2026-08-17
>
> All of it is in the Doc. Shaked approved §1–§8 with amendments, which are folded in and recorded in
> **§11 — closing state**. Read §11 for what actually went in; §1–§10 below are the deliberation
> record and describe the *proposal*, not the final text, in four places (§3 acknowledgements,
> §5 abstract, §6 תקציר, and the new Discussion subheadings, which were not in the proposal at all).
>
> **One thing still to do by hand:** the table of contents itself. See §4.

---

## §0 — Decisions you took before drafting

| Question | Your answer |
|---|---|
| How much of his Abstract rewrite | **His whole rewrite**, adapted to V2 facts |
| Bandwidth in the Abstract | **Keep**, with the N2-duration caveat |
| Title-page date | **August, 2026** |
| Table of contents | **Native auto-TOC** — you click Insert → Table of contents |
| Acknowledgements | The drafted prose in §3; **no funding statement** |
| Hebrew title page (not his ask) | **Yes** |
| Student ID on title page (not his ask) | No |
| Ethics / consent / COI statements (finding S4) | No |

---

## §1 — Title page

His only tracked change on the whole page. Everything else — title, `By`, your name, `The thesis was
carried out under the supervision of`, `Professor Yuval Nir` — he left untouched.

**Before** (Doc indices 34–88, 288–300):

```
Sagol School of Neuroscience
Department of Physiology and Pharmacology (Medicine)

[title]
By
Shaked Aglamaz
The thesis was carried out under the supervision of
Professor Yuval Nir
July, 2026
```

**After:**

```
Sagol School of Neuroscience
Department of Neuroscience and Brain Disorders
Gray Faculty of Medical and Health Sciences

[title unchanged]
By
Shaked Aglamaz
The thesis was carried out under the supervision of
Professor Yuval Nir
August, 2026
```

His docx paras 003/004 read `Department of {{-Physiology -}}{{+Neuroscience and Brain Disorders+}}` /
`{{+Gray Faculty of Medical and Health Sciences+}}{{-and Pharmacology (Medicine)-}}` — so the faculty
line is a **new third line**, not a replacement of the department line. Applied exactly that way.
Existing 14 pt CENTER run styling preserved.

The date is the one thing he did not set: his copy still read `July, 2023`, V2 reads `July, 2026`,
and July 2026 is now past. Set to **August, 2026** on your instruction.

---

## §2 — Hebrew title page ← **CHECK THIS**

New page 2, mirroring the English page. This is *not* one of his asks — you approved it because the
TAU house format carries it. The model is `thesis/references/YaelG_MSc_thesis.pdf` p. 2, the last
thesis Yuval supervised on this cohort.

**Institutional names — verified, not translated by ear:**

| English | Hebrew | Source |
|---|---|---|
| Sagol School of Neuroscience | `בית הספר סגול למדעי המוח` | sagol.tau.ac.il, and Yael's p. 2 |
| Department of Neuroscience and Brain Disorders | `החוג למדעי העצב ומחלות נוירולוגיות` | med.tau.ac.il/head-departments (head: Prof. Eran Perlson) |
| Gray Faculty of Medical and Health Sciences | `הפקולטה למדעי הרפואה והבריאות ע"ש גריי` | med.tau.ac.il |

> ⚠ **Worth knowing:** TAU's official Hebrew for that department back-translates as *"Neuroscience
> and Neurological Diseases"*, not *"Brain Disorders"*. It is the same department — the page lists it
> under Perlson, whose older English listings still say *Department of Physiology and Pharmacology*,
> which is exactly the rename Yuval's edit reflects. I used the official Hebrew rather than a literal
> translation of his English. Say if you want the literal `מחלות מוח` instead.

**Proposed page:**

```
בית הספר סגול למדעי המוח
החוג למדעי העצב ומחלות נוירולוגיות
הפקולטה למדעי הרפואה והבריאות ע"ש גריי


הזדקנות, אך לא ליקוי קוגניטיבי קל אמנסטי, מעצבת מחדש
את המקצב האינפרה-איטי של עוצמת כישורי השינה


מאת

שקד אגלמז

החיבור בוצע בהנחייתו של

פרופסור יובל ניר

אוגוסט, 2026
```

**Two things only you can confirm:**

1. **Your name.** I wrote `שקד אגלמז`. Correct it if you spell it otherwise.
2. **The Hebrew title.** Term choices: *sleep spindles* → `כישורי שינה` (Yael's term), *infra-slow*
   → `אינפרה-איטי`, *amnestic MCI* → `ליקוי קוגניטיבי קל אמנסטי`. Yael's thesis used
   `הפרעה קוגנטיבית אמנסטית מתונה`, with two spellings that are non-standard; I did not copy them.

Set RTL through a direct Docs API `batchUpdate` (`paragraphStyle.direction = RIGHT_TO_LEFT`) — the
MCP's `applyParagraphStyle` has no direction field.

---

## §3 — Acknowledgements

**Before:** nothing. V2 has no Acknowledgements at all.

**His stub** (docx paras 016–019, inserted verbatim as below, ellipses and all):

```
Acknowledgements
I would like to thank …
e.g.
Noa Bregman for professional support at the Tel Aviv Sourasky Medical Center, and for referring
patients to the study; Rivi Tauman and Jenny Zitser for guidance with sleep medicine and PSG
monitoring Rotem Falach, Flavio Schmidig, … … for
```

**After** — all five of his names kept in the roles he assigned them:

> **Acknowledgements**
>
> I would like to thank Prof. Yuval Nir for his guidance, his scientific standards, and for trusting
> me with this project; Noa Bregman for professional support at the Tel Aviv Sourasky Medical Center,
> and for referring patients to the study; Rivi Tauman and Jenny Zitser for their guidance in sleep
> medicine and PSG monitoring; Rotem Falach and Flavio Schmidig for teaching me EEG analysis and for
> their advice throughout; and the members of the Nir lab for their help and good company. Finally, I
> thank the participants and their families for their time and willingness to take part.

No funding line, per your instruction. Nothing in the repo records a funder, grant number or ethics
approval anyway.

> **Noted, not done:** the Sydney site is unacknowledged. Sixteen of the 30 aMCI patients and nine of
> the 39 healthy older adults were recorded at CIRUS / Woolcock. His stub names only Tel Aviv people.
> Give me names and I will add them.

---

## §4 — Table of Contents (his email #1)

**Before:** absent. `00_front_matter.md:19` has a `## Table of contents` placeholder reading
`_Pandoc-generated._`, which never happened.

**What he pasted** into his copy (docx paras 023–054) was a full example TOC lifted from Yael Gat's
thesis, headed:

```
Table of Contents #here's an example from different thesis, adapt to yours#
```

It is her contents page, not ours — her rows are *Sleep, Learning and Memory*, *The memory paradigm*,
*Memory Performance* and so on. It also carries three copy-paste artefacts: a stray one-character row
`T` with no page number, `Sleep in MCI` mis-levelled as TOC1, and `Supplementary` / `References`
left unstyled. His `#adapt to yours#` covers all of it.

**After:** a native Google Docs TOC generated from our own headings.

**The Docs API cannot create one** — there is no request type for it, in the MCP or in the raw API.
So I insert the page and verify the heading structure; **you click Insert → Table of contents → with
page numbers**, once, and it fills itself in with live, clickable, self-updating page numbers.

What it will generate, from the headings already in the Doc:

```
Abstract
Introduction
    2.1 Human sleep and scalp EEG
    2.2 Infra-slow fluctuation of sigma power (ISFS) in NREM sleep
    2.3 Changes in sleep and EEG across aging and MCI
    2.4 Changes in sleep spindles and ISFS across aging and MCI
    2.5 The present study
Methods
    3.1 Participants  …  3.7 Software
Results
    4.1 Sleep differs with age…  …  4.6 No association with cognitive score
Discussion
Supplementary figures
References
```

Two consequences worth knowing before you click:

- **`Acknowledgements`, `Table of Contents` and `תקציר` are styled as bold centred body text, not as
  HEADING_2**, so they stay out of the generated list. That matches his model, whose first row is
  `Abstract 6`. Say if you would rather they appear.
- **The Discussion has no subsections** — it is one unbroken run of 41 paragraphs in both the Doc and
  `05_discussion.md`, so it contributes a single TOC row. Yael's has five. Not a defect, but he may
  notice the asymmetry against the model he sent.

---

## §5 — Abstract

You chose **his whole rewrite, adapted**. His tracked version rewrites essentially every sentence of
the V1 abstract; V2's abstract is the post-Flavio rewrite, so his edits do not re-anchor and the
paragraph is replaced wholesale.

**Before** (Doc index 309; `01_abstract.md:7`):

> During the second stage of non-rapid eye movement (NREM) sleep (N2), sleep spindles arrive in trains
> rather than at random: power in the fast-spindle sigma band (13–16 Hz) rises and falls roughly every
> 50 seconds, an infra-slow fluctuation of sigma power (ISFS) that in rodents is paced by the
> locus-coeruleus noradrenergic system. Whether the human ISFS changes with healthy aging, and whether
> mild cognitive impairment (MCI) adds a signal beyond that of age, is unknown. We quantified the ISFS
> during clean N2 sleep in 35 young adults, 39 healthy older adults, and 30 patients with amnestic MCI
> (aMCI), applying an established young-adult analysis of the sigma-envelope spectrum to high-density
> 256-channel EEG. The rhythm ran faster in both older groups than in young adults (p = 0.0026), with a
> non-significant tendency toward a broader spectral peak. Its overall strength was unchanged, but the
> central-parietal focus prominent in young adults was flattened in both older groups (cluster
> p = 0.023). Patients with aMCI were indistinguishable from healthy older adults on every measure, and
> no measure tracked cognitive score. The rhythm that paces spindle trains therefore becomes faster and
> less spatially focused with age, and does so whether or not cognition has begun to decline. The ISFS
> is a sensitive read-out of how aging reorganizes the machinery that generates spindles, rather than a
> marker of early cognitive impairment.

**After:**

> During non-rapid eye movement (NREM) sleep stage 2 (N2), the occurrence of sleep spindles in the
> sigma (13–16 Hz) band fluctuates following an infra-slow timescale of roughly 0.02 Hz (oscillation
> with period of ~50 sec). This infra-slow fluctuation of sigma power (ISFS) is associated with
> changes in locus-coeruleus noradrenergic (LC-NE) activity and other arousal systems and with related
> autonomic measures. In humans, the ISFS has been characterized in young adults. Here, we set out to
> investigate whether the ISFS changes with aging and mild cognitive impairment (MCI). To this end, we
> performed full-night polysomnography (PSG) including high-density (256-channel) EEG in 35 young
> adults (mean age 27.1), 39 healthy older adults (mean age 66.5), and 30 patients with amnestic MCI
> (aMCI) referred from a cognitive neurology clinic (mean age 67.8). We characterized the ISFS during
> N2 sleep by applying an established analysis pipeline to the EEG data, quantifying the peak
> frequency, bandwidth, and area under the spectral peak (AUC) of the sigma-envelope spectrum at every
> scalp EEG channel. We found that older age was associated with a significantly higher peak frequency
> (p = 0.0026); bandwidth showed a tendency toward broader peaks (p = 0.061), but this tracked how much
> N2 sleep each participant contributed rather than age. Overall AUC across all scalp channels was
> preserved, but the central-parietal ISFS hotspot, prominent in young adults, was significantly less
> focal in older age (cluster p = 0.023). We could not reveal significant differences in ISFS
> parameters between individuals with aMCI and healthy older adults, or correlations between ISFS
> parameters and cognitive status. Thus, the ISFS is reshaped by aging, becoming faster and losing its
> central-parietal focus, rather than by mild cognitive impairment.

### 5.1 Where this departs from his tracked text, and why

| # | His text | What went in | Why |
|---|---|---|---|
| 1 | `and with a trend towards broader bandwidth (p = 0.061)` | `bandwidth showed a tendency toward broader peaks (p = 0.061), but this tracked how much N2 sleep each participant contributed rather than age` | **C429, his own comment.** The ANCOVA takes the group effect from p = 0.061 to p = 0.206 with analyzed-N2 duration as covariate (r = +0.386, covariate p = 0.0002). Results 4.2 already says it "should not be read as an effect of age". His clause as written would contradict the Results chapter |
| 2 | `30 patients with amnestic MCI referred from…`, then `individuals with aMCI` later | `…amnestic MCI (aMCI) referred from…` | He uses `aMCI` later in the same paragraph without defining it. Abstracts are read standalone, so the definition lives here (abbreviation policy, triage §8.2.4) |
| 3 | `mean age = 27` / `= 66` / `= 67` | `mean age 27.1` / `66.5` / `67.8` | **His third figure is wrong.** The true means are 27.1 ± 4.3, 66.5 ± 9.7, 67.8 ± 9.1 (`demographics_V3/demographics_table.txt`); 67.8 rounds to 68, not 67. One decimal matches Methods 3.1 exactly and sidesteps the question. **Say the word and I will use bare integers 27 / 66 / 68 instead** — but not 67 |
| 4 | `significantly less focal (smeared) in older age` | `significantly less focal in older age` | `(smeared)` is a coinage; the recorded prose rules bar them. The substance — reframing the AUC result as loss of focus rather than focal reduction — is his and is kept |
| 5 | `(oscillation with period of ~50 sec)` | kept verbatim | Redundant with "roughly 0.02 Hz" in the same clause, but it is his phrasing and he wrote it deliberately |
| 6 | closing sentence deleted | deleted | See the flag below |

### 5.2 ⚠ One thing to be aware of before you approve

His rewrite **deletes the closing sentence** —

> *"The ISFS is a sensitive read-out of how aging reorganizes the machinery that generates spindles,
> rather than a marker of early cognitive impairment."*

— and replaces the ending with `Thus, the ISFS is reshaped by aging … rather than by mild cognitive
impairment.` That deleted sentence is the locked closing of the project's four-sentence scientific
story, and it is the one line that says what the measure is *for*. Taking his version is what you
chose, and his ending does carry the same aging-not-MCI verdict — but the "sensitive read-out" claim
is now gone from the Abstract entirely. **Worth one line in the reply to him.**

Also note the V2 phrase `The rhythm that paces spindle trains` disappears with the old ending, which
is tidy: he objected to `paced by` in the opening sentence, and this was the second instance of the
same verb.

---

## §6 — Hebrew abstract, תקציר (his email #2) ← **CHECK THIS**

**Before:** absent. There is no Hebrew prose anywhere in the repo apart from one clinical scoring note
about an excluded subject.

**Placement:** immediately after the English Abstract, before the Introduction — Yael's p. 7.

**Heading:** `תקציר`, bold centred body text (not HEADING_2, so it stays out of the TOC — see §4).

**Proposed translation of the final §5 abstract:**

> **תקציר**
>
> במהלך שלב 2 של שנת ללא תנועות עיניים מהירות (NREM), הופעתם של כישורי שינה בפס הסיגמא (13–16 הרץ)
> משתנה במחזוריות אינפרה-איטית של כ-0.02 הרץ (תנודה בעלת מחזור של כ-50 שניות). תנודה אינפרה-איטית זו
> של עוצמת הסיגמא (ISFS) קשורה לשינויים בפעילות המערכת הנוראדרנרגית של הלוקוס קוארולאוס (LC-NE)
> ובמערכות עוררות נוספות, וכן למדדים אוטונומיים נלווים. בבני אדם, ה-ISFS אופיין עד כה בצעירים. במחקר
> הנוכחי ביקשנו לבחון האם ה-ISFS משתנה עם ההזדקנות ועם ליקוי קוגניטיבי קל (MCI). לשם כך ביצענו
> פוליסומנוגרפיה (PSG) לאורך לילה שלם, הכוללת אלקטרואנצפלוגרפיה (EEG) בצפיפות גבוהה (256 ערוצים),
> ב-35 צעירים בריאים (גיל ממוצע 27.1), 39 מבוגרים בריאים (גיל ממוצע 66.5) ו-30 מטופלים עם ליקוי
> קוגניטיבי קל אמנסטי (aMCI) שהופנו ממרפאה לנוירולוגיה קוגניטיבית (גיל ממוצע 67.8). איפיינו את ה-ISFS
> במהלך שנת N2 באמצעות יישום שיטת ניתוח מקובלת על נתוני ה-EEG, וכימתנו את תדירות השיא, רוחב הפס והשטח
> שמתחת לשיא הספקטרלי (AUC) של ספקטרום מעטפת הסיגמא בכל ערוץ EEG על הקרקפת. מצאנו כי גיל מבוגר יותר
> היה קשור לתדירות שיא גבוהה יותר באופן מובהק (p = 0.0026); רוחב הפס הראה מגמה להתרחבות (p = 0.061),
> אך מגמה זו שיקפה את כמות שנת ה-N2 שתרם כל משתתף ולא את הגיל. ה-AUC הכולל בכל ערוצי הקרקפת נשמר, אך
> מוקד ה-ISFS המרכזי-פריאטלי, הבולט בצעירים, היה ממוקד פחות באופן מובהק בגיל מבוגר (p = 0.023
> לאשכול). לא נמצאו הבדלים מובהקים בפרמטרים של ה-ISFS בין אנשים עם aMCI לבין מבוגרים בריאים, ולא
> נמצאו מתאמים בין פרמטרי ה-ISFS למצב הקוגניטיבי. לפיכך, ה-ISFS מעוצב מחדש על ידי ההזדקנות — נעשה
> מהיר יותר ומאבד את מיקודו המרכזי-פריאטלי — ולא על ידי ליקוי קוגניטיבי קל.

**Translation choices you may want to overrule:**

| English | Hebrew used | Alternative |
|---|---|---|
| sleep spindles | `כישורי שינה` | Yael's term; standard |
| infra-slow | `אינפרה-איטית` | transliterated; there is no settled Hebrew term |
| amnestic MCI | `ליקוי קוגניטיבי קל אמנסטי` | Yael wrote `הפרעה קוגנטיבית אמנסטית מתונה` (with two non-standard spellings, not copied) |
| analysis pipeline | `שיטת ניתוח מקובלת` | literal `צנרת ניתוח` is not idiomatic Hebrew |
| young adults / healthy older adults | `צעירים בריאים` / `מבוגרים בריאים` | avoids the `מבוגרים צעירים` vs `מבוגרים` collision |
| cluster p | `p = 0.023 לאשכול` | |

All Latin acronyms (ISFS, AUC, EEG, PSG, aMCI, LC-NE, NREM, N2) kept in Latin script, as Yael's
תקציר does. Numbers and `p =` values stay LTR inside the RTL paragraph — Docs handles this via the
bidi algorithm once `direction = RIGHT_TO_LEFT` is set on the paragraph.

---

## §7 — Resulting document order

```
p1  English title page          (edited, §1)
p2  Hebrew title page           (new, §2)
p3  Acknowledgements            (new, §3)
p4  Table of Contents           (new page, §4 — you generate the list itself)
p5  Abstract                    (replaced, §5)
p6  תקציר                        (new, §6)
p7+ Introduction … References   (untouched)
```

---

## §8 — Markdown mirrors

| File | Change |
|---|---|
| `thesis/chapters/00_front_matter.md` | Replace the 2026-05-27 placeholder wholesale — it predates the real title page and contradicts it. New content: both title pages, the Acknowledgements, and a note on how the TOC is produced |
| `thesis/chapters/01_abstract.md` | New body + a `v4 (2026-08-17)` provenance note; `## תקציר` added below it |
| `thesis/reviews/yuval_review_triage.md` | Status row → **APPLIED**; §8 carry-over updated for the reply session |

---

## §9 — Spotted, not done

1. **Student ID on the title page.** Yael's page 1 carries hers under her name; ours has none.
   Declined.
2. **Ethics / informed-consent / funding / COI statements** — finding S4, `final_check_report.md:106`.
   Still entirely absent from a two-site human study. Declined.
3. **He has never seen the current title.** His copy carried the V1 title and he did not comment on
   it. Already logged for the reply session (triage §8.3.6); untouched here.
4. **The Sydney site is unacknowledged** — see the note in §3.
5. **`06_conclusion.md`, `07_references.md` and `08_appendix.md` are empty stubs** in
   `thesis/chapters/`, and the Doc has no Conclusion or Appendix. His model TOC has neither, so this
   changes nothing for the TOC, but the chapter files are misleading as a "source of truth".

---

## §10 — Order of application

Applied bottom-up so Docs indices stayed valid: Discussion subheadings → §6 תקציר → §5 Abstract →
§2+§3+§4 block → §1 title page → markdown mirrors.

---

## §11 — Closing state, applied 2026-08-17

### 11.1 Shaked's amendments to the proposal

| § | Proposal | What went in |
|---|---|---|
| 3 | Tel Aviv names only | **Sydney and Zurich added**: Angela D'Rozario and Rick Wassing for the CIRUS recordings, and **Maria E. Dimitriades for the analysis pipeline and her personal help** |
| 4 | one bare `Discussion` row in the TOC | **Seven Discussion subheadings added, 5.1–5.7** (new work, see 11.2) |
| 5 | `…characterized in young adults.` | `…characterized mainly in young adults, with isolated reports in clinical populations and in early childhood.` — all cited in Discussion 5.4 |
| 5 | `aging and mild cognitive impairment (MCI)`, then `amnestic MCI (aMCI)` defined later | `aging and amnestic mild cognitive impairment (aMCI)` at first use; the later re-definition removed; closing sentence now reads `rather than by amnestic mild cognitive impairment` |
| 6 | `שנת ללא תנועות עיניים מהירות (NREM)` | `שנת NREM` — untranslated, as in Yael's תקציר |
| 6 | `פס הסיגמא` | `תחום הסיגמא` |
| 6 | `הלוקוס קוארולאוס` | `הלוקוס סרולאוס` |
| 6 | — | all four §5 amendments mirrored into the Hebrew |

Everything else went in as proposed. The `mean age 27.1 / 66.5 / 67.8` form was kept, `(smeared)`
stayed out, and his deletion of the "sensitive read-out" closing sentence stands.

### 11.2 Discussion subheadings (new — not part of the original proposal)

Shaked asked for these so the TOC shows more than one row for the largest chapter. Topic-style,
matching the model Yuval pasted in; numbered 5.x to match 2.x/3.x/4.x elsewhere. **No prose was
rewritten** — every heading falls on an existing paragraph break, and the opening recap and summary
of findings stay unheaded.

| Heading | Covers |
|---|---|
| *(unheaded)* | ¶1 aim recap, ¶2 summary of findings |
| **5.1 Two changes with age, and what they imply** | ¶3 |
| **5.2 A noradrenergic account** | ¶4–¶8 |
| **5.3 No evidence for an effect of amnestic MCI beyond age** | ¶9–¶10 |
| **5.4 The ISFS in other populations** | ¶11–¶12 |
| **5.5 The ISFS among other sleep changes, and the cyclic alternating pattern** | ¶13–¶14 |
| **5.6 Limitations and future directions** | ¶15–¶16 |
| **5.7 Conclusion** | ¶17 |

### 11.3 Verification run against the Doc

| Check | Result |
|---|---|
| Page order: English title page → Hebrew title page → Acknowledgements → Table of Contents → Abstract → תקציר → Introduction | ✅ |
| Heading tree: 2.1–2.5, 3.1–3.7, 4.1–4.6, **5.1–5.7**, Supplementary figures, References | ✅ |
| Reference entries | **63 → 63**, unchanged |
| Reference invariant (n-th distinct superscript top-to-bottom == n) | ✅ holds for all 63 |
| Body-paragraph diff vs the pre-edit snapshot | **25 added, 3 removed**, all intended; nothing else moved |
| Hebrew paragraphs carry `direction: RIGHT_TO_LEFT` | ✅ all 14 title-page lines + heading + תקציר body |
| Title-page run styles preserved | ✅ 14 pt on all three institution lines, 23 pt bold title |

The three removed paragraphs are exactly `Department of Physiology and Pharmacology (Medicine)`,
`July, 2026`, and the old abstract.

### 11.4 Still to do by hand

**Insert the table of contents.** Open the Doc, click into the blank line under the
`Table of Contents` heading on page 4, then Insert → Table of contents → **with page numbers**.
Nothing else is outstanding.

### 11.5 Noticed while applying, not fixed

**Two paragraphs on the English title page were already `direction: RIGHT_TO_LEFT` before this pass**
— *"The thesis was carried out under the supervision of"* and the blank line after it. Pre-existing,
not caused by these edits, and invisible because both are centred. Left alone; a one-click fix if it
ever matters.
