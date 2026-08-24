# Front matter

> _v2 (2026-08-17). **Replaces the 2026-05-27 placeholder**, which predated the real title page and
> contradicted it. Written in the front-matter pass that closed Yuval's email items #1 (TOC) and #2
> (Hebrew abstract) and his tracked title-page correction. Record:
> `thesis/reviews/front_matter_edits_before_after.md`. Document order in the Doc is now: English title
> page, Hebrew title page, Acknowledgements, Table of Contents, Abstract, תקציר, Introduction._

## Title page

Centred, in this order:

```
Sagol School of Neuroscience
Department of Neuroscience and Brain Disorders
Gray Faculty of Medical and Health Sciences

Aging, but not amnestic mild cognitive impairment, reshapes the infra-slow rhythm of
sleep spindle power

By

Shaked Aglamaz

The thesis was carried out under the supervision of

Professor Yuval Nir

August, 2026
```

The department and faculty lines are Yuval's tracked correction, applied verbatim (his docx paras
003–004): the old line read `Department of Physiology and Pharmacology (Medicine)`, and the faculty
line is a new third line, not a replacement. The date was `July, 2026`, set to **August, 2026**.

**No student ID line.** The TAU template carries one under the author's name; declined for now.

## Hebrew title page

Page 2, RTL, centred — mirrors the English page, following the TAU house format. Institutional names
verified against TAU's current Hebrew site, not translated by ear.

```
בית הספר סגול למדעי המוח
החוג למדעי העצב ומחלות נוירולוגיות
הפקולטה למדעי הרפואה והבריאות ע"ש גריי

הזדקנות, אך לא ליקוי קוגניטיבי קל אמנסטי, מעצבת מחדש את המקצב האינפרה-איטי של עוצמת כישורי השינה

מאת

שקד אגלמז

החיבור בוצע בהנחייתו של

פרופסור יובל ניר

אוגוסט, 2026
```

Note: TAU's official Hebrew for the department back-translates as *Neuroscience and Neurological
Diseases*, not *Brain Disorders*. It is the same department (head: Prof. Eran Perlson, whose older
English listings still read *Department of Physiology and Pharmacology* — the rename Yuval's edit
reflects).

## Acknowledgements

Page 3. Built out from Yuval's skeletal stub (docx paras 016–019), keeping all five of the names he
supplied in the roles he assigned them, plus the Sydney site and Maria Dimitriades.

> I would like to thank Prof. Yuval Nir for his guidance, his scientific standards, and for trusting
> me with this project; Noa Bregman for professional support at the Tel Aviv Sourasky Medical Center,
> and for referring patients to the study; Rivi Tauman and Jenny Zitser for their guidance in sleep
> medicine and PSG monitoring; Maria E. Dimitriades, whose analysis pipeline this work is built on,
> for sharing it and for her help throughout; Angela D'Rozario and Rick Wassing for the recordings
> made at the CIRUS Centre in Sydney; Rotem Falach and Flavio Schmidig for teaching me EEG analysis
> and for their advice along the way; and the members of the Nir lab for their help and good company.
> Finally, I thank the participants and their families for their time and willingness to take part.

**No funding statement** — nothing in the project records a funder or grant number.
No ethics, consent or conflict-of-interest statements either; that is finding S4 in
`thesis/final_check_report.md`, still open by decision.

## Table of contents

Page 4, generated natively by Google Docs (Insert → Table of contents, with page numbers) from the
document's own HEADING_2 / HEADING_3 structure. **The Docs API cannot create a TOC** — there is no
request type for it — so this is the one step done by hand in the browser.

`Acknowledgements`, `Table of Contents` and `תקציר` are bold centred body text rather than HEADING_2,
so the generated list starts at `Abstract`, matching the model Yuval pasted in.

Yuval's pasted example TOC is Yael Gat's contents page, not ours; it also carries a stray
one-character row `T`, mis-levels `Sleep in MCI`, and leaves `Supplementary` / `References` unstyled.
His `#adapt to yours#` note covers all of it.

## Abstract and תקציר

Both live in `01_abstract.md`. In the Doc they sit after the TOC, English first, each on its own page.

## List of figures / List of tables / Abbreviations

Not present, and not requested. The locked display set is Table 1, Table 2, Figures 1–5 and Figures
S1–S3 (`thesis/figure_manifest.md`).
