# Cohort summary — Phase 0.3 output

_Pulled 2026-05-27 from Google Sheet `1bGjm-GKiwBQT3QwvHM5JGbjI4kRLmkH5i_RfJ0I3SWw` (sheets: `subjects`, `excluded`, `group_summary`). This is the **source of truth** — supersedes `subject_summary.md`, `new_subjects_isfs_summary.md`, `overview/group_summary.csv` for any thesis-writing purpose._

---

## Table 1 — Participant demographics (draft for Methods §3.1)

| Group | Label | N | Age (mean ± SD) | Age range | Sex (F / M / missing) | MoCA (mean ± SD, n) |
|-------|-------|---|------------------|-----------|------------------------|----------------------|
| Young controls | YA | 34 | 27.7 ± 5.1 | 21–42 | 19 / 15 / 0 | _not collected_ |
| Elderly controls | HE | 38 | 66.5 ± 9.9 | 49–82 | 23 / 15 / 0 | 27.2 ± 2.6 (n=29) |
| MCI patients | MCI | 31 | 67.8 ± 9.0 | 46–80 | 18 / 12 / 1 | 21.0 ± 4.1 (n=14) |

**Notes for prose:**
- Total N = 103.
- Age difference between HE and MCI is statistically negligible (66.5 vs 67.8) — confirm with t-test in Results.
- Sex distribution is roughly balanced and similar across groups (~40% M / ~60% F).
- MoCA available only for a subset of HE and MCI (n=29, n=14 respectively, after 11 elderly MoCA scores were added 2026-06-15). Report mean ± SD with sample size in caption. Use MoCA only where it's available (e.g. for the optional MoCA-correlation analysis).
- Sex is missing for 1 MCI subject (KS5).

---

## Sleep architecture summary (for Methods §3.2 + Results §4.1)

| Group | Sleep efficiency % | TST (sec) | % N1 | % N2 | % N3 | % REM | % Wake |
|-------|--------------------|-----------|------|------|------|-------|--------|
| YA | 87.6 ± 10.7 | 22,904 ± 3,142 | 3.9 ± 2.8 | 38.3 ± 13.6 | 31.0 ± 9.3 | 26.8 ± 11.9 | 11.8 ± 10.6 |
| HE | 83.8 ± 8.5 | 22,288 ± 3,130 | 11.5 ± 7.8 | 53.3 ± 12.2 | 21.7 ± 11.5 | 13.6 ± 5.8 | 15.0 ± 7.4 |
| MCI | 76.1 ± 13.9 | 20,457 ± 5,044 | 14.1 ± 7.7 | 48.8 ± 12.0 | 24.4 ± 11.5 | 12.7 ± 6.0 | 23.6 ± 13.7 |

**Notable patterns:**
- **% N1 rises sharply with age** (3.9% → 11.5% → 14.1%) — typical age-related fragmentation.
- **% N2 rises in HE+MCI** relative to YA (38% → 53% → 49%).
- **% N3 and % REM both drop with age**.
- **Sleep efficiency drops across groups** (87.6 → 83.8 → 76.1%) — MCI cohort shows the most disrupted sleep.

---

## N2 bout availability (for Methods §3.6)

| Group | N2 duration (sec, mean ± SD) | N2 bouts ≥ 300 s (mean ± SD, range) |
|-------|------------------------------|--------------------------------------|
| YA | 8,772 ± 3,250 | 8.4 ± 3.6 (3–16) |
| HE | 11,847 ± 3,035 | 11.9 ± 3.7 (3–18) |
| MCI | 9,995 ± 3,796 | 11.4 ± 5.1 (2–22) |

Sufficient long-bout coverage in all three groups; minimum subject still passes the 300 s minimum-bout-duration criterion.

---

## Bad-channel and bad-epoch summary (for Methods §3.4)

| Group | Bad channels (n) | Bad channels % | Bad epochs in N2 (sec) | Bad epochs in N2 % |
|-------|-------------------|----------------|------------------------|---------------------|
| YA | 11.0 ± 6.2 | 6.3 ± 3.5 | 377 ± 405 | 4.6 ± 5.2 |
| HE | 2.5 ± 3.1 | 1.4 ± 1.8 | 443 ± 617 | 3.4 ± 4.0 |
| MCI | 5.9 ± 7.7 (n=27) | 3.4 ± 4.4 | 210 ± 284 | 1.8 ± 2.0 |

**Methods caveat:** YA group has notably more bad channels on average than HE/MCI — likely reflects the EGI cap fit on younger subjects or a different recording site. Worth noting in Methods/Limitations. (4 MCI subjects don't have a `bad_channels.txt` file: SM07, SM13, SM14, VZ9 — flag if any of these end up in main figures.)

---

## Exclusions (for Methods §3.1 + Appendix)

| Group | N excluded | Top exclusion reasons |
|-------|-----------|-----------------------|
| YA | 11 | Insufficient N2 (4), too many bad channels (4), sweat artifact (1), only 1–2 long bouts (3) |
| HE | 5 | Insufficient N2 (4), pre-referenced data (1) |
| MCI | 12 | Insufficient N2 (3), bad-channel count too high (3 subjects with 42–61 bad channels), all-bad epochs (1), unclear diagnosis (1), other (4) |
| **MCI → AD-tagged** | 4 | Subjects later identified as full Alzheimer's (AB7, AY1, SK6, YG9) — excluded from MCI group |

Full exclusion list with per-subject reasons is in the Sheet's `excluded` tab. Recommend an appendix table reproducing this.

**Note for Methods prose:** the 4 AD-tagged subjects are excluded from "MCI" — be explicit that the MCI group refers to patients who had not progressed to clinical AD diagnosis at the time of recording.

---

## Sheet quirks flagged for user attention

1. **HE-group subjects with `MCI` prefix in their IDs**: e.g. `MCI20`, `MCI25`, `MCI27`, ... through `MCI43` — sit under `group=HE`, `group_dir=elderly_control_clean`. These were apparently initially recruited as MCI but reclassified as elderly controls. Worth a note in Appendix A. The naming convention is misleading but the group label is what matters for the analysis.
2. **Hebrew note** on excluded subject HR72: "הקלטה חרדות עצרתי באמצע" ≈ "recording — anxieties — I stopped midway."
3. **Subject KS5** (MCI) has empty sex field.
4. **4 MCI subjects** (SM07, SM13, SM14, VZ9) have `has_bad_channels_file = FALSE` — pipeline ran without an explicit bad-channels list. Verify whether this is intentional before drafting Methods.

---

## What's NOT in the Sheet (would need to fall back to local files)

The Sheet covers demographics, sleep architecture, N2 bout counts, bad-channels/epochs. It does **not** contain:
- ISFS detection rates per group (in `new_subjects_isfs_summary.md`)
- ISFS parameter values (peak frequency, bandwidth, AUC, peak power) per subject — these come from `results/*_V{max}/` CSVs
- ROI-level statistics

For the Methods chapter, the Sheet is sufficient for §3.1 (Participants) and §3.4 (Bad channels/epochs). Results chapter will need the ISFS outputs from the latest V-numbered results directories.
