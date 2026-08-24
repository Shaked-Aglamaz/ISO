"""
C429 — does the amount of analyzed N2 sleep explain the group differences?

Yuval asked (comment C429) how we know the ISFS group differences do not reflect
differing amounts of N2 sleep, and whether N2 amount can be entered as a factor.
This script answers the second half: an ANCOVA on the three whole-scalp ISFS
parameters (peak frequency, bandwidth, AUC) with each subject's total analyzed
N2 bout duration as covariate.

Per-subject values are produced by exactly the same two calls that
step6_groups_comparison.run_three_group_comparison makes, so the unadjusted
block reproduces results/group_comparison_results/three_groups_V10 line for line
and the ANCOVA can be read as a delta against it.

Covariate: total_dur_min from results/demographics_V3/n2_bouts_per_subject.csv
(total duration of the clean N2 bouts that actually entered the ISFS analysis).

Run from repo root with the venv active:

    PYTHONIOENCODING=utf-8 python code/ancova_n2_duration.py
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

from utils.config import BASE_DIR
from utils.utils import get_all_subjects

from step4_distribution_analysis import (
    filter_subjects_by_detection_rate,
    load_and_process_all_channels_data,
)
from step6_groups_comparison import (
    test_normality_shapiro,
    test_omnibus,
    test_posthoc,
)


OUTPUT_DIR = Path("results/group_comparison_results/three_groups_V11")
COVARIATE_CSV = Path("results/demographics_V3/n2_bouts_per_subject.csv")

# (display name, data dir under BASE_DIR, ISFS results dir) — same as step6.
GROUPS = [
    ("Young",   "control_clean",         Path("results/sigma_fix_YA")),
    ("Elderly", "elderly_control_clean", Path("results/sigma_fix_HE")),
    ("MCI",     "MCI_clean",             Path("results/sigma_fix_MCI")),
]

METRICS = {
    "peak_frequency": {"name": "Peak Frequency", "unit": "Hz"},
    "bandwidth":      {"name": "Bandwidth",      "unit": "Hz"},
    "auc":            {"name": "Area Under Curve", "unit": "AU"},
}

COVARIATE_LABEL = "Total analyzed N2 bout duration (min)"


def load_group_data() -> tuple[dict, pd.DataFrame]:
    """Load per-subject whole-scalp ISFS values exactly as step6 does.

    Returns (group_data, long_df) where group_data is the {name: {'data': df}}
    structure the step6 test functions expect, and long_df is the same values
    stacked with a 'group' column.
    """
    group_data: dict = {}
    frames = []

    for name, group_dir, results_dir in GROUPS:
        buf = io.StringIO()
        with redirect_stdout(buf):
            subjects = get_all_subjects(f"{BASE_DIR}/{group_dir}/")
            subjects = [s for s in subjects
                        if (results_dir / s).exists() and s != "dashboards"]
            subjects, _ = filter_subjects_by_detection_rate(subjects, dir_path=results_dir)
            subject_averages, total_channels, n_subjects = \
                load_and_process_all_channels_data(subjects, results_dir)

        group_data[name] = {
            "data": subject_averages,
            "n_subjects": n_subjects,
            "total_channels": total_channels,
        }
        df = subject_averages.copy()
        df["group"] = name
        frames.append(df)
        print(f"  {name:<8} N={n_subjects:3d}  ({total_channels} channel rows)")

    long_df = pd.concat(frames, ignore_index=True)
    return group_data, long_df


def attach_covariate(long_df: pd.DataFrame) -> pd.DataFrame:
    """Merge total analyzed N2 bout duration onto the per-subject ISFS values."""
    if not COVARIATE_CSV.exists():
        raise FileNotFoundError(f"Covariate source not found: {COVARIATE_CSV}")

    cov = pd.read_csv(COVARIATE_CSV)[["subject_id", "group", "total_dur_min"]]
    cov = cov.rename(columns={"group": "group_short"})

    merged = long_df.merge(cov, left_on="subject", right_on="subject_id", how="left")

    missing = merged.loc[merged["total_dur_min"].isna(), "subject"].tolist()
    if missing:
        raise RuntimeError(
            f"{len(missing)} subject(s) have no covariate value in {COVARIATE_CSV}: {missing}"
        )
    if len(merged) != len(long_df):
        raise RuntimeError(
            f"Covariate merge changed row count: {len(long_df)} -> {len(merged)}"
        )

    # Guard against a group-label mismatch between the two sources.
    short = {"Young": "YA", "Elderly": "HE", "MCI": "MCI"}
    bad = merged.loc[merged["group"].map(short) != merged["group_short"], "subject"].tolist()
    if bad:
        raise RuntimeError(f"Group label mismatch between ISFS results and sheet: {bad}")

    merged = merged.drop(columns=["subject_id", "group_short"])
    # Mean-centred covariate: adjusted means then sit at the cohort grand mean.
    merged["n2_min_c"] = merged["total_dur_min"] - merged["total_dur_min"].mean()
    return merged


def _partial_eta_sq(anova_tbl: pd.DataFrame, term: str) -> float:
    ss = float(anova_tbl.loc[term, "sum_sq"])
    ss_res = float(anova_tbl.loc["Residual", "sum_sq"])
    return ss / (ss + ss_res) if (ss + ss_res) > 0 else float("nan")


def _effect_label(eta2: float) -> str:
    if not np.isfinite(eta2):
        return ""
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


def run_ancova(df: pd.DataFrame, metric: str) -> dict:
    """ANCOVA: metric ~ group + centred analyzed-N2-duration."""
    d = df[["group", metric, "n2_min_c", "total_dur_min"]].dropna().copy()
    d = d.rename(columns={metric: "y"})

    model = smf.ols("y ~ C(group) + n2_min_c", data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)

    # Adjusted (marginal) group means at the grand-mean covariate (n2_min_c = 0).
    adjusted = {}
    for g in d["group"].unique():
        pred = model.get_prediction(pd.DataFrame({"group": [g], "n2_min_c": [0.0]}))
        frame = pred.summary_frame(alpha=0.05)
        adjusted[g] = {
            "mean": float(frame["mean"].iloc[0]),
            "se": float(frame["mean_se"].iloc[0]),
            "ci_low": float(frame["mean_ci_lower"].iloc[0]),
            "ci_upp": float(frame["mean_ci_upper"].iloc[0]),
        }

    # Holm-corrected pairwise contrasts of the adjusted means. Tukey HSD on
    # adjusted means would need emmeans, which is not available here.
    pairwise = model.t_test_pairwise("C(group)", method="holm").result_frame

    # Assumption checks.
    inter = smf.ols("y ~ C(group) * n2_min_c", data=d).fit()
    inter_tbl = sm.stats.anova_lm(inter, typ=2)
    slope_f = float(inter_tbl.loc["C(group):n2_min_c", "F"])
    slope_p = float(inter_tbl.loc["C(group):n2_min_c", "PR(>F)"])
    resid_w, resid_p = shapiro(model.resid)

    return {
        "n": int(len(d)),
        "model": model,
        "table": table,
        "group_f": float(table.loc["C(group)", "F"]),
        "group_p": float(table.loc["C(group)", "PR(>F)"]),
        "group_eta2": _partial_eta_sq(table, "C(group)"),
        "cov_f": float(table.loc["n2_min_c", "F"]),
        "cov_p": float(table.loc["n2_min_c", "PR(>F)"]),
        "cov_eta2": _partial_eta_sq(table, "n2_min_c"),
        "cov_slope": float(model.params["n2_min_c"]),
        "df_resid": int(model.df_resid),
        "raw_means": d.groupby("group")["y"].agg(["mean", "std", "size"]).to_dict("index"),
        "adjusted": adjusted,
        "pairwise": pairwise,
        "slope_f": slope_f,
        "slope_p": slope_p,
        "resid_shapiro_w": float(resid_w),
        "resid_shapiro_p": float(resid_p),
        "corr": float(np.corrcoef(d["y"], d["total_dur_min"])[0, 1]),
    }


def write_report(path: Path, df: pd.DataFrame, unadjusted: dict, ancova: dict) -> None:
    L: list[str] = []
    A = L.append

    A("")
    A("#" * 80)
    A("  THREE-GROUP ANCOVA — ISFS PARAMETERS WITH ANALYZED N2 DURATION AS COVARIATE")
    A("#" * 80)
    A("")
    A("Reviewer comment C429: are the group differences explained by differing")
    A("amounts of N2 sleep? Here the amount is entered as a covariate.")
    A("")
    A(f"Covariate: {COVARIATE_LABEL}")
    A(f"  source: {COVARIATE_CSV} (column 'total_dur_min')")
    A("  mean-centred, so adjusted means are evaluated at the cohort grand mean")
    A(f"  grand mean = {df['total_dur_min'].mean():.2f} min")
    A("")
    A("Per-subject ISFS values: mean across all channels (channels without a")
    A("detected ISFS are skipped by the mean), identical to the raw whole-scalp")
    A("values in three_groups_V10/three_group_statistics.txt.")
    A("")
    A("Post-hoc note: unadjusted post-hoc is Tukey HSD (as in V10); the ANCOVA")
    A("post-hoc is Holm-corrected pairwise contrasts of the adjusted means,")
    A("because Tukey HSD on adjusted means requires emmeans, unavailable here.")
    A("")

    # --- covariate description -------------------------------------------------
    A("=" * 80)
    A("COVARIATE DISTRIBUTION")
    A("=" * 80)
    A("")
    for g in ("Young", "Elderly", "MCI"):
        s = df.loc[df["group"] == g, "total_dur_min"]
        A(f"  {g:<10}: N={len(s):3d}, mean={s.mean():7.2f} ± {s.std():6.2f} min "
          f"(median={s.median():.2f})")
    A("")
    A("  Correlation of the covariate with each ISFS parameter (all subjects):")
    for metric, info in METRICS.items():
        A(f"    {info['name']:<20}: r = {ancova[metric]['corr']:+.3f}")
    A("")

    # --- per metric ------------------------------------------------------------
    for metric, info in METRICS.items():
        u = unadjusted[metric]
        a = ancova[metric]

        A("=" * 80)
        A(f"{info['name'].upper()} ({info['unit']})")
        A("=" * 80)
        A("")

        A("--- 1. UNADJUSTED (reproduces three_groups_V10) ---")
        A("")
        for g in ("Young", "Elderly", "MCI"):
            rm = a["raw_means"][g]
            A(f"  {g:<10}: N={int(rm['size']):3d}, μ={rm['mean']:.4f} ± {rm['std']:.4f}")
        A("")
        A(f"  {u['omnibus']['test_used']}: "
          f"{'F' if u['omnibus']['test_used'] == 'One-Way ANOVA' else 'H'} = "
          f"{u['omnibus']['statistic']:.4f}, p = {u['omnibus']['p_value']:.4f} "
          f"[{'SIGNIFICANT' if u['omnibus']['significant'] else 'NOT SIGNIFICANT'}]")
        A(f"  eta^2 = {u['omnibus']['eta_squared']:.4f} "
          f"({_effect_label(u['omnibus']['eta_squared'])} effect)")
        if u["posthoc"].get("pairs"):
            A(f"  Post-hoc ({u['posthoc']['test_used']}):")
            for (g1, g2), pdata in u["posthoc"]["pairs"].items():
                tag = "*" if pdata["significant"] else "ns"
                A(f"    {g1} vs {g2}: p = {pdata['p_value']:.4f} ({tag})")
        else:
            A("  Post-hoc: not run (omnibus ns)")
        A("")

        A("--- 2. ANCOVA: value ~ group + analyzed N2 duration ---")
        A("")
        A(f"  N = {a['n']}, residual df = {a['df_resid']}")
        A("")
        A(f"  Group (adjusted for covariate): F = {a['group_f']:.4f}, "
          f"p = {a['group_p']:.4f} "
          f"[{'SIGNIFICANT' if a['group_p'] < 0.05 else 'NOT SIGNIFICANT'}]")
        A(f"    partial eta^2 = {a['group_eta2']:.4f} ({_effect_label(a['group_eta2'])} effect)")
        A(f"  Covariate (analyzed N2 min): F = {a['cov_f']:.4f}, p = {a['cov_p']:.4f} "
          f"[{'SIGNIFICANT' if a['cov_p'] < 0.05 else 'NOT SIGNIFICANT'}]")
        A(f"    partial eta^2 = {a['cov_eta2']:.4f} ({_effect_label(a['cov_eta2'])} effect)")
        A(f"    slope = {a['cov_slope']:+.6g} {info['unit']} per minute of analyzed N2")
        A("")
        A("  Type-II ANOVA table:")
        for line in a["table"].to_string().split("\n"):
            A(f"    {line}")
        A("")

        A("--- 3. RAW vs ADJUSTED GROUP MEANS ---")
        A("")
        A(f"  {'Group':<10} {'raw mean':>12} {'adjusted mean':>16} {'SE':>10} "
          f"{'95% CI':>26}")
        for g in ("Young", "Elderly", "MCI"):
            rm = a["raw_means"][g]
            ad = a["adjusted"][g]
            A(f"  {g:<10} {rm['mean']:>12.4f} {ad['mean']:>16.4f} {ad['se']:>10.4f} "
              f"  [{ad['ci_low']:.4f}, {ad['ci_upp']:.4f}]")
        A("")

        A("--- 4. POST-HOC ON ADJUSTED MEANS (Holm-corrected) ---")
        A("")
        for line in a["pairwise"].to_string().split("\n"):
            A(f"    {line}")
        A("")

        A("--- 5. ASSUMPTION CHECKS ---")
        A("")
        A(f"  Homogeneity of regression slopes (group x covariate interaction):")
        A(f"    F = {a['slope_f']:.4f}, p = {a['slope_p']:.4f} "
          f"[{'VIOLATED — slopes differ' if a['slope_p'] < 0.05 else 'OK — slopes comparable'}]")
        A(f"  Shapiro-Wilk on model residuals:")
        A(f"    W = {a['resid_shapiro_w']:.4f}, p = {a['resid_shapiro_p']:.4f} "
          f"[{'NOT NORMAL' if a['resid_shapiro_p'] < 0.05 else 'NORMAL'}]")
        if metric == "auc":
            A("")
            A("  CAVEAT: AUC failed the per-group normality check in V10 (MCI Shapiro")
            A("  p = 0.0412) and the unadjusted omnibus there is Kruskal-Wallis, which")
            A("  remains the primary test for this parameter. ANCOVA is parametric, so")
            A("  the block above is a covariate-adjusted supplement, not a replacement.")
        A("")

    # --- summary ---------------------------------------------------------------
    A("#" * 80)
    A("  SUMMARY — effect of adding analyzed N2 duration as a covariate")
    A("#" * 80)
    A("")
    A(f"  {'Parameter':<22} {'unadjusted p':>14} {'ANCOVA group p':>16} "
      f"{'covariate p':>13}  verdict")
    for metric, info in METRICS.items():
        u = unadjusted[metric]
        a = ancova[metric]
        before = u["omnibus"]["p_value"] < 0.05
        after = a["group_p"] < 0.05
        if before and after:
            verdict = "group effect survives"
        elif before and not after:
            verdict = "group effect LOST after adjustment"
        elif not before and after:
            verdict = "group effect APPEARS after adjustment"
        else:
            verdict = "ns before and after"
        A(f"  {info['name']:<22} {u['omnibus']['p_value']:>14.4f} {a['group_p']:>16.4f} "
          f"{a['cov_p']:>13.4f}  {verdict}")
    A("")

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    print("Loading per-subject whole-scalp ISFS values (same calls as step6)...")
    group_data, long_df = load_group_data()

    df = attach_covariate(long_df)
    print(f"\nCovariate merged for {len(df)}/{len(long_df)} subjects.")
    print("Covariate group means (min): " + ", ".join(
        f"{g}={df.loc[df['group'] == g, 'total_dur_min'].mean():.1f}"
        for g in ("Young", "Elderly", "MCI")))

    # Unadjusted baseline, computed by the same step6 functions that produced V10.
    buf = io.StringIO()
    with redirect_stdout(buf):
        normality = test_normality_shapiro(group_data, METRICS)
        omnibus = test_omnibus(group_data, METRICS, normality)
        posthoc = test_posthoc(group_data, METRICS, omnibus)
    unadjusted = {m: {"normality": normality[m], "omnibus": omnibus[m], "posthoc": posthoc[m]}
                  for m in METRICS}

    ancova = {m: run_ancova(df, m) for m in METRICS}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "three_group_ancova_statistics.txt"
    write_report(report_path, df, unadjusted, ancova)

    csv_path = OUTPUT_DIR / "three_group_ancova_per_subject.csv"
    df[["subject", "group", *METRICS.keys(), "total_dur_min"]].to_csv(csv_path, index=False)

    print("\nUnadjusted vs ANCOVA group p-values:")
    for metric, info in METRICS.items():
        print(f"  {info['name']:<22} unadjusted p={unadjusted[metric]['omnibus']['p_value']:.4f} "
              f"({unadjusted[metric]['omnibus']['test_used']})   "
              f"ANCOVA group p={ancova[metric]['group_p']:.4f}   "
              f"covariate p={ancova[metric]['cov_p']:.4f}")

    print(f"\nSaved: {report_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
