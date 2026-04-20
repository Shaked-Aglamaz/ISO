import os
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from utils.config import BASE_DIR, CENTRAL_PARIETAL_ROI, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects

from step4_distribution_analysis import (
    load_subject_data,
    filter_subjects_by_detection_rate,
    load_and_process_all_channels_data,
    normalize_subject_channels,
)


def test_normality_shapiro(group_data, metrics):
    """
    Perform Shapiro-Wilk test for normality on each group and metric.

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dictionaries containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dictionaries as values

    Returns:
    --------
    results : dict
        Nested dictionary: {metric: {group_name: {'statistic': float, 'p_value': float, 'is_normal': bool}}}
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"SHAPIRO-WILK NORMALITY TEST")
    print(f"{'='*80}")
    print(f"H0: Data is normally distributed")
    print(f"Significance level: α = 0.05")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        results[metric] = {}
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        for group_name, gdata in group_data.items():
            # Get valid data (drop NaN)
            values = gdata['data'][metric].dropna()

            if len(values) < 3:
                print(f"  {group_name:15s}: Insufficient data (N={len(values)}) - Test skipped")
                results[metric][group_name] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'is_normal': None,
                    'n': len(values)
                }
                continue

            # Perform Shapiro-Wilk test
            statistic, p_value = shapiro(values)
            is_normal = p_value > 0.05

            results[metric][group_name] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal,
                'n': len(values)
            }

            # Print results
            normality_status = "✓ NORMAL" if is_normal else "✗ NOT NORMAL"
            print(f"  {group_name:15s}: W={statistic:.4f}, p={p_value:.4f} [{normality_status}] (N={len(values)})")

    print(f"\n{'='*80}\n")
    return results


def test_homogeneity_levene(group_data, metrics):
    """
    Perform Levene's test for homogeneity of variances across groups for each metric.
    Tests if the groups have equal variances (homoscedasticity).

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dictionaries containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dictionaries as values

    Returns:
    --------
    results : dict
        Dictionary: {metric: {'statistic': float, 'p_value': float, 'equal_variances': bool, 'group_variances': dict}}
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"LEVENE'S TEST FOR HOMOGENEITY OF VARIANCES")
    print(f"{'='*80}")
    print(f"H0: All groups have equal variances")
    print(f"Significance level: α = 0.05")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        # Collect data from all groups
        group_samples = []
        group_variances = {}
        group_names = list(group_data.keys())

        for group_name in group_names:
            values = group_data[group_name]['data'][metric].dropna()
            if len(values) >= 2:  # Need at least 2 values to compute variance
                group_samples.append(values)
                group_variances[group_name] = {
                    'variance': values.var(),
                    'std': values.std(),
                    'n': len(values)
                }
            else:
                group_variances[group_name] = {
                    'variance': np.nan,
                    'std': np.nan,
                    'n': len(values)
                }

        # Print individual group variances
        print(f"  Group Variances:")
        for group_name in group_names:
            gv = group_variances[group_name]
            if pd.notna(gv['variance']):
                print(f"    {group_name:15s}: σ²={gv['variance']:.6f}, σ={gv['std']:.4f} (N={gv['n']})")
            else:
                print(f"    {group_name:15s}: Insufficient data (N={gv['n']})")

        # Perform Levene's test if we have at least 2 groups with sufficient data
        if len(group_samples) >= 2:
            statistic, p_value = levene(*group_samples, center='median')
            equal_variances = p_value > 0.05

            results[metric] = {
                'statistic': statistic,
                'p_value': p_value,
                'equal_variances': equal_variances,
                'group_variances': group_variances
            }

            # Print test results
            variance_status = "✓ EQUAL VARIANCES" if equal_variances else "✗ UNEQUAL VARIANCES"
            print(f"\n  Levene's Test: W={statistic:.4f}, p={p_value:.4f} [{variance_status}]")

            # Calculate variance ratio (max/min) for reference
            valid_vars = [gv['variance'] for gv in group_variances.values() if pd.notna(gv['variance'])]
            if len(valid_vars) >= 2:
                var_ratio = max(valid_vars) / min(valid_vars)
                print(f"  Variance Ratio (max/min): {var_ratio:.4f}")
        else:
            print(f"\n  ✗ Insufficient groups with data - Test skipped")
            results[metric] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'equal_variances': None,
                'group_variances': group_variances
            }

    print(f"\n{'='*80}\n")
    return results


def test_ttest_independent(group_data, metrics, equal_var=True):
    """
    Perform independent samples t-test (Student's t-test) for group comparisons.
    Use this when data is normally distributed (Shapiro-Wilk passed).

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dictionaries containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dictionaries as values
    equal_var : bool
        If True, perform standard t-test (equal variances assumed)
        If False, perform Welch's t-test (unequal variances)

    Returns:
    --------
    results : dict
        Nested dictionary with comparison results for each metric
    """
    results = {}
    group_names = list(group_data.keys())

    if len(group_names) != 2:
        print(f"\n⚠️  Warning: t-test requires exactly 2 groups. Found {len(group_names)} groups.")
        print(f"   For multiple group comparisons, use ANOVA instead.")
        return results

    test_type = "Student's t-test" if equal_var else "Welch's t-test"

    print(f"\n{'='*80}")
    print(f"INDEPENDENT SAMPLES T-TEST ({test_type})")
    print(f"{'='*80}")
    print(f"H0: The two groups have equal means")
    print(f"Significance level: α = 0.05")
    print(f"Equal variances assumed: {equal_var}")
    print(f"Groups: {group_names[0]} vs {group_names[1]}")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        # Get data for both groups
        group1_data = group_data[group_names[0]]['data'][metric].dropna()
        group2_data = group_data[group_names[1]]['data'][metric].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            print(f"  ✗ Insufficient data (N1={len(group1_data)}, N2={len(group2_data)}) - Test skipped")
            results[metric] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': None,
                'groups': group_names,
                'n1': len(group1_data),
                'n2': len(group2_data)
            }
            continue

        # Perform t-test
        statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=equal_var)
        significant = p_value < 0.05

        # Calculate effect size (Cohen's d)
        mean1, mean2 = group1_data.mean(), group2_data.mean()
        std1, std2 = group1_data.std(), group2_data.std()

        # Pooled standard deviation for Cohen's d
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

        results[metric] = {
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'groups': group_names,
            'n1': n1,
            'n2': n2,
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean1 - mean2,
            'cohens_d': cohens_d
        }

        # Print results
        significance_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
        print(f"  {group_names[0]:15s}: μ={mean1:.4f} ± {std1:.4f} (N={n1})")
        print(f"  {group_names[1]:15s}: μ={mean2:.4f} ± {std2:.4f} (N={n2})")
        print(f"\n  t-statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f} [{significance_status}]")
        print(f"  Mean difference: {mean1 - mean2:.4f} ({group_names[0]} - {group_names[1]})")
        print(f"  Cohen's d: {cohens_d:.4f}", end="")

        # Interpret effect size
        if pd.notna(cohens_d):
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect = "negligible"
            elif abs_d < 0.5:
                effect = "small"
            elif abs_d < 0.8:
                effect = "medium"
            else:
                effect = "large"
            print(f" ({effect} effect)")
        else:
            print()

    print(f"\n{'='*80}\n")
    return results


def test_mannwhitneyu(group_data, metrics):
    """
    Perform Mann-Whitney U test (Wilcoxon rank-sum test) for group comparisons.
    Use this when data is NOT normally distributed (Shapiro-Wilk failed).
    This is a non-parametric alternative to the t-test.

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dictionaries containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dictionaries as values

    Returns:
    --------
    results : dict
        Nested dictionary with comparison results for each metric
    """
    results = {}
    group_names = list(group_data.keys())

    if len(group_names) != 2:
        print(f"\n⚠️  Warning: Mann-Whitney U test requires exactly 2 groups. Found {len(group_names)} groups.")
        print(f"   For multiple group comparisons, use Kruskal-Wallis test instead.")
        return results

    print(f"\n{'='*80}")
    print(f"MANN-WHITNEY U TEST (Non-parametric)")
    print(f"{'='*80}")
    print(f"H0: The two groups have equal distributions")
    print(f"Significance level: α = 0.05")
    print(f"Groups: {group_names[0]} vs {group_names[1]}")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        # Get data for both groups
        group1_data = group_data[group_names[0]]['data'][metric].dropna()
        group2_data = group_data[group_names[1]]['data'][metric].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            print(f"  ✗ Insufficient data (N1={len(group1_data)}, N2={len(group2_data)}) - Test skipped")
            results[metric] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': None,
                'groups': group_names,
                'n1': len(group1_data),
                'n2': len(group2_data)
            }
            continue

        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        significant = p_value < 0.05

        # Calculate descriptive statistics
        median1, median2 = group1_data.median(), group2_data.median()
        q1_1, q3_1 = group1_data.quantile(0.25), group1_data.quantile(0.75)
        q1_2, q3_2 = group2_data.quantile(0.25), group2_data.quantile(0.75)

        # Calculate rank-biserial correlation (effect size for Mann-Whitney U)
        n1, n2 = len(group1_data), len(group2_data)
        rank_biserial = 1 - (2 * statistic) / (n1 * n2)

        results[metric] = {
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'groups': group_names,
            'n1': n1,
            'n2': n2,
            'median1': median1,
            'median2': median2,
            'median_diff': median1 - median2,
            'rank_biserial': rank_biserial
        }

        # Print results
        significance_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
        print(f"  {group_names[0]:15s}: M={median1:.4f}, IQR=[{q1_1:.4f}, {q3_1:.4f}] (N={n1})")
        print(f"  {group_names[1]:15s}: M={median2:.4f}, IQR=[{q1_2:.4f}, {q3_2:.4f}] (N={n2})")
        print(f"\n  U-statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f} [{significance_status}]")
        print(f"  Median difference: {median1 - median2:.4f} ({group_names[0]} - {group_names[1]})")
        print(f"  Rank-biserial correlation: {rank_biserial:.4f}", end="")

        # Interpret effect size
        if pd.notna(rank_biserial):
            abs_rb = abs(rank_biserial)
            if abs_rb < 0.1:
                effect = "negligible"
            elif abs_rb < 0.3:
                effect = "small"
            elif abs_rb < 0.5:
                effect = "medium"
            else:
                effect = "large"
            print(f" ({effect} effect)")
        else:
            print()

    print(f"\n{'='*80}\n")
    return results


def run_appropriate_tests(group_data, metrics, normality_results, levene_results):
    """
    Automatically run the appropriate statistical test for each metric based on:
    1. Normality test results (Shapiro-Wilk)
    2. Homogeneity of variances test results (Levene)

    Decision logic:
    - If both groups are normally distributed AND variances are equal → Student's t-test
    - If both groups are normally distributed BUT variances are unequal → Welch's t-test
    - If either group is NOT normally distributed → Mann-Whitney U test (non-parametric)

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dictionaries containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dictionaries as values
    normality_results : dict
        Results from test_normality_shapiro()
    levene_results : dict
        Results from test_homogeneity_levene()

    Returns:
    --------
    results : dict
        Dictionary with test results for each metric
    """
    results = {}
    group_names = list(group_data.keys())

    print(f"\n{'='*80}")
    print(f"AUTOMATIC TEST SELECTION BASED ON ASSUMPTIONS")
    print(f"{'='*80}")
    print(f"Groups: {' vs '.join(group_names)}")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        # Check normality for both groups
        all_normal = True
        for group_name in group_names:
            if metric in normality_results and group_name in normality_results[metric]:
                is_normal = normality_results[metric][group_name]['is_normal']
                if is_normal is None or not is_normal:
                    all_normal = False
                    print(f"  {group_name:15s}: NOT normally distributed")
                else:
                    print(f"  {group_name:15s}: Normally distributed ✓")
            else:
                all_normal = False
                print(f"  {group_name:15s}: No normality test result")

        # Check homogeneity of variances
        equal_variances = False
        if metric in levene_results:
            equal_variances = levene_results[metric].get('equal_variances', False)
            if equal_variances:
                print(f"  Variances: Equal ✓")
            else:
                print(f"  Variances: Unequal")
        else:
            print(f"  Variances: No test result")

        # Decide which test to use
        if all_normal:
            if equal_variances:
                print(f"\n  → Using: Student's t-test (parametric, equal variances)")
                test_results = {}

                group1_data = group_data[group_names[0]]['data'][metric].dropna()
                group2_data = group_data[group_names[1]]['data'][metric].dropna()

                if len(group1_data) >= 2 and len(group2_data) >= 2:
                    statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=True)
                    significant = p_value < 0.05

                    # Calculate Cohen's d
                    mean1, mean2 = group1_data.mean(), group2_data.mean()
                    std1, std2 = group1_data.std(), group2_data.std()
                    n1, n2 = len(group1_data), len(group2_data)
                    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

                    test_results = {
                        'test_used': 'Student t-test',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': significant,
                        'effect_size': cohens_d,
                        'effect_size_name': 'Cohen\'s d'
                    }

                    sig_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
                    print(f"     t = {statistic:.4f}, p = {p_value:.4f} [{sig_status}]")
                    print(f"     Cohen's d = {cohens_d:.4f}")
                else:
                    test_results = {'test_used': 'Student t-test', 'error': 'Insufficient data'}
                    print(f"     ✗ Insufficient data")

                results[metric] = test_results

            else:
                print(f"\n  → Using: Welch's t-test (parametric, unequal variances)")
                test_results = {}

                group1_data = group_data[group_names[0]]['data'][metric].dropna()
                group2_data = group_data[group_names[1]]['data'][metric].dropna()

                if len(group1_data) >= 2 and len(group2_data) >= 2:
                    statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
                    significant = p_value < 0.05

                    # Calculate Cohen's d
                    mean1, mean2 = group1_data.mean(), group2_data.mean()
                    std1, std2 = group1_data.std(), group2_data.std()
                    n1, n2 = len(group1_data), len(group2_data)
                    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

                    test_results = {
                        'test_used': 'Welch t-test',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': significant,
                        'effect_size': cohens_d,
                        'effect_size_name': 'Cohen\'s d'
                    }

                    sig_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
                    print(f"     t = {statistic:.4f}, p = {p_value:.4f} [{sig_status}]")
                    print(f"     Cohen's d = {cohens_d:.4f}")
                else:
                    test_results = {'test_used': 'Welch t-test', 'error': 'Insufficient data'}
                    print(f"     ✗ Insufficient data")

                results[metric] = test_results
        else:
            print(f"\n  → Using: Mann-Whitney U test (non-parametric)")
            test_results = {}

            group1_data = group_data[group_names[0]]['data'][metric].dropna()
            group2_data = group_data[group_names[1]]['data'][metric].dropna()

            if len(group1_data) >= 2 and len(group2_data) >= 2:
                statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                significant = p_value < 0.05

                # Calculate rank-biserial correlation
                n1, n2 = len(group1_data), len(group2_data)
                rank_biserial = 1 - (2 * statistic) / (n1 * n2)

                test_results = {
                    'test_used': 'Mann-Whitney U',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': significant,
                    'effect_size': rank_biserial,
                    'effect_size_name': 'Rank-biserial'
                }

                sig_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
                print(f"     U = {statistic:.4f}, p = {p_value:.4f} [{sig_status}]")
                print(f"     Rank-biserial = {rank_biserial:.4f}")
            else:
                test_results = {'test_used': 'Mann-Whitney U', 'error': 'Insufficient data'}
                print(f"     ✗ Insufficient data")

            results[metric] = test_results

    print(f"\n{'='*80}\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY OF STATISTICAL TESTS")
    print(f"{'='*80}\n")
    for metric, result in results.items():
        if 'error' not in result:
            print(f"{metrics[metric]['name']:20s}: {result['test_used']:20s} | "
                  f"p={result['p_value']:.4f} | "
                  f"{'SIGNIFICANT ✓' if result['significant'] else 'NOT SIGNIFICANT'}")
    print(f"\n{'='*80}\n")

    return results


def test_omnibus(group_data, metrics, normality_results):
    """
    Run omnibus test for 3+ group comparisons.

    - If ALL groups are normal for a metric → One-Way ANOVA (F-statistic)
    - If ANY group is non-normal → Kruskal-Wallis (H-statistic)

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dicts containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dicts as values
    normality_results : dict
        Results from test_normality_shapiro()

    Returns:
    --------
    results : dict
        {metric: {'test_used': str, 'statistic': float, 'p_value': float,
                  'significant': bool, 'eta_squared': float}}
    """
    results = {}
    group_names = list(group_data.keys())

    print(f"\n{'='*80}")
    print(f"OMNIBUS TEST ({len(group_names)}-GROUP COMPARISON)")
    print(f"{'='*80}")
    print(f"Groups: {', '.join(group_names)}")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        # Check normality across all groups
        all_normal = True
        for group_name in group_names:
            if metric in normality_results and group_name in normality_results[metric]:
                is_normal = normality_results[metric][group_name].get('is_normal')
                if is_normal is None or not is_normal:
                    all_normal = False
            else:
                all_normal = False

        # Collect group samples
        samples = []
        for group_name in group_names:
            values = group_data[group_name]['data'][metric].dropna()
            samples.append(values)
            n = len(values)
            mean_val = values.mean() if n > 0 else float('nan')
            std_val = values.std() if n > 0 else float('nan')
            print(f"  {group_name:15s}: N={n}, μ={mean_val:.4f} ± {std_val:.4f}")

        # Check minimum sample sizes
        if any(len(s) < 2 for s in samples):
            print(f"\n  ✗ Insufficient data in one or more groups - Test skipped")
            results[metric] = {'test_used': 'skipped', 'statistic': np.nan,
                              'p_value': np.nan, 'significant': None, 'eta_squared': np.nan}
            continue

        # Compute eta-squared (proportion of variance explained by group)
        all_values = np.concatenate([s.values for s in samples])
        grand_mean = np.mean(all_values)
        ss_total = np.sum((all_values - grand_mean) ** 2)
        ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        if all_normal:
            # One-Way ANOVA
            statistic, p_value = f_oneway(*samples)
            test_name = 'One-Way ANOVA'
            stat_symbol = 'F'
        else:
            # Kruskal-Wallis
            statistic, p_value = kruskal(*samples)
            test_name = 'Kruskal-Wallis'
            stat_symbol = 'H'

        significant = p_value < 0.05

        results[metric] = {
            'test_used': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'eta_squared': eta_sq,
            'all_normal': all_normal
        }

        sig_status = "✓ SIGNIFICANT" if significant else "✗ NOT SIGNIFICANT"
        print(f"\n  Normality: {'All groups normal → parametric' if all_normal else 'Non-normal group(s) → non-parametric'}")
        print(f"  → {test_name}: {stat_symbol} = {statistic:.4f}, p = {p_value:.4f} [{sig_status}]")
        print(f"  η² = {eta_sq:.4f}", end="")

        # Interpret eta-squared
        if pd.notna(eta_sq):
            if eta_sq < 0.01:
                effect = "negligible"
            elif eta_sq < 0.06:
                effect = "small"
            elif eta_sq < 0.14:
                effect = "medium"
            else:
                effect = "large"
            print(f" ({effect} effect)")
        else:
            print()

    print(f"\n{'='*80}\n")
    return results


def test_posthoc(group_data, metrics, omnibus_results):
    """
    Run post-hoc pairwise comparisons for metrics where the omnibus test was significant.

    - After ANOVA → Tukey's HSD
    - After Kruskal-Wallis → Dunn's test with Holm correction

    Parameters:
    -----------
    group_data : dict
        Dictionary with group names as keys and dicts containing 'data' DataFrames
    metrics : dict
        Dictionary with metric names as keys and info dicts as values
    omnibus_results : dict
        Results from test_omnibus()

    Returns:
    --------
    results : dict
        {metric: {'test_used': str, 'pairs': {(g1,g2): {'p_value': float, 'significant': bool, ...}}}}
    """
    results = {}
    group_names = list(group_data.keys())

    print(f"\n{'='*80}")
    print(f"POST-HOC PAIRWISE COMPARISONS")
    print(f"{'='*80}")
    print(f"Groups: {', '.join(group_names)}")
    print(f"{'='*80}\n")

    for metric in metrics.keys():
        print(f"\n{metrics[metric]['name']} ({metrics[metric]['unit']}):")
        print(f"{'-'*60}")

        omnibus = omnibus_results.get(metric, {})

        if not omnibus.get('significant', False):
            print(f"  Omnibus test was NOT significant (p={omnibus.get('p_value', np.nan):.4f}) → No post-hoc needed")
            results[metric] = {'test_used': 'none', 'pairs': {},
                              'reason': 'omnibus not significant'}
            continue

        # Build combined DataFrame for post-hoc tests
        combined = []
        for group_name in group_names:
            values = group_data[group_name]['data'][metric].dropna()
            df_temp = pd.DataFrame({'value': values, 'group': group_name})
            combined.append(df_temp)
        combined_df = pd.concat(combined, ignore_index=True)

        pairs = {}

        if omnibus.get('all_normal', False):
            # Tukey's HSD
            test_name = "Tukey's HSD"
            tukey = pairwise_tukeyhsd(combined_df['value'], combined_df['group'], alpha=0.05)

            print(f"  → {test_name}")
            print(f"  {tukey.summary()}\n")

            # Extract pairwise results using Tukey's public attributes
            for k in range(len(tukey.pvalues)):
                g1 = str(tukey.groupsunique[tukey._multicomp.pairindices[0][k]])
                g2 = str(tukey.groupsunique[tukey._multicomp.pairindices[1][k]])
                pairs[(g1, g2)] = {
                    'p_value': float(tukey.pvalues[k]),
                    'significant': bool(tukey.reject[k]),
                    'mean_diff': float(tukey.meandiffs[k]),
                }
        else:
            # Dunn's test with Holm correction
            test_name = "Dunn's test (Holm correction)"

            dunn_results = sp.posthoc_dunn(combined_df, val_col='value',
                                           group_col='group', p_adjust='holm')

            print(f"  → {test_name}")
            print(f"  P-value matrix:")
            print(f"  {dunn_results.to_string()}\n")

            # Extract pairwise results
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1, g2 = group_names[i], group_names[j]
                    p_val = dunn_results.loc[g1, g2]

                    # Compute median difference for reference
                    med1 = group_data[g1]['data'][metric].dropna().median()
                    med2 = group_data[g2]['data'][metric].dropna().median()

                    pairs[(g1, g2)] = {
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'median_diff': med1 - med2
                    }

        results[metric] = {'test_used': test_name, 'pairs': pairs}

        # Print pairwise summary
        print(f"  Pairwise summary:")
        for (g1, g2), pdata in pairs.items():
            sig = "✓ SIGNIFICANT" if pdata['significant'] else "✗ NOT SIGNIFICANT"
            print(f"    {g1} vs {g2}: p = {pdata['p_value']:.4f} [{sig}]")

    print(f"\n{'='*80}\n")
    return results


def run_three_group_tests(group_data, metrics):
    """
    Full 3-group statistical comparison pipeline:
      0. Shapiro-Wilk normality check (existing function)
      1. Omnibus test (ANOVA or Kruskal-Wallis) with eta-squared
      2. Post-hoc pairwise tests (Tukey HSD or Dunn-Holm) if omnibus is significant

    Parameters:
    -----------
    group_data : dict
        {group_name: {'data': DataFrame, ...}} — must contain 3 groups
    metrics : dict
        {metric_col: {'name': str, 'unit': str}}

    Returns:
    --------
    dict with keys: 'normality', 'omnibus', 'posthoc'
    """
    print(f"\n{'#'*80}")
    print(f"  THREE-GROUP STATISTICAL ANALYSIS PIPELINE")
    print(f"{'#'*80}\n")

    # Step 0: Normality
    normality_results = test_normality_shapiro(group_data, metrics)

    # Step 1: Omnibus
    omnibus_results = test_omnibus(group_data, metrics, normality_results)

    # Step 2: Post-hoc (only where omnibus is significant)
    posthoc_results = test_posthoc(group_data, metrics, omnibus_results)

    # Final summary
    print(f"\n{'#'*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*80}\n")
    for metric in metrics.keys():
        omnibus = omnibus_results.get(metric, {})
        posthoc = posthoc_results.get(metric, {})

        print(f"{metrics[metric]['name']}:")
        print(f"  Omnibus: {omnibus.get('test_used', 'N/A')}, "
              f"p={omnibus.get('p_value', np.nan):.4f}, "
              f"η²={omnibus.get('eta_squared', np.nan):.4f}")

        if posthoc.get('pairs'):
            for (g1, g2), pdata in posthoc['pairs'].items():
                sig = "*" if pdata['significant'] else "ns"
                print(f"  {g1} vs {g2}: p={pdata['p_value']:.4f} ({sig})")
        elif omnibus.get('significant'):
            print(f"  Post-hoc: not run")
        else:
            print(f"  Post-hoc: not needed (omnibus ns)")
        print()

    return {
        'normality': normality_results,
        'omnibus': omnibus_results,
        'posthoc': posthoc_results
    }


def load_and_process_all_channels_data_normalized(subjects, group=None):
    """
    Load per-subject data, normalize each subject's channels using
    normalize_subject_channels (same as topo plots), then average
    normalized values across channels to get one value per subject.

    Returns same format as load_and_process_all_channels_data:
        (subject_averages DataFrame, total_channels, n_valid_subjects)
    """
    metrics = ['peak_frequency', 'bandwidth', 'auc']
    all_rows = []
    total_channels = 0

    for subject in subjects:
        subject_data = load_subject_data(subject, group)
        if subject_data is None:
            continue

        eeg_data = subject_data.copy()
        if len(eeg_data) == 0:
            continue

        total_channels += len(eeg_data)

        # Normalize each metric's channel values, then average across channels
        row = {'subject': subject}
        for metric in metrics:
            values = eeg_data[metric].values.astype(float)
            normalized = normalize_subject_channels(values)
            row[metric] = np.nanmean(normalized)

        all_rows.append(row)

    if not all_rows:
        return None, 0, 0

    subject_averages = pd.DataFrame(all_rows)
    print(f"✓ Processed {len(subject_averages)} subjects (normalized) with {total_channels} total channels")
    return subject_averages, total_channels, len(subject_averages)


def load_and_process_roi_data(subjects, group=None, roi_channels=None):
    """
    Load per-subject data, keep only channels inside the given ROI,
    then average across those ROI channels to get one value per subject.

    roi_channels: iterable of channel names. Defaults to CENTRAL_PARIETAL_ROI.

    Returns same format as load_and_process_all_channels_data:
        (subject_averages DataFrame, total_channels, n_valid_subjects)
    """
    if roi_channels is None:
        roi_channels = CENTRAL_PARIETAL_ROI
    roi_set = set(roi_channels)
    all_data = []
    total_channels = 0

    for subject in subjects:
        subject_data = load_subject_data(subject, group)
        if subject_data is None:
            continue

        # Filter to ROI channels only
        eeg_data = subject_data[subject_data['channel'].isin(roi_set)].copy()
        if len(eeg_data) == 0:
            continue

        eeg_data['subject'] = subject
        all_data.append(eeg_data)
        total_channels += len(eeg_data)

    if not all_data:
        return None, 0, 0

    combined_df = pd.concat(all_data, ignore_index=True)
    subject_averages = combined_df.groupby('subject')[['peak_frequency', 'bandwidth', 'auc']].mean().reset_index()
    print(f"✓ Processed {len(subject_averages)} subjects (ROI, {len(roi_set)} channels) with {total_channels} total channel-observations")
    return subject_averages, total_channels, len(subject_averages)


def load_and_process_roi_data_normalized(subjects, group=None, roi_channels=None):
    """
    Per-subject normalization by the WHOLE-SCALP mean (same as the topo pipeline),
    then average the normalized values over the ROI channels only.

    A per-subject ROI value > 1 means the ROI is enhanced relative to the rest
    of that subject's scalp — i.e. the hotspot is concentrated there.

    roi_channels: iterable of channel names. Defaults to CENTRAL_PARIETAL_ROI.

    Returns same format as load_and_process_all_channels_data:
        (subject_averages DataFrame, total_channels, n_valid_subjects)
    """
    if roi_channels is None:
        roi_channels = CENTRAL_PARIETAL_ROI
    roi_set = set(roi_channels)
    metrics = ['peak_frequency', 'bandwidth', 'auc']
    all_rows = []
    total_channels = 0

    for subject in subjects:
        subject_data = load_subject_data(subject, group)
        if subject_data is None or len(subject_data) == 0:
            continue

        # Normalize using ALL channels, then restrict to ROI
        roi_mask = subject_data['channel'].isin(roi_set).values
        if not roi_mask.any():
            continue

        total_channels += int(roi_mask.sum())

        row = {'subject': subject}
        for metric in metrics:
            full_values = subject_data[metric].values.astype(float)
            normalized_full = normalize_subject_channels(full_values)
            row[metric] = np.nanmean(normalized_full[roi_mask])

        all_rows.append(row)

    if not all_rows:
        return None, 0, 0

    subject_averages = pd.DataFrame(all_rows)
    print(f"✓ Processed {len(subject_averages)} subjects (ROI, scalp-normalized) with {total_channels} total ROI channel-observations")
    return subject_averages, total_channels, len(subject_averages)


def plot_group_comparison(groups_dict, output_dir, test_results=None, normalize=False,
                          roi_only=False, roi_channels=None, roi_label='ROI',
                          metrics_filter=None):
    """
    Compare multiple groups side-by-side for each spectral parameter.
    Each data point represents one subject's average across all channels.

    Parameters:
    -----------
    groups_dict : dict
        Dictionary with group names as keys and (subjects_list, dir_path) tuples as values
    output_dir : Path
    test_results : dict, optional
        Statistical test results; adds significance brackets on plots.
    normalize : bool
        If True, normalize each subject's channel values before averaging.
    roi_only : bool
        If True, restrict to ROI channels (uses roi_channels if given, else CENTRAL_PARIETAL_ROI).
    roi_channels : iterable of str, optional
        Explicit channel list to restrict to. Only used when roi_only=True.
    roi_label : str
        Label for the ROI (used in title + filename, e.g. 'ROI', 'extended_ROI').
    metrics_filter : list of str, optional
        Subset of metrics to plot (e.g. ['auc']). If None, plot all three.
    """
    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    if metrics_filter is not None:
        metrics = {k: metrics[k] for k in metrics_filter if k in metrics}

    # Define colors for groups
    colors = ['#8dd3c7', '#fb8072', '#80b1d3', '#fdb462', '#b3de69']

    # Collect data for all groups
    group_data = {}
    for group_name, (subjects, dir_path) in groups_dict.items():
        label_parts = [group_name]
        if roi_only:
            label_parts.append(roi_label)
        if normalize:
            label_parts.append("normalized")
        print(f"\nLoading data for group: {' '.join(label_parts)}")
        if roi_only and normalize:
            subject_averages, total_channels, n_subjects = load_and_process_roi_data_normalized(
                subjects, dir_path, roi_channels=roi_channels)
        elif roi_only:
            subject_averages, total_channels, n_subjects = load_and_process_roi_data(
                subjects, dir_path, roi_channels=roi_channels)
        elif normalize:
            subject_averages, total_channels, n_subjects = load_and_process_all_channels_data_normalized(subjects, dir_path)
        else:
            subject_averages, total_channels, n_subjects = load_and_process_all_channels_data(subjects, dir_path)
        group_data[group_name] = {
            'data': subject_averages,
            'n_subjects': n_subjects,
            'total_channels': total_channels
        }

    # Create figure: one subplot per metric
    n_metrics = len(metrics)
    fig_width = max(7, 6.5 * n_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, 8), squeeze=False)
    axes = axes[0]
    roi_title = f" ({roi_label})" if roi_only else ""
    norm_title = " (Normalized)" if normalize else ""
    fig.suptitle(f'Group Comparison - Spectral Parameters{roi_title}{norm_title}\n(Each point = 1 subject)', fontsize=16, fontweight='bold', y=0.98)

    for metric_idx, (metric, metric_info) in enumerate(metrics.items()):
        ax = axes[metric_idx]

        # Prepare data for seaborn
        plot_data = []
        for group_name in group_data.keys():
            group_df = group_data[group_name]['data']
            valid_data = group_df[metric].dropna()
            for val in valid_data:
                plot_data.append({'Group': group_name, metric: val})

        plot_df = pd.DataFrame(plot_data)

        if len(plot_df) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title(metric_info['name'])
            continue

        # Create violin plot
        violin_parts = sns.violinplot(data=plot_df, x='Group', y=metric, hue='Group', ax=ax,
                                      palette=colors[:len(group_data)], inner=None,
                                      cut=0, linewidth=1.5, alpha=0.7, legend=False)

        # Modify violins to show only right half
        for collection in ax.collections:
            if hasattr(collection, 'get_paths'):
                paths = collection.get_paths()
                if len(paths) > 0:
                    for path in paths:
                        vertices = path.vertices
                        # Find center x position for this violin
                        center_x = np.mean(vertices[:, 0])
                        # Keep only right half (x >= center)
                        mask = vertices[:, 0] >= center_x
                        vertices[~mask, 0] = center_x  # Set left side to center

        # Add boxplot overlay
        box_parts = ax.boxplot(
            [group_data[gname]['data'][metric].dropna() for gname in group_data.keys()],
            positions=range(len(group_data)),
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color='orange', linewidth=2)
        )

        # Add individual dots with jitter
        np.random.seed(42)
        for i, group_name in enumerate(group_data.keys()):
            group_values = group_data[group_name]['data'][metric].dropna()
            x_jitter = np.random.normal(i, 0.04, size=len(group_values))
            ax.scatter(x_jitter, group_values, alpha=0.5, s=40, color='darkblue',
                      edgecolors='black', linewidths=0.5, zorder=3)

        # Formatting
        ax.set_title(metric_info['name'], fontsize=14, fontweight='bold')
        unit_label = 'Normalized (/ mean)' if normalize else metric_info["unit"]
        ax.set_ylabel(f'{metric_info["name"]} ({unit_label})', fontsize=12)
        ax.set_xlabel('Group', fontsize=12)
        ax.set_xticks(range(len(group_data)))
        ax.set_xticklabels(list(group_data.keys()), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text for each group
        stats_y_position = 0.98
        for group_idx, group_name in enumerate(group_data.keys()):
            group_values = group_data[group_name]['data'][metric].dropna()
            n = len(group_values)
            mean_val = group_values.mean()
            std_val = group_values.std()
            median_val = group_values.median()

            stats_text = f'{group_name}: N={n}, μ={mean_val:.3f}, σ={std_val:.3f}, M={median_val:.3f}'
            ax.text(0.02, stats_y_position - (group_idx * 0.06), stats_text,  # Changed from 0.08 to 0.06
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor=colors[group_idx], alpha=0.3))

        # Add significance markers if test results are provided
        if test_results is not None and metric in test_results:
            result = test_results[metric]
            pairs = result.get('pairs', {})
            # Build mapping from group name to x-position
            group_name_list = list(group_data.keys())
            group_pos = {name: idx for idx, name in enumerate(group_name_list)}

            # Collect significant pairs
            sig_pairs = []
            for (g1, g2), pdata in pairs.items():
                if pdata.get('significant', False):
                    p_value = pdata['p_value']
                    if p_value < 0.001:
                        sig_marker = '***'
                    elif p_value < 0.01:
                        sig_marker = '**'
                    elif p_value < 0.05:
                        sig_marker = '*'
                    else:
                        continue
                    x1 = group_pos.get(g1, group_pos.get(g2))
                    x2 = group_pos.get(g2, group_pos.get(g1))
                    if x1 is not None and x2 is not None:
                        sig_pairs.append((min(x1, x2), max(x1, x2), p_value, sig_marker))

            if sig_pairs:
                # Sort by span width so narrower brackets are drawn lower
                sig_pairs.sort(key=lambda t: t[1] - t[0])

                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                # Extend upper limit to fit all brackets
                new_y_max = y_max + 0.12 * y_range * len(sig_pairs)
                ax.set_ylim(y_min, new_y_max)
                y_range = new_y_max - y_min
                bar_height = 0.015 * y_range

                for bracket_idx, (x1, x2, p_value, sig_marker) in enumerate(sig_pairs):
                    y_position = y_max + (0.03 + bracket_idx * 0.10) * y_range

                    # Horizontal line
                    ax.plot([x1, x2], [y_position, y_position], 'k-', linewidth=1.5, zorder=10)
                    # Vertical ticks
                    ax.plot([x1, x1], [y_position - bar_height, y_position], 'k-', linewidth=1.5, zorder=10)
                    ax.plot([x2, x2], [y_position - bar_height, y_position], 'k-', linewidth=1.5, zorder=10)
                    # Asterisks
                    ax.text((x1 + x2) / 2, y_position + 0.005 * y_range, sig_marker,
                           ha='center', va='bottom', fontsize=18, fontweight='bold', zorder=10)
                    # P-value
                    ax.text((x1 + x2) / 2, y_position - 0.020 * y_range, f'p={p_value:.4f}',
                           ha='center', va='top', fontsize=9, style='italic', zorder=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    roi_suffix = f"_{roi_label}" if roi_only else ""
    norm_suffix = "_normalized" if normalize else ""
    metrics_suffix = f"_{'_'.join(metrics.keys())}" if metrics_filter is not None else ""
    plot_path = output_dir / f"group_comparison_violin{roi_suffix}{norm_suffix}{metrics_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Group comparison plot saved to: {plot_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"GROUP COMPARISON SUMMARY")
    print(f"{'='*80}")
    for group_name, gdata in group_data.items():
        print(f"\n{group_name}:")
        print(f"  Subjects: {gdata['n_subjects']}")
        print(f"  Total channels: {gdata['total_channels']}")
        for metric, metric_info in metrics.items():
            values = gdata['data'][metric].dropna()
            if len(values) > 0:
                print(f"  {metric_info['name']}: {values.mean():.3f} ± {values.std():.3f} {metric_info['unit']} (median: {values.median():.3f})")


def run_two_group_comparison():
    """Run 2-group statistical comparison (Young vs Elderly) and save results."""
    young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    young_results_dir = Path("results/new_iso_results")
    young_subjects = [sub for sub in young_subjects if (young_results_dir / sub).exists() and sub != "dashboards"]
    young_subjects, _ = filter_subjects_by_detection_rate(young_subjects, dir_path=young_results_dir)

    elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
    elderly_results_dir = Path("results/new_elderly_results")
    elderly_subjects = [sub for sub in elderly_subjects if (elderly_results_dir / sub).exists() and sub != "dashboards"]
    elderly_subjects, _ = filter_subjects_by_detection_rate(elderly_subjects, dir_path=elderly_results_dir)

    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }

    group_data = {}
    group_data['Young'] = {
        'data': load_and_process_all_channels_data(young_subjects, young_results_dir)[0],
        'n_subjects': len(young_subjects)
    }
    group_data['Elderly'] = {
        'data': load_and_process_all_channels_data(elderly_subjects, elderly_results_dir)[0],
        'n_subjects': len(elderly_subjects)
    }

    # Run assumption tests
    normality_results = test_normality_shapiro(group_data, metrics)
    levene_results = test_homogeneity_levene(group_data, metrics)

    # Run appropriate statistical tests based on assumptions
    appropriate_test_results = run_appropriate_tests(group_data, metrics, normality_results, levene_results)

    # Plot group comparison with significance markers
    groups_dict = {
        'Young': (young_subjects, young_results_dir),
        'Elderly': (elderly_subjects, elderly_results_dir)
    }
    plot_group_comparison(groups_dict, output_dir=Path("results/group_comparison_results/two_groups"),
                          test_results=appropriate_test_results)


def run_three_group_comparison():
    """Run 3-group statistical comparison (Young vs Elderly vs MCI) and save results."""
    # Load all 3 groups
    young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    young_results_dir = Path("results/new_iso_results")
    young_subjects = [sub for sub in young_subjects if (young_results_dir / sub).exists() and sub != "dashboards"]
    young_subjects, _ = filter_subjects_by_detection_rate(young_subjects, dir_path=young_results_dir)

    elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
    elderly_results_dir = Path("results/new_elderly_results")
    elderly_subjects = [sub for sub in elderly_subjects if (elderly_results_dir / sub).exists() and sub != "dashboards"]
    elderly_subjects, _ = filter_subjects_by_detection_rate(elderly_subjects, dir_path=elderly_results_dir)

    mci_subjects = get_all_subjects(f"{BASE_DIR}/MCI_clean/")
    mci_results_dir = Path("results/new_MCI_results")
    mci_subjects = [sub for sub in mci_subjects if (mci_results_dir / sub).exists() and sub != "dashboards"]
    mci_subjects, _ = filter_subjects_by_detection_rate(mci_subjects, dir_path=mci_results_dir)

    metrics = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }

    group_data = {}
    group_data['Young'] = {
        'data': load_and_process_all_channels_data(young_subjects, young_results_dir)[0],
        'n_subjects': len(young_subjects)
    }
    group_data['Elderly'] = {
        'data': load_and_process_all_channels_data(elderly_subjects, elderly_results_dir)[0],
        'n_subjects': len(elderly_subjects)
    }
    group_data['MCI'] = {
        'data': load_and_process_all_channels_data(mci_subjects, mci_results_dir)[0],
        'n_subjects': len(mci_subjects)
    }

    # Run the full 3-group statistical pipeline (output to file)
    comparison_output_dir = Path("results/group_comparison_results/three_groups")
    comparison_output_dir.mkdir(exist_ok=True, parents=True)
    stats_report_path = comparison_output_dir / "three_group_statistics.txt"
    with open(stats_report_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            three_group_results = run_three_group_tests(group_data, metrics)
    print(f"Three-group statistics report saved to: {stats_report_path}")

    # Plot 3-group comparison with post-hoc significance markers
    groups_dict = {
        'Young': (young_subjects, young_results_dir),
        'Elderly': (elderly_subjects, elderly_results_dir),
        'MCI': (mci_subjects, mci_results_dir)
    }
    plot_group_comparison(groups_dict, output_dir=comparison_output_dir,
                          test_results=three_group_results['posthoc'])

    # Normalized violin plot (same normalization as topo plots)
    plot_group_comparison(groups_dict, output_dir=comparison_output_dir,
                          test_results=three_group_results['posthoc'],
                          normalize=True)

    # Normalized ROI violin plots (same per-subject normalization as the topo).
    # Each subject's channel values are divided by their own outlier-trimmed scalp mean
    # before averaging over the ROI, so points > 1 mean the ROI is enhanced vs the rest
    # of the scalp. This isolates hotspot concentration from rising global baseline.
    for roi_channels, roi_label in [
        (CENTRAL_PARIETAL_ROI, 'ROI'),
        (EXTENDED_CENTRAL_PARIETAL_ROI, 'extended_ROI'),
    ]:
        roi_group_data = {
            'Young': {
                'data': load_and_process_roi_data_normalized(
                    young_subjects, young_results_dir, roi_channels=roi_channels)[0],
                'n_subjects': len(young_subjects),
            },
            'Elderly': {
                'data': load_and_process_roi_data_normalized(
                    elderly_subjects, elderly_results_dir, roi_channels=roi_channels)[0],
                'n_subjects': len(elderly_subjects),
            },
            'MCI': {
                'data': load_and_process_roi_data_normalized(
                    mci_subjects, mci_results_dir, roi_channels=roi_channels)[0],
                'n_subjects': len(mci_subjects),
            },
        }
        roi_stats_path = comparison_output_dir / f"three_group_statistics_{roi_label}_normalized_auc.txt"
        with open(roi_stats_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                roi_results = run_three_group_tests(roi_group_data, {'auc': metrics['auc']})
        print(f"Normalized ROI ({roi_label}) statistics saved to: {roi_stats_path}")

        plot_group_comparison(
            groups_dict, output_dir=comparison_output_dir,
            test_results=roi_results['posthoc'],
            normalize=True, roi_only=True,
            roi_channels=roi_channels, roi_label=roi_label,
            metrics_filter=['auc'],
        )


def main():
    """Run group comparison analyses."""
    # run_two_group_comparison()
    run_three_group_comparison()


if __name__ == "__main__":
    main()
