import os
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import mne
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
from scipy.stats import f_oneway, f as f_dist
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from utils.config import BASE_DIR, CENTRAL_PARIETAL_ROI, EXTENDED_CENTRAL_PARIETAL_ROI
from utils.utils import get_all_subjects

import sys
sys.path.append(str(Path(__file__).parent))
from step4_distribution_analysis import (
    load_subject_data,
    filter_subjects_by_detection_rate,
    create_electrode_montage,
    load_and_combine_subject_data,
    filter_eeg_channels,
    setup_electrode_info,
    extract_channel_values,
    plot_single_topography,
    normalize_subject_channels
)


# =============================================================================
# STEP 1: DATA STRUCTURING AND PREPROCESSING
# =============================================================================

def load_all_subjects_data(subjects, dir_path=None):
    """Load channel summary data for all subjects."""
    subjects_data = {}
    failed_subjects = []
    for subject in subjects:
        data = load_subject_data(subject, dir_path)
        if data is not None:
            subjects_data[subject] = data
        else:
            failed_subjects.append(subject)
    
    if failed_subjects:
        print(f"✗ Failed subjects: {', '.join(failed_subjects)}")
    return subjects_data


def create_evoked_array_for_subject(subject_data, metric, montage, available_channels, normalize=False, subject_id=''):
    """
    Create an MNE EvokedArray object for a single subject and metric.
    Returns:
    --------
    evoked : mne.EvokedArray
        Evoked array with topography data as a single time point
    """
    # Filter to only channels with known positions
    subject_data_filtered = subject_data[subject_data['channel'].isin(available_channels)].copy()
    
    # Sort by available_channels order to ensure consistency
    subject_data_filtered = subject_data_filtered.set_index('channel').loc[available_channels].reset_index()
    
    # Extract metric values
    values = subject_data_filtered[metric].values

    # NaN imputation + optional per-subject normalization (shared with step4)
    if normalize:
        values = normalize_subject_channels(values)
    else:
        # Still need to impute NaN for EvokedArray (can't have NaN)
        if np.any(np.isnan(values)):
            values = np.where(np.isnan(values), np.nanmean(values), values)
    
    # Create info object
    info = mne.create_info(ch_names=available_channels, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    
    # Create evoked array with data as a single time point
    # Shape: (n_channels, 1) - one time point
    data = values.reshape(-1, 1)
    evoked = mne.EvokedArray(data, info, tmin=0, comment=f'{subject_id}_{metric}')
    return evoked


def prepare_evoked_arrays_for_group(subjects_data, metric, normalize=False):
    """
    Prepare EvokedArray objects for all subjects in a group.    
    Returns:
    --------
    evoked_list : list
        List of mne.EvokedArray objects (one per subject)
    montage : mne.channels.DigMontage
        Common electrode montage
    available_channels : list
        List of channels with known positions
    """
    print(f"\nCREATING EVOKED ARRAYS FOR METRIC: {metric.upper()} (N={len(subjects_data)}, Normalize={normalize})")
    
    # Get all unique channels across subjects
    all_channels = set()
    for subject_data in subjects_data.values():
        all_channels.update(subject_data['channel'].tolist())
    
    # Create montage for all channels
    """ Create electrode montage and filter channels with known positions."""
    montage, available_channels = create_electrode_montage(list(all_channels))
    if montage is None:
        raise ValueError("Failed to create electrode montage")

    print(f"🎯 Electrode positioning: {len(available_channels)}/{len(all_channels)} channels with known positions")
    print(f"{'='*80}")
    
    # Create evoked array for each subject
    evoked_list = []
    for subject_id, subject_data in subjects_data.items():
        evoked = create_evoked_array_for_subject(subject_data, metric, montage, available_channels, normalize, subject_id)
        evoked_list.append(evoked)
    
    return evoked_list, montage, available_channels


def create_evoked_from_group_average(subjects, metric, dir_path=None):
    """
    Create a single EvokedArray by averaging across subjects (like step4).
    Returns RAW averaged data (not normalized) - normalization will be done during plotting.
    
    Returns:
    --------
    evoked : mne.EvokedArray
        Single evoked array representing group average (raw values)
    montage : mne.channels.DigMontage
        Electrode montage
    available_channels : list
        List of channels with known positions
    """
    # Use step4's data loading and averaging approach
    combined_data = load_and_combine_subject_data(subjects, dir_path)
    if combined_data is None:
        raise ValueError("Failed to load subject data")
    
    # Filter and average channels (step4's approach)
    channel_averages = filter_eeg_channels(combined_data)
    if channel_averages is None:
        raise ValueError("No EEG channels found")
    
    # Setup electrode info
    info, channel_averages_filtered, channel_names = setup_electrode_info(channel_averages)
    if info is None:
        raise ValueError("Failed to setup electrode info")
    
    # Extract values for the metric (using step4's column naming convention)
    metric_column = f'avg_{metric}'
    if metric_column not in channel_averages_filtered.columns:
        raise ValueError(f"Metric {metric_column} not found in data")
    
    values = extract_channel_values(channel_averages_filtered, channel_names, metric_column)
    
    # NO normalization here - will be done in plot_single_topography (like step4)
    
    # Create evoked array with raw values
    data = values.reshape(-1, 1)  # (n_channels, 1)
    evoked = mne.EvokedArray(data, info, tmin=0, comment=f'group_avg_{metric}')
    
    return evoked, info.get_montage(), channel_names


def compute_adjacency_matrix(evoked_list):
    """
    Compute spatial adjacency matrix for clustering.
    Returns:
    --------
    adjacency : scipy.sparse matrix
        Adjacency matrix defining spatial neighbors
    ch_names : list
        Channel names in the same order as adjacency matrix
    """
    if len(evoked_list) == 0:
        raise ValueError("Empty evoked_list")
    
    # Get info from first subject (all should have same montage)
    info = evoked_list[0].info
    
    # Compute adjacency matrix
    adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')
    
    print(f"\nSPATIAL ADJACENCY MATRIX")
    print(f"N channels: {len(ch_names)}, matrix shape: {adjacency.shape}")
    print(f"Number of edges (channel pairs): {adjacency.nnz}, Average neighbors per channel: {adjacency.nnz / len(ch_names):.1f}")
    print(f"{'='*80}\n")
    return adjacency, ch_names


def prepare_data_for_comparison(group1_subjects, group2_subjects, metric, dir_path1=None, dir_path2=None, 
                                normalize=False, min_detection_rate=0.2):
    """
    Complete pipeline to prepare data for topographic comparison between two groups.
    Returns:
    --------
    group1_evoked : list
        List of EvokedArray objects for group 1
    group2_evoked : list
        List of EvokedArray objects for group 2
    adjacency : scipy.sparse matrix
        Spatial adjacency matrix
    ch_names : list
        Channel names
    group1_filtered : list
        Filtered subject IDs for group 1
    group2_filtered : list
        Filtered subject IDs for group 2
    """
    # Filter subjects by detection rate
    print("GROUP 1:")
    group1_filtered, _ = filter_subjects_by_detection_rate(
        group1_subjects, min_detection_rate=min_detection_rate, dir_path=dir_path1
    )
    
    print("\nGROUP 2:")
    group2_filtered, _ = filter_subjects_by_detection_rate(
        group2_subjects, min_detection_rate=min_detection_rate, dir_path=dir_path2
    )
    
    group1_data = load_all_subjects_data(group1_filtered, dir_path1)
    group2_data = load_all_subjects_data(group2_filtered, dir_path2)

    # Create evoked arrays for both groups
    group1_evoked, _, ch_names = prepare_evoked_arrays_for_group(group1_data, metric, normalize)
    group2_evoked, _, _ = prepare_evoked_arrays_for_group(group2_data, metric, normalize)

    # Compute adjacency matrix
    adjacency, _ = compute_adjacency_matrix(group1_evoked)
    
    print(f"DATA PREPARATION COMPLETE")
    print(f"Group 1: N={len(group1_evoked)}, Group 2: N={len(group2_evoked)}, Channels: {len(ch_names)}")
    print(f"Metric: {metric}, Normalized: {normalize}")
    print(f"{'='*80}\n")
    
    return group1_evoked, group2_evoked, adjacency, ch_names, group1_filtered, group2_filtered


# =============================================================================
# STEP 2: STATISTICAL COMPARISON (CLUSTER-BASED PERMUTATION TEST)
# =============================================================================

def prepare_data_arrays_for_clustering(group1_evoked, group2_evoked):
    """
    Convert EvokedArray lists to 3D NumPy arrays for cluster testing.
    
    Parameters:
    -----------
    group1_evoked : list
        List of mne.EvokedArray objects for group 1
    group2_evoked : list
        List of mne.EvokedArray objects for group 2
    
    Returns:
    --------
    X1 : np.ndarray
        Group 1 data array (n_subjects, n_times, n_channels)
    X2 : np.ndarray
        Group 2 data array (n_subjects, n_times, n_channels)
    """
    # Extract data from evoked objects: shape (n_channels, n_times) for each subject
    # Then transpose to (n_times, n_channels) and stack across subjects
    
    X1 = np.array([evoked.data.T for evoked in group1_evoked])  # (n_subjects, n_times, n_channels)
    X2 = np.array([evoked.data.T for evoked in group2_evoked])
    
    print(f"\n📊 Data arrays prepared for clustering:")
    print(f"  Group 1: {X1.shape} (subjects, times, channels)")
    print(f"  Group 2: {X2.shape} (subjects, times, channels)")
    
    return X1, X2


def run_cluster_permutation_test(group1_evoked, group2_evoked, adjacency, 
                                   n_permutations=5000, threshold_p=0.05, 
                                   tail=0, n_jobs=-1):
    """
    Run cluster-based permutation test to compare two groups.
    
    Parameters:
    -----------
    group1_evoked : list
        List of mne.EvokedArray objects for group 1
    group2_evoked : list
        List of mne.EvokedArray objects for group 2
    adjacency : scipy.sparse matrix
        Spatial adjacency matrix
    n_permutations : int
        Number of permutations for building null distribution (default: 5000)
    threshold_p : float
        Threshold p-value for cluster formation (default: 0.05)
    tail : int
        -1 for one-tailed (group1 < group2)
         0 for two-tailed (default)
        +1 for one-tailed (group1 > group2)
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    
    Returns:
    --------
    T_obs : np.ndarray
        Observed t-statistic at each channel/timepoint
    clusters : list
        List of cluster arrays
    cluster_pv : np.ndarray
        P-value for each cluster
    H0 : np.ndarray
        Max cluster statistic distribution under H0 (null)
    """
    print(f"\n{'='*80}")
    print(f"CLUSTER-BASED PERMUTATION TEST")
    print(f"{'='*80}")
    
    # Prepare data arrays
    X1, X2 = prepare_data_arrays_for_clustering(group1_evoked, group2_evoked)
    
    # Set threshold from p-value (for t-test, use t-distribution)
    from scipy import stats
    n1, n2 = len(group1_evoked), len(group2_evoked)
    df = n1 + n2 - 2  # degrees of freedom for independent t-test
    threshold_t = stats.t.ppf(1 - threshold_p / 2, df)  # two-tailed
    
    tail_names = {-1: 'one-tailed (group1 < group2)', 0: 'two-tailed', 1: 'one-tailed (group1 > group2)'}
    
    print(f"\n🔬 Test parameters:")
    print(f"  N permutations: {n_permutations}")
    print(f"  Threshold p-value: {threshold_p}")
    print(f"  Threshold t-statistic: {threshold_t:.3f} (df={df})")
    print(f"  Test tail: {tail_names[tail]}")
    print(f"  Parallel jobs: {n_jobs if n_jobs > 0 else 'all available'}")
    
    print(f"\n⏳ Running permutation test (this may take a few minutes)...")
    
    # Run cluster-based permutation test
    T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
        [X1, X2],
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=threshold_t,
        tail=tail,
        n_jobs=n_jobs,
        buffer_size=None,
        out_type='mask'
    )
    
    print(f"✓ Permutation test complete!")
    print(f"{'='*80}\n")
    
    return T_obs, clusters, cluster_pv, H0


def extract_significant_clusters(T_obs, clusters, cluster_pv, alpha=0.05):
    """
    Extract and summarize significant clusters.
    
    Parameters:
    -----------
    T_obs : np.ndarray
        Observed t-statistic at each channel/timepoint
    clusters : list
        List of cluster arrays (boolean masks)
    cluster_pv : np.ndarray
        P-value for each cluster
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    sig_clusters : list
        List of dictionaries with significant cluster information
    """
    print(f"\n{'='*80}")
    print(f"CLUSTER ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Significance level: α = {alpha}")
    print(f"Total clusters detected: {len(clusters)}")
    
    # Find significant clusters
    sig_cluster_idx = np.where(cluster_pv < alpha)[0]
    
    if len(sig_cluster_idx) == 0:
        print(f"\n❌ No significant clusters found (all p-values > {alpha})")
        print(f"{'='*80}\n")
        return []
    
    print(f"✓ Significant clusters found: {len(sig_cluster_idx)}")
    print(f"\n{'─'*80}")
    
    sig_clusters = []
    for idx in sig_cluster_idx:
        cluster_mask = clusters[idx][0]  # Shape: (n_times, n_channels)
        p_val = cluster_pv[idx]
        
        # Get channels involved in cluster
        channels_in_cluster = np.where(cluster_mask.any(axis=0))[0]
        n_channels = len(channels_in_cluster)
        
        # Get mean t-statistic in cluster
        t_vals_in_cluster = T_obs[cluster_mask]
        mean_t = np.mean(t_vals_in_cluster)
        max_t = np.max(np.abs(t_vals_in_cluster))
        
        cluster_info = {
            'index': idx,
            'p_value': p_val,
            'n_channels': n_channels,
            'channels': channels_in_cluster,
            'mask': cluster_mask,
            'mean_t': mean_t,
            'max_t': max_t
        }
        sig_clusters.append(cluster_info)
        
        print(f"Cluster {idx + 1}:")
        print(f"  p-value: {p_val:.6f}")
        print(f"  Channels involved: {n_channels}")
        print(f"  Mean t-statistic: {mean_t:+.3f}")
        print(f"  Max |t|: {max_t:.3f}")
        print(f"{'─'*80}")
    
    print(f"{'='*80}\n")
    
    return sig_clusters


def run_statistical_comparison(group1_evoked, group2_evoked, adjacency, 
                                 n_permutations=5000, threshold_p=0.05, 
                                 alpha=0.05, tail=0, n_jobs=-1):
    """
    Complete pipeline for statistical comparison between two groups.
    
    Parameters:
    -----------
    group1_evoked : list
        List of mne.EvokedArray objects for group 1
    group2_evoked : list
        List of mne.EvokedArray objects for group 2
    adjacency : scipy.sparse matrix
        Spatial adjacency matrix
    n_permutations : int
        Number of permutations (default: 5000)
    threshold_p : float
        Threshold p-value for cluster formation (default: 0.05)
    alpha : float
        Significance level for cluster p-values (default: 0.05)
    tail : int
        Test tail: -1, 0 (two-tailed), or +1
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'T_obs': observed t-statistics
        - 'clusters': all clusters
        - 'cluster_pv': cluster p-values
        - 'H0': null distribution
        - 'sig_clusters': significant clusters info
    """
    # Run cluster permutation test
    T_obs, clusters, cluster_pv, H0 = run_cluster_permutation_test(
        group1_evoked, group2_evoked, adjacency,
        n_permutations=n_permutations,
        threshold_p=threshold_p,
        tail=tail,
        n_jobs=n_jobs
    )
    
    # Extract significant clusters
    sig_clusters = extract_significant_clusters(T_obs, clusters, cluster_pv, alpha=alpha)
    
    results = {
        'T_obs': T_obs,
        'clusters': clusters,
        'cluster_pv': cluster_pv,
        'H0': H0,
        'sig_clusters': sig_clusters
    }
    
    return results


# =============================================================================
# STEP 3: VISUALIZATION (DIFFERENCE MAPS WITH CLUSTER HIGHLIGHTING)
# =============================================================================

def compute_group_topography(evoked_list, agg='mean'):
    """
    Compute group topography across subjects (mean or median).

    Parameters:
    -----------
    evoked_list : list
        List of mne.EvokedArray objects
    agg : {'mean', 'median'}
        Aggregation across the subject dimension.

    Returns:
    --------
    group_data : np.ndarray
        Per-channel aggregated values across subjects (n_channels,)
    info : mne.Info
        Info object from first subject
    """
    # Stack all subjects' data (n_subjects, n_channels, n_times)
    all_data = np.array([evoked.data for evoked in evoked_list])

    if agg == 'median':
        group_data = np.nanmedian(all_data, axis=0).squeeze()
    else:
        group_data = np.nanmean(all_data, axis=0).squeeze()

    info = evoked_list[0].info
    return group_data, info


# Backwards-compatible alias (other call sites still use the mean-only name)
compute_group_mean_topography = compute_group_topography


def plot_topography_with_clusters(data, info, ax, title, sig_channels=None, 
                                    cmap='RdBu_r', vmin=None, vmax=None):
    """
    Plot topography with optional cluster highlighting.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to plot (n_channels,)
    info : mne.Info
        MNE info object with channel positions
    ax : matplotlib axis
        Axis to plot on
    title : str
        Plot title
    sig_channels : np.ndarray or None
        Boolean mask or indices of significant channels to highlight
    cmap : str
        Colormap name
    vmin, vmax : float or None
        Color scale limits
    
    Returns:
    --------
    im : matplotlib image
        Image object for colorbar
    """
    # Auto-scale if not provided
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # Create mask for significant channels if provided
    mask = None
    mask_params = None
    if sig_channels is not None:
        if sig_channels.dtype == bool:
            mask = sig_channels
        else:
            # Convert indices to boolean mask
            mask = np.zeros(len(data), dtype=bool)
            mask[sig_channels] = True
        
        # Set mask parameters for visualization
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                          linewidth=0, markersize=8, markeredgewidth=1.5)
    
    # Plot the topography
    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax, show=False,
        cmap=cmap, vlim=(vmin, vmax), 
        contours=6,
        mask=mask,
        mask_params=mask_params
    )
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    return im


def create_comparison_figure(group1_subjects, group2_subjects, sig_clusters,
                              metric, output_dir, dir_path1=None, dir_path2=None,
                              group1_name='Group 1', group2_name='Group 2',
                              normalize=True):
    """
    Create comparison figure using step4's averaging approach (average first, then normalize).
    This reuses step4's functions for consistent visualization.
    
    Parameters:
    -----------
    group1_subjects : list
        List of subject IDs for group 1
    group2_subjects : list
        List of subject IDs for group 2
    sig_clusters : list
        List of significant cluster dictionaries
    metric : str
        Metric name (without 'avg_' prefix)
    output_dir : Path
        Directory to save figure
    dir_path1, dir_path2 : str or Path
        Directories containing subject data
    group1_name, group2_name : str
        Names for groups
    normalize : bool
        Whether to normalize data
    """
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATION: {metric.upper()}")
    print(f"{'='*80}")
    
    # Use step4's approach to get group averages (raw data)
    group1_evoked, _, channel_names = create_evoked_from_group_average(
        group1_subjects, metric, dir_path1
    )
    group2_evoked, _, _ = create_evoked_from_group_average(
        group2_subjects, metric, dir_path2
    )
    
    # Extract raw data
    group1_values = group1_evoked.data.squeeze()
    group2_values = group2_evoked.data.squeeze()
    info = group1_evoked.info
    
    # Compute difference (on raw values, will normalize separately for plotting)
    diff_values = group1_values - group2_values
    
    # Get significant channels from clusters
    sig_channels_mask = np.zeros(len(group1_values), dtype=bool)
    if len(sig_clusters) > 0:
        for cluster in sig_clusters:
            sig_channels_mask[cluster['channels']] = True
        n_sig = np.sum(sig_channels_mask)
        print(f"  Significant channels: {n_sig}/{len(sig_channels_mask)}")
    else:
        print(f"  No significant channels to highlight")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define metric info for step4's plotting function
    metric_info = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }
    metric_display = metric_info.get(metric, {'name': metric.upper(), 'unit': 'AU'})
    
    # Plot Group 1 using step4's function (with normalization)
    plot_single_topography(group1_values, info, axes[0], metric_display, normalize=normalize)
    axes[0].set_title(f'{group1_name}\n(N={len(group1_subjects)})', fontsize=11, fontweight='bold')
    
    # Add colorbar for Group 1
    if len(axes[0].images) > 0:
        im1 = axes[0].images[0]
        cbar_label = 'Normalized Value\n(/ mean)' if normalize else f'{metric_display["unit"]}'
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label(cbar_label, fontsize=9)
    
    # Plot Group 2 using step4's function (with normalization)
    plot_single_topography(group2_values, info, axes[1], metric_display, normalize=normalize)
    axes[1].set_title(f'{group2_name}\n(N={len(group2_subjects)})', fontsize=11, fontweight='bold')
    
    # Add colorbar for Group 2
    if len(axes[1].images) > 0:
        im2 = axes[1].images[0]
        cbar_label = 'Normalized Value\n(/ mean)' if normalize else f'{metric_display["unit"]}'
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label(cbar_label, fontsize=9)
    
    # Plot Difference with cluster highlighting
    # For difference, we need to use our custom plotting (step4 doesn't have cluster highlighting)
    diff_title = f'Difference ({group1_name} - {group2_name})'
    if len(sig_clusters) > 0:
        diff_title += f'\n{len(sig_clusters)} significant cluster(s)'
    
    # Normalize difference if requested
    if normalize:
        mean_val = np.nanmean(diff_values)
        std_val = np.nanstd(diff_values)
        if std_val > 0:
            outlier_mask = np.abs(diff_values - mean_val) > 3 * std_val
            if np.any(outlier_mask):
                cleaned_diff = diff_values.copy()
                cleaned_diff[outlier_mask] = np.nan
                clean_mean = np.nanmean(cleaned_diff)
            else:
                clean_mean = mean_val
            if clean_mean != 0:
                plot_diff_values = diff_values / clean_mean
            else:
                plot_diff_values = diff_values
        else:
            if mean_val != 0:
                plot_diff_values = diff_values / mean_val
            else:
                plot_diff_values = diff_values
    else:
        plot_diff_values = diff_values
    
    im3 = plot_topography_with_clusters(
        plot_diff_values, info, axes[2],
        diff_title,
        sig_channels=sig_channels_mask if len(sig_clusters) > 0 else None,
        cmap='RdBu_r'
    )
    
    # Add colorbar only for the difference plot
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Difference', rotation=270, labelpad=15)
    
    # Main title
    fig.suptitle(f'Topographic Comparison: {metric.upper()}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_path = output_dir / f'topo_comparison_{metric}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to: {fig_path}")
    print(f"{'='*80}\n")


def create_tstat_topography(T_obs, info, sig_clusters, metric, output_dir,
                             group1_name='Group 1', group2_name='Group 2'):
    """
    Create topography showing t-statistic values at each electrode.
    Positive t-values (red) indicate group1 > group2.
    Negative t-values (blue) indicate group1 < group2.
    
    Parameters:
    -----------
    T_obs : np.ndarray
        Observed t-statistics (n_times, n_channels) - squeeze to (n_channels,)
    info : mne.Info
        MNE info object with channel positions
    sig_clusters : list
        List of significant cluster dictionaries
    metric : str
        Metric name for title
    output_dir : Path
        Directory to save figure
    group1_name, group2_name : str
        Names for groups
    """
    print(f"\n{'='*80}")
    print(f"CREATING T-STATISTIC TOPOGRAPHY: {metric.upper()}")
    print(f"{'='*80}")
    
    # Squeeze t-statistics to (n_channels,)
    t_values = T_obs.squeeze()
    
    # Get significant channels mask
    sig_channels_mask = np.zeros(len(t_values), dtype=bool)
    if len(sig_clusters) > 0:
        for cluster in sig_clusters:
            sig_channels_mask[cluster['channels']] = True
        n_sig = np.sum(sig_channels_mask)
        print(f"  Significant channels: {n_sig}/{len(sig_channels_mask)}")
    else:
        print(f"  No significant channels")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot t-statistic topography
    # Mask for significant channels
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                      linewidth=0, markersize=10, markeredgewidth=2)
    
    im, _ = mne.viz.plot_topomap(
        t_values, info, axes=ax, show=False,
        cmap='RdBu_r',
        contours=8,
        mask=sig_channels_mask if len(sig_clusters) > 0 else None,
        mask_params=mask_params if len(sig_clusters) > 0 else None
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('t-statistic', rotation=270, labelpad=20, fontsize=11)
    
    # Title with interpretation
    title = f'T-Statistic Topography: {metric.upper()}\n'
    title += f'{group1_name} vs {group2_name}\n'
    title += f'(Red: {group1_name} > {group2_name}, Blue: {group1_name} < {group2_name})'
    if len(sig_clusters) > 0:
        title += f'\nWhite circles: {len(sig_clusters)} significant cluster(s) (p < 0.05)'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_path = output_dir / f'tstat_topography_{metric}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print t-statistic summary
    print(f"\n  T-statistic summary:")
    print(f"    Mean: {np.mean(t_values):+.3f}")
    print(f"    Min: {np.min(t_values):+.3f} (strongest {group2_name} > {group1_name})")
    print(f"    Max: {np.max(t_values):+.3f} (strongest {group1_name} > {group2_name})")
    if len(sig_clusters) > 0:
        sig_t_values = t_values[sig_channels_mask]
        print(f"    Significant channels mean: {np.mean(sig_t_values):+.3f}")
    
    print(f"\n  ✓ Saved to: {fig_path}")
    print(f"{'='*80}\n")


def create_multi_metric_comparison_figure(results_dict, output_dir, 
                                          group1_name='Group 1', group2_name='Group 2'):
    """
    Create multi-row figure comparing multiple metrics (like paper Figure 2C).
    Layout: Rows = metrics (Peak Freq, BW, AUC), Columns = (Group1, Group2, Difference)
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with metric names as keys, each containing:
        - 'group1_evoked': list of EvokedArrays
        - 'group2_evoked': list of EvokedArrays  
        - 'sig_clusters': list of significant clusters
    output_dir : Path
        Directory to save figure
    group1_name, group2_name : str
        Names for groups
    """
    print(f"\n{'='*80}")
    print(f"CREATING MULTI-METRIC COMPARISON FIGURE")
    print(f"{'='*80}")
    
    n_metrics = len(results_dict)
    metric_names = list(results_dict.keys())
    
    # Create figure with grid: rows=metrics, cols=3 (group1, group2, diff)
    fig, axes = plt.subplots(n_metrics, 3, figsize=(15, 5*n_metrics))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    metric_labels = {
        'peak_frequency': 'Peak Frequency (Hz)',
        'bandwidth': 'Bandwidth (Hz)',
        'auc': 'Area Under Curve'
    }
    
    for row_idx, metric in enumerate(metric_names):
        data = results_dict[metric]
        group1_evoked = data['group1_evoked']
        group2_evoked = data['group2_evoked']
        sig_clusters = data['sig_clusters']
        
        print(f"\n  Processing {metric}:")
        print(f"    Group 1: N={len(group1_evoked)}, Group 2: N={len(group2_evoked)}")
        print(f"    Significant clusters: {len(sig_clusters)}")
        
        # Compute group means
        group1_mean, info = compute_group_mean_topography(group1_evoked)
        group2_mean, _ = compute_group_mean_topography(group2_evoked)
        diff_data = group1_mean - group2_mean
        
        # Get significant channels
        sig_channels_mask = np.zeros(len(group1_mean), dtype=bool)
        if len(sig_clusters) > 0:
            for cluster in sig_clusters:
                sig_channels_mask[cluster['channels']] = True
        
        # Determine common color scale
        vmin_groups = min(np.min(group1_mean), np.min(group2_mean))
        vmax_groups = max(np.max(group1_mean), np.max(group2_mean))
        
        # Column 0: Group 1
        title1 = f'{group1_name}' if row_idx > 0 else f'{group1_name}\n(N={len(group1_evoked)})'
        im1 = plot_topography_with_clusters(
            group1_mean, info, axes[row_idx, 0], title1,
            cmap='RdBu_r', vmin=vmin_groups, vmax=vmax_groups
        )
        
        # Column 1: Group 2
        title2 = f'{group2_name}' if row_idx > 0 else f'{group2_name}\n(N={len(group2_evoked)})'
        im2 = plot_topography_with_clusters(
            group2_mean, info, axes[row_idx, 1], title2,
            cmap='RdBu_r', vmin=vmin_groups, vmax=vmax_groups
        )
        
        # Column 2: Difference with clusters
        diff_title = f'Difference' if row_idx > 0 else f'Difference\n({len(sig_clusters)} cluster(s))'
        im3 = plot_topography_with_clusters(
            diff_data, info, axes[row_idx, 2], diff_title,
            sig_channels=sig_channels_mask if len(sig_clusters) > 0 else None,
            cmap='RdBu_r'
        )
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
        cbar2 = plt.colorbar(im2, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        cbar3 = plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
        
        # Add metric label on the left
        metric_label = metric_labels.get(metric, metric)
        axes[row_idx, 0].text(-0.3, 0.5, metric_label, 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold',
                             rotation=90, va='center', ha='center')
    
    # Main title
    fig.suptitle(f'Topographic Comparison: {group1_name} vs {group2_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_path = output_dir / 'topo_comparison_all_metrics.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Multi-metric figure saved to: {fig_path}")
    print(f"{'='*80}\n")


# =============================================================================
# THREE-GROUP TOPOGRAPHIC COMPARISON (ANOVA + CLUSTER PERMUTATION + POST-HOC)
# =============================================================================

def prepare_three_group_data(young_subjects, elderly_subjects, mci_subjects,
                             metric, young_dir, elderly_dir, mci_dir,
                             normalize=True, min_detection_rate=0.2):
    """
    Load and prepare normalized EvokedArray data for all 3 groups.

    Returns:
    --------
    group_evokeds_list : list of 3 lists of EvokedArray
    group_names : list of str
    adjacency : sparse matrix
    available_channels : list of str
    filtered_subjects : dict {group_name: [subject_ids]}
    """
    group_names = ['Young', 'Elderly', 'MCI']
    all_subjects = [young_subjects, elderly_subjects, mci_subjects]
    all_dirs = [young_dir, elderly_dir, mci_dir]

    group_evokeds_list = []
    filtered_subjects = {}

    for name, subjects, dir_path in zip(group_names, all_subjects, all_dirs):
        print(f"\n{name.upper()} GROUP:")
        filt, _ = filter_subjects_by_detection_rate(subjects, min_detection_rate, dir_path)
        filtered_subjects[name] = filt

        data = load_all_subjects_data(filt, dir_path)
        evoked_list, _, available_channels = prepare_evoked_arrays_for_group(
            data, metric, normalize
        )
        group_evokeds_list.append(evoked_list)

    adjacency, _ = compute_adjacency_matrix(group_evokeds_list[0])

    print(f"\nDATA PREPARATION COMPLETE")
    for name, evoked_list in zip(group_names, group_evokeds_list):
        print(f"  {name}: N={len(evoked_list)}")
    print(f"  Channels: {len(available_channels)}")
    print(f"  Metric: {metric}, Normalized: {normalize}")

    return group_evokeds_list, group_names, adjacency, available_channels, filtered_subjects


def electrode_wise_anova(group_evokeds_list, group_names, available_channels):
    """
    Run one-way ANOVA at each electrode across 3 groups (uncorrected).

    Returns:
    --------
    F_values : np.ndarray (n_channels,)
    p_values : np.ndarray (n_channels,)
    """
    # Extract data: list of (n_subjects, n_channels) arrays
    group_arrays = []
    for evoked_list in group_evokeds_list:
        arr = np.array([ev.data.squeeze() for ev in evoked_list])  # (n_subjects, n_channels)
        group_arrays.append(arr)

    n_channels = group_arrays[0].shape[1]
    F_values = np.zeros(n_channels)
    p_values = np.ones(n_channels)

    for ch in range(n_channels):
        samples = [arr[:, ch] for arr in group_arrays]
        F_values[ch], p_values[ch] = f_oneway(*samples)

    n_sig = np.sum(p_values < 0.05)
    print(f"\nELECTRODE-WISE ANOVA (uncorrected)")
    print(f"  Electrodes with p < 0.05: {n_sig}/{n_channels}")
    print(f"  Min p-value: {np.min(p_values):.6f} (electrode {available_channels[np.argmin(p_values)]})")
    print(f"  Max F-value: {np.max(F_values):.4f}")

    return F_values, p_values


def cluster_permutation_anova(group_evokeds_list, adjacency,
                              n_permutations=5000, threshold_p=0.05, n_jobs=-1):
    """
    Run cluster-based permutation test with F-statistic for 3-group comparison.

    Returns:
    --------
    F_obs, clusters, cluster_pv, H0
    """
    # Build data arrays: each (n_subjects, 1, n_channels)
    X_list = []
    for evoked_list in group_evokeds_list:
        X = np.array([ev.data.T for ev in evoked_list])  # (n_subjects, 1, n_channels)
        X_list.append(X)

    # F-critical threshold for cluster formation
    n_groups = len(X_list)
    n_total = sum(X.shape[0] for X in X_list)
    dfn = n_groups - 1
    dfd = n_total - n_groups
    f_threshold = f_dist.ppf(1 - threshold_p, dfn, dfd)

    print(f"\nCLUSTER-BASED PERMUTATION TEST (F-test, {n_groups} groups)")
    print(f"  N total: {n_total} (dfn={dfn}, dfd={dfd})")
    print(f"  F threshold: {f_threshold:.3f} (p={threshold_p})")
    print(f"  Permutations: {n_permutations}")
    print(f"  Running...")

    F_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
        X_list,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=f_threshold,
        tail=1,  # F-test is one-tailed
        n_jobs=n_jobs,
        buffer_size=None,
        out_type='mask'
    )

    print(f"  Done! Total clusters: {len(clusters)}")
    return F_obs, clusters, cluster_pv, H0


def extract_significant_clusters_f(F_obs, clusters, cluster_pv, available_channels, alpha=0.05):
    """
    Extract significant clusters from F-test permutation results.

    Returns:
    --------
    sig_clusters : list of dicts with 'channels', 'p_value', etc.
    sig_channel_indices : np.ndarray of all unique significant channel indices
    """
    print(f"\nCLUSTER RESULTS (alpha={alpha})")
    print(f"  Total clusters: {len(clusters)}")

    sig_idx = np.where(cluster_pv < alpha)[0]

    if len(sig_idx) == 0:
        print(f"  No significant clusters found")
        return [], np.array([], dtype=int)

    print(f"  Significant clusters: {len(sig_idx)}")

    sig_clusters = []
    all_sig_channels = set()

    for i in sig_idx:
        mask = clusters[i]
        if isinstance(mask, tuple):
            mask = mask[0]
        channels = np.where(mask.any(axis=0))[0]
        f_vals = F_obs.squeeze()[channels]
        p_val = cluster_pv[i]

        sig_clusters.append({
            'index': i,
            'p_value': p_val,
            'channels': channels,
            'n_channels': len(channels),
            'mean_F': np.mean(f_vals),
            'max_F': np.max(f_vals),
        })
        all_sig_channels.update(channels)

        ch_names = [available_channels[c] for c in channels[:10]]
        print(f"\n  Cluster {i+1}: p={p_val:.6f}, {len(channels)} electrodes, "
              f"mean F={np.mean(f_vals):.3f}, max F={np.max(f_vals):.3f}")
        print(f"    Electrodes: {ch_names}{'...' if len(channels) > 10 else ''}")

    return sig_clusters, np.array(sorted(all_sig_channels), dtype=int)


def posthoc_tukey_at_clusters(group_evokeds_list, group_names, sig_channel_indices,
                              available_channels, alpha=0.05):
    """
    Run Tukey-Kramer post-hoc at each electrode in significant clusters.

    Returns:
    --------
    posthoc_results : dict {(g1, g2): np.ndarray of significant channel indices}
    """
    if len(sig_channel_indices) == 0:
        return {(group_names[i], group_names[j]): np.array([], dtype=int)
                for i in range(len(group_names)) for j in range(i+1, len(group_names))}

    # Extract data arrays
    group_arrays = []
    for evoked_list in group_evokeds_list:
        arr = np.array([ev.data.squeeze() for ev in evoked_list])
        group_arrays.append(arr)

    # Initialize results for each pair
    pairs = [(group_names[i], group_names[j])
             for i in range(len(group_names)) for j in range(i+1, len(group_names))]
    pair_sig_channels = {pair: [] for pair in pairs}

    print(f"\nPOST-HOC TUKEY-KRAMER AT CLUSTER ELECTRODES")
    print(f"  Testing {len(sig_channel_indices)} electrodes across {len(pairs)} pairs")

    for ch_idx in sig_channel_indices:
        # Build combined values + labels
        all_vals = []
        all_labels = []
        for g_idx, name in enumerate(group_names):
            vals = group_arrays[g_idx][:, ch_idx]
            all_vals.extend(vals)
            all_labels.extend([name] * len(vals))

        tukey = pairwise_tukeyhsd(np.array(all_vals), np.array(all_labels), alpha=alpha)

        # Extract which pairs reject H0
        for k in range(len(tukey.reject)):
            if tukey.reject[k]:
                g1 = str(tukey.groupsunique[tukey._multicomp.pairindices[0][k]])
                g2 = str(tukey.groupsunique[tukey._multicomp.pairindices[1][k]])
                pair = (g1, g2) if (g1, g2) in pair_sig_channels else (g2, g1)
                if pair in pair_sig_channels:
                    pair_sig_channels[pair].append(ch_idx)

    # Convert to arrays and print summary
    for pair in pairs:
        pair_sig_channels[pair] = np.array(sorted(set(pair_sig_channels[pair])), dtype=int)
        n = len(pair_sig_channels[pair])
        print(f"  {pair[0]} vs {pair[1]}: {n} significant electrodes")
        if n > 0 and n <= 15:
            ch_names = [available_channels[c] for c in pair_sig_channels[pair]]
            print(f"    Electrodes: {ch_names}")

    return pair_sig_channels


def plot_three_group_topos(group_evokeds_list, group_names, metric, info,
                           posthoc_results, F_obs, sig_channel_indices,
                           output_dir, clusters=None, cluster_pv=None,
                           agg='mean', show_roi=True):
    """
    Plot normalized grand-average topographies for all 3 groups.
    Overlay black circles on electrodes with significant post-hoc differences.
    Also plot F-statistic topography.

    ``agg`` controls subject-dimension aggregation of the DISPLAYED topography
    ('mean' or 'median'). Statistics (ANOVA / cluster / post-hoc) are
    mean-based and unaffected.

    ``show_roi`` (AUC only) toggles the green ROI overlay. When False, the
    file is saved with a ``_no_roi`` suffix and the F-stat figure is skipped
    to avoid duplicating it.
    """
    metric_info = {
        'peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'auc': {'name': 'Area Under Curve', 'unit': 'AU'}
    }.get(metric, {'name': metric, 'unit': 'AU'})

    # Compute grand averages per group
    group_means = []
    for evoked_list in group_evokeds_list:
        mean_data, _ = compute_group_topography(evoked_list, agg=agg)
        group_means.append(mean_data)

    # Common color scale across groups
    vmin = min(np.nanmin(m) for m in group_means)
    vmax = max(np.nanmax(m) for m in group_means)

    # Build per-group mask: union of all post-hoc pairs involving that group
    group_masks = {}
    for g_idx, name in enumerate(group_names):
        mask = np.zeros(len(group_means[0]), dtype=bool)
        for (g1, g2), channels in posthoc_results.items():
            if name in (g1, g2) and len(channels) > 0:
                mask[channels] = True
        group_masks[name] = mask

    # --- Figure 1: 3 group topos ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    is_auc = metric == 'auc'
    draw_roi = is_auc and show_roi
    roi_subtitle = ' | Green dots = ROI' if draw_roi else ''
    agg_tag = f' [{agg}]' if agg != 'mean' else ''
    fig.suptitle(f'Normalized Topographies: {metric_info["name"]}{agg_tag}\n'
                 f'(Black circles = significant post-hoc difference){roi_subtitle}',
                 fontsize=14, fontweight='bold')

    n_channels = len(group_means[0])
    for g_idx, name in enumerate(group_names):
        ax = axes[g_idx]
        data = group_means[g_idx]
        mask = group_masks[name] if np.any(group_masks[name]) else None

        mask_params = dict(marker='o', markerfacecolor='none', markeredgecolor='black',
                           linewidth=0, markersize=10, markeredgewidth=2) if mask is not None else None

        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            cmap='RdBu_r', vlim=(vmin, vmax), contours=6,
            mask=mask, mask_params=mask_params
        )

        if draw_roi:
            _overlay_roi_markers(ax, info, n_channels)

        n_subjects = len(group_evokeds_list[g_idx])
        ax.set_title(f'{name}\n(N={n_subjects})', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized', fontsize=9)

    # Add post-hoc legend text
    legend_lines = []
    for (g1, g2), channels in posthoc_results.items():
        if len(channels) > 0:
            legend_lines.append(f'{g1} vs {g2}: {len(channels)} electrodes')
    if legend_lines:
        fig.text(0.5, 0.01, 'Significant pairs: ' + ' | '.join(legend_lines),
                 ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    suffix_parts = []
    if agg != 'mean':
        suffix_parts.append(agg)
    if is_auc and not show_roi:
        suffix_parts.append('no_roi')
    suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''
    fig_path = output_dir / f'three_group_topo_{metric}{suffix}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # F-statistic figure is mean-based and ROI-independent; only emit it once
    # (on the default mean + show_roi pass) to avoid duplicates.
    if agg != 'mean' or not show_roi:
        return

    # --- Figure 2: F-statistic topography with all clusters ---
    f_values = F_obs.squeeze()

    # Collect all clusters (before permutation significance) for visualization
    all_cluster_channels = []
    if clusters is not None:
        for i, cl in enumerate(clusters):
            mask_cl = cl
            if isinstance(mask_cl, tuple):
                mask_cl = mask_cl[0]
            ch_idx = np.where(mask_cl.any(axis=0))[0]
            p_val = cluster_pv[i] if cluster_pv is not None else 1.0
            all_cluster_channels.append((ch_idx, p_val))

    n_clusters = len(all_cluster_channels)
    sig_count = sum(1 for _, p in all_cluster_channels if p < 0.05)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))

    # Plot F-statistic topography with RdBu_r colormap
    im_f, _ = mne.viz.plot_topomap(
        f_values, info, axes=ax2, show=False,
        cmap='RdBu_r', contours=6
    )

    cbar_f = plt.colorbar(im_f, ax=ax2, fraction=0.046, pad=0.04)
    cbar_f.set_label('F-statistic', fontsize=11)

    # Overlay all clusters as enclosing circles
    if n_clusters > 0:
        from matplotlib.patches import Circle
        from scipy.spatial.distance import cdist
        # Extract plotted 2D positions from the topomap axes (same coordinates MNE used)
        # The first PathCollection in the axes contains the sensor dots
        pos = None
        for child in ax2.get_children():
            if isinstance(child, plt.matplotlib.collections.PathCollection):
                offsets = child.get_offsets()
                if len(offsets) == len(f_values):
                    pos = offsets.copy()
                    break
        if pos is None:
            # Fallback: project raw 3D positions to 2D the way MNE does
            from mne.viz.topomap import _find_topomap_coords
            pos = _find_topomap_coords(info, picks=range(len(info['chs'])))

        # Padding: half the nearest-neighbor distance
        nn_dists = np.sort(cdist(pos, pos), axis=1)[:, 1]  # nearest neighbor per electrode
        pad = np.median(nn_dists) * 0.5

        cluster_cmap = plt.cm.tab10
        for cl_idx, (ch_idx, p_val) in enumerate(all_cluster_channels):
            if len(ch_idx) == 0:
                continue
            color = cluster_cmap(cl_idx % 10)
            lw = 2.5 if p_val < 0.05 else 1.5
            cluster_pos = pos[ch_idx]
            centroid = cluster_pos.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_pos - centroid, axis=1)) + pad
            circle = Circle(centroid, radius, facecolor='none', edgecolor=color,
                           linewidth=lw, zorder=5)
            ax2.add_patch(circle)

    title = f'F-statistic Topography: {metric_info["name"]}\nOne-Way ANOVA across 3 groups'
    title += f'\n{n_clusters} clusters found ({sig_count} significant after permutation)'
    ax2.set_title(title, fontsize=12, fontweight='bold')

    # Add legend for clusters
    if n_clusters > 0:
        from matplotlib.patches import Patch
        legend_handles = []
        for cl_idx, (ch_idx, p_val) in enumerate(all_cluster_channels):
            if len(ch_idx) == 0:
                continue
            color = cluster_cmap(cl_idx % 10)
            sig_label = " *" if p_val < 0.05 else ""
            legend_handles.append(
                Patch(facecolor='none', edgecolor=color,
                      linewidth=2.5 if p_val < 0.05 else 1.5,
                      label=f'Cluster {cl_idx+1}: {len(ch_idx)} ch, p={p_val:.3f}{sig_label}')
            )
        if legend_handles:
            ax2.legend(handles=legend_handles, loc='lower center',
                      bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(legend_handles)),
                      fontsize=8, frameon=True)

    plt.tight_layout()
    fig_path_f = output_dir / f'three_group_fstat_topo_{metric}.png'
    plt.savefig(fig_path_f, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path_f}")


def _overlay_roi_markers(ax, info, n_channels):
    """Overlay ROI (filled green dots) on a topomap axis.

    Uses the extended central-parietal ROI as the single displayed ROI.
    The core / extended lists are kept separate in config so we can fall
    back to distinguishing them, but visually they are treated as one ROI.
    """
    ch_names = [info['chs'][i]['ch_name'] for i in range(len(info['chs']))]
    roi_set = set(EXTENDED_CENTRAL_PARIETAL_ROI)
    roi_idx = [i for i, ch in enumerate(ch_names) if ch in roi_set]

    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks=range(len(info['chs'])))

    if roi_idx:
        ax.scatter(pos[roi_idx, 0], pos[roi_idx, 1],
                   s=28, c='#00aa00', marker='o',
                   edgecolors='black', linewidths=0.6, zorder=4)


def _draw_roi_ellipse(ax, info, roi_channels, n_channels):
    """Draw a covariance-fitted ellipse around ROI electrodes on a topomap axis."""
    from matplotlib.patches import Ellipse
    from scipy.spatial.distance import cdist

    # Find ROI channel indices in the info object
    ch_names = [info['chs'][i]['ch_name'] for i in range(len(info['chs']))]
    roi_idx = [i for i, ch in enumerate(ch_names) if ch in roi_channels]
    if len(roi_idx) == 0:
        return

    # Extract 2D positions from the topomap axes (same coordinates MNE used)
    pos = None
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.collections.PathCollection):
            offsets = child.get_offsets()
            if len(offsets) == n_channels:
                pos = offsets.copy()
                break
    if pos is None:
        from mne.viz.topomap import _find_topomap_coords
        pos = _find_topomap_coords(info, picks=range(len(info['chs'])))

    roi_pos = pos[roi_idx]
    centroid = roi_pos.mean(axis=0)

    # Fit ellipse via covariance matrix
    cov = np.cov(roi_pos, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by largest eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Padding: half the median nearest-neighbor distance
    nn_dists = np.sort(cdist(pos, pos), axis=1)[:, 1]
    pad = np.median(nn_dists) * 0.6

    # Ellipse axes: ~2.5 std covers the outermost electrodes, plus padding
    scale = 2.5
    width = 2 * (np.sqrt(eigenvalues[0]) * scale + pad)
    height = 2 * (np.sqrt(eigenvalues[1]) * scale + pad)

    # Rotation angle from first eigenvector
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(centroid, width, height, angle=angle,
                      facecolor='none', edgecolor='black',
                      linewidth=2.5, linestyle='--', zorder=5)
    ax.add_patch(ellipse)


def plot_three_group_topos_roi(group_evokeds_list, group_names, info, output_dir):
    """
    Plot normalized grand-average AUC topographies for all 3 groups
    with the central-parietal ROI highlighted as a dashed circle.
    """
    metric_info = {'name': 'Area Under Curve', 'unit': 'AU'}

    # Compute grand averages per group
    group_means = []
    for evoked_list in group_evokeds_list:
        mean_data, _ = compute_group_mean_topography(evoked_list)
        group_means.append(mean_data)

    # Common color scale across groups
    vmin = min(np.nanmin(m) for m in group_means)
    vmax = max(np.nanmax(m) for m in group_means)

    n_channels = len(group_means[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Normalized Topographies: {metric_info["name"]} (ROI)\n'
                 f'(Dashed ellipse = central-parietal ROI)',
                 fontsize=14, fontweight='bold')

    for g_idx, name in enumerate(group_names):
        ax = axes[g_idx]
        data = group_means[g_idx]

        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            cmap='RdBu_r', vlim=(vmin, vmax), contours=6,
        )

        n_subjects = len(group_evokeds_list[g_idx])
        ax.set_title(f'{name}\n(N={n_subjects})', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized', fontsize=9)

        # Draw ROI ellipse
        _draw_roi_ellipse(ax, info, set(CENTRAL_PARIETAL_ROI), n_channels)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_path = output_dir / 'three_group_topo_auc_ROI.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


def main_three_group():
    """Run 3-group topographic comparison: electrode-wise ANOVA + cluster permutation + post-hoc."""
    # Load subjects
    young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    young_dir = Path("results/new_iso_results")
    young_subjects = [s for s in young_subjects if (young_dir / s).exists() and s != "dashboards"]

    elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
    elderly_dir = Path("results/new_elderly_results")
    elderly_subjects = [s for s in elderly_subjects if (elderly_dir / s).exists() and s != "dashboards"]

    mci_subjects = get_all_subjects(f"{BASE_DIR}/MCI_clean/")
    mci_dir = Path("results/new_MCI_results")
    mci_subjects = [s for s in mci_subjects if (mci_dir / s).exists() and s != "dashboards"]

    output_dir = Path("results/group_comparison_results/three_groups_V3")
    output_dir.mkdir(exist_ok=True, parents=True)

    metrics = ['peak_frequency', 'bandwidth', 'auc']
    report_path = output_dir / "three_group_topo_statistics.txt"

    print(f"Running 3-group topographic comparison...")
    print(f"  Young: {len(young_subjects)}, Elderly: {len(elderly_subjects)}, MCI: {len(mci_subjects)}")
    print(f"  Output: {output_dir}")

    # Run analysis for all metrics, store results
    all_results = {}

    with open(report_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print("################################################################################")
            print("  THREE-GROUP TOPOGRAPHIC COMPARISON")
            print("################################################################################\n")

            for metric in metrics:
                print(f"\n{'#'*80}")
                print(f"  METRIC: {metric.upper()}")
                print(f"{'#'*80}\n")

                # Step 1-4: Prepare normalized data
                group_evokeds_list, group_names, adjacency, available_channels, filtered_subjects = \
                    prepare_three_group_data(
                        young_subjects, elderly_subjects, mci_subjects,
                        metric, young_dir, elderly_dir, mci_dir,
                        normalize=True
                    )

                # Step 5: Electrode-wise ANOVA (uncorrected, descriptive)
                F_values, p_values = electrode_wise_anova(
                    group_evokeds_list, group_names, available_channels
                )

                # Step 6: Cluster permutation test
                F_obs, clusters, cluster_pv, H0 = cluster_permutation_anova(
                    group_evokeds_list, adjacency,
                    n_permutations=5000, threshold_p=0.05, n_jobs=-1
                )

                sig_clusters, sig_channel_indices = extract_significant_clusters_f(
                    F_obs, clusters, cluster_pv, available_channels
                )

                # Step 7: Post-hoc Tukey at cluster electrodes
                posthoc_results = posthoc_tukey_at_clusters(
                    group_evokeds_list, group_names, sig_channel_indices, available_channels
                )

                # Print post-hoc summary
                print(f"\n{'='*80}")
                print(f"SUMMARY: {metric.upper()}")
                print(f"{'='*80}")
                print(f"  Significant clusters: {len(sig_clusters)}")
                for pair, channels in posthoc_results.items():
                    print(f"  {pair[0]} vs {pair[1]}: {len(channels)} significant electrodes")

                # Store for plotting
                all_results[metric] = {
                    'group_evokeds_list': group_evokeds_list,
                    'group_names': group_names,
                    'F_obs': F_obs,
                    'sig_channel_indices': sig_channel_indices,
                    'posthoc_results': posthoc_results,
                    'clusters': clusters,
                    'cluster_pv': cluster_pv,
                }

            print(f"\n{'#'*80}")
            print(f"  ANALYSIS COMPLETE")
            print(f"{'#'*80}")

    print(f"Statistics report saved to: {report_path}")

    # Plot using stored results (no recomputation).
    # Render both mean and median group topographies; stats are mean-based either way.
    for metric in metrics:
        r = all_results[metric]
        info = r['group_evokeds_list'][0][0].info
        for agg in ('mean', 'median'):
            print(f"\nPlotting {metric} ({agg})...")
            plot_three_group_topos(
                r['group_evokeds_list'], r['group_names'], metric, info,
                r['posthoc_results'], r['F_obs'], r['sig_channel_indices'], output_dir,
                clusters=r['clusters'], cluster_pv=r['cluster_pv'],
                agg=agg,
            )

    # Extra AUC mean topo without the ROI overlay (keeps sig dots + color scale).
    if 'auc' in all_results:
        r = all_results['auc']
        info = r['group_evokeds_list'][0][0].info
        print(f"\nPlotting auc (mean, no ROI)...")
        plot_three_group_topos(
            r['group_evokeds_list'], r['group_names'], 'auc', info,
            r['posthoc_results'], r['F_obs'], r['sig_channel_indices'], output_dir,
            clusters=r['clusters'], cluster_pv=r['cluster_pv'],
            agg='mean', show_roi=False,
        )

    # Plot AUC topo with ROI circle overlay (disabled for V2)
    # if 'auc' in all_results:
    #     print(f"\nPlotting AUC with ROI overlay...")
    #     r = all_results['auc']
    #     info = r['group_evokeds_list'][0][0].info
    #     plot_three_group_topos_roi(
    #         r['group_evokeds_list'], r['group_names'], info, output_dir
    #     )

    print(f"\nAll done! Results in {output_dir}")


def main():
    """Run full pipeline for all three metrics: peak_frequency, bandwidth, and auc."""
    young_subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    young_dir = Path("new_iso_results")
    young_subjects = [sub for sub in young_subjects if (young_dir / sub).exists() and sub != "dashboards"]
    
    elderly_subjects = get_all_subjects(f"{BASE_DIR}/elderly_control_clean/")
    elderly_dir = Path("new_elderly_results")
    elderly_subjects = [sub for sub in elderly_subjects if (elderly_dir / sub).exists()]
    
    # Output directory
    output_dir = Path("group_comparison_results")
    
    # Parameters
    metrics = ['peak_frequency', 'bandwidth', 'auc']
    normalize = True
    
    print(f"\n{'='*80}")
    print(f"RUNNING TOPOGRAPHIC COMPARISON FOR ALL METRICS")
    print(f"{'='*80}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Young subjects: {len(young_subjects)}")
    print(f"Elderly subjects: {len(elderly_subjects)}")
    print(f"Normalization: {normalize}")
    print(f"{'='*80}\n")
    
    results_dict = {}
    
    for metric in metrics:
        print(f"\n{'#'*80}")
        print(f"PROCESSING METRIC: {metric.upper()}")
        print(f"{'#'*80}\n")
        
        # Step 1: Prepare data
        group1_evoked, group2_evoked, adjacency, ch_names, young_filtered, elderly_filtered = prepare_data_for_comparison(
            young_subjects,
            elderly_subjects,
            metric=metric,
            dir_path1=young_dir,
            dir_path2=elderly_dir,
            normalize=normalize,
        )
        
        # Step 2: Statistical comparison
        results = run_statistical_comparison(
            group1_evoked,
            group2_evoked,
            adjacency,
            n_permutations=1000,  # Use 1000 for testing (5000+ for final analysis)
            threshold_p=0.05,
            alpha=0.05,
            tail=0,  # two-tailed test
            n_jobs=-1
        )
        
        # Step 3: Visualization (using step4's averaging approach for visual consistency)
        # Use the FILTERED subject lists to match the statistical analysis
        create_comparison_figure(
            young_filtered,
            elderly_filtered,
            results['sig_clusters'],
            metric=metric,
            output_dir=output_dir,
            dir_path1=young_dir,
            dir_path2=elderly_dir,
            group1_name='Young',
            group2_name='Elderly',
            normalize=normalize
        )
        
        # Step 4: T-statistic topography
        info = group1_evoked[0].info  # Get info from first subject
        create_tstat_topography(
            results['T_obs'],
            info,
            results['sig_clusters'],
            metric=metric,
            output_dir=output_dir,
            group1_name='Young',
            group2_name='Elderly'
        )
        
        # Store results for multi-metric figure
        results_dict[metric] = {
            'group1_evoked': group1_evoked,
            'group2_evoked': group2_evoked,
            'sig_clusters': results['sig_clusters']
        }
        
        print(f"\n✓ {metric.upper()} completed - Significant clusters: {len(results['sig_clusters'])}")
    
    print(f"\n{'='*80}")
    print(f"ALL METRICS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    for metric in metrics:
        n_clusters = len(results_dict[metric]['sig_clusters'])
        print(f"  {metric}: {n_clusters} significant cluster(s)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main_three_group()
