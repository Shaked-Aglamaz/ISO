import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import mne
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test

from utils.config import BASE_DIR
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
    plot_single_topography
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
    
    # Check for missing data
    if np.any(np.isnan(values)):
        n_missing = np.sum(np.isnan(values))
        # Replace NaN with channel mean (simple imputation)
        channel_mean = np.nanmean(values)
        values = np.where(np.isnan(values), channel_mean, values)
    
    # Optional normalization (divide by subject mean to preserve spatial pattern)
    # Match step4's approach: detect outliers, compute mean without outliers, but normalize ALL values
    if normalize:
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            # Detect outliers (like step4)
            outlier_mask = np.abs(values - mean_val) > 3 * std_val
            
            # Compute mean excluding outliers
            if np.any(outlier_mask):
                cleaned_values = values.copy()
                cleaned_values[outlier_mask] = np.nan
                clean_mean = np.nanmean(cleaned_values)
            else:
                clean_mean = mean_val
            
            # Normalize ALL values by the cleaned mean (don't set outliers to NaN)
            if clean_mean > 0:
                values = values / clean_mean
            else:
                print(f"  ⚠ {subject_id}: Cannot normalize {metric} (cleaned mean=0)")
        else:
            # If std=0, all values are the same, just divide by mean
            if mean_val > 0:
                values = values / mean_val
            else:
                print(f"  ⚠ {subject_id}: Cannot normalize {metric} (mean=0)")
    
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

def compute_group_mean_topography(evoked_list):
    """
    Compute mean topography across subjects in a group.
    
    Parameters:
    -----------
    evoked_list : list
        List of mne.EvokedArray objects
    
    Returns:
    --------
    mean_data : np.ndarray
        Mean values across subjects (n_channels,)
    info : mne.Info
        Info object from first subject
    """
    # Stack all subjects' data (n_subjects, n_channels, n_times)
    all_data = np.array([evoked.data for evoked in evoked_list])
    
    # Compute mean across subjects and squeeze time dimension
    mean_data = np.mean(all_data, axis=0).squeeze()  # (n_channels,)
    
    # Get info from first subject
    info = evoked_list[0].info
    
    return mean_data, info


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
    main()
