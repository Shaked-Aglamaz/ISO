import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from pathlib import Path
from scipy.interpolate import interp1d
from config import BASE_DIR
from step3_spectral import get_all_subjects
from scipy.ndimage import uniform_filter1d

def load_subject_data(subject_id):
    """Load channel summary data for a specific subject."""
    summary_file = f"{subject_id}/{subject_id}_all_channels_summary.csv"
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found for {subject_id}: {summary_file}")
        return None
    
    try:
        df = pd.read_csv(summary_file)
        df['subject'] = subject_id  # Add subject ID column
        return df
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return None


def collect_all_data(subjects, target_channel='VREF'):
    """
    Collect spectral data from all subjects for the target channel.
    Combines data from individual subject CSV files into one dataset.
    """
    all_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject)
        if subject_data is not None:
            channel_data = subject_data[subject_data['channel'] == target_channel]
            if not channel_data.empty:
                all_data.append(channel_data)
            else:
                print(f"No {target_channel} data found for subject {subject}")
        else:
            print(f"Could not load data for subject {subject}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records: {len(combined_df)}")
        return combined_df
    else:
        print(f"No data collected for channel {target_channel}")
        return None


def calculate_optimal_bins(data):
    """
    Calculate optimal number of bins for histogram based on data size and distribution.
    Uses Freedman-Diaconis rule with reasonable limits for small datasets.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))  # Freedman-Diaconis rule
    
    # For small datasets, use a more conservative approach
    # Rule of thumb: roughly sqrt(n) to n/3 bins for small n
    max_reasonable_bins = min(int(np.sqrt(len(data))) + 5, len(data) // 2)
    n_bins = max(8, min(max_reasonable_bins, int((data.max() - data.min()) / bin_width))) if bin_width > 0 else max_reasonable_bins
    
    return n_bins


def plot_distributions(df, channel_name='VREF'):
    """
    Create individual distribution plots for each spectral metric.
    Generates fine-grained histograms with KDE curves for peak frequency, bandwidth, and AUC.
    Includes statistical annotations and saves each plot separately.
    """
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    
    for metric, info in metrics.items():
        # Create individual figure for each metric
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data = df[metric].dropna()
        
        # Calculate optimal number of bins using consistent method
        n_bins = calculate_optimal_bins(data)
        
        # Create histogram with calculated bins for consistent detail
        sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
        
        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        # Add statistical lines
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.3f}')
        
        # Formatting
        ax.set_title(f'{info["name"]} Distribution - {channel_name} Channel\n(N={len(data)} subjects)', fontsize=14)
        ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}\nBins = {n_bins}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = output_dir / f"{channel_name}_{metric}_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
    print(f"\nAll individual distribution plots saved to: {output_dir}/")


def plot_combined_distributions(df, channel_name='VREF'):
    """
    Create a combined plot with all distributions in subplots (optional).
    """
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Combined Distribution Analysis - {channel_name} Channel\n(N={len(df)} subjects)', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = df[metric].dropna()
        
        # Calculate optimal number of bins using same method as individual plots
        n_bins = calculate_optimal_bins(data)
        
        sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins)
        
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
        
        ax.set_title(f'{info["name"]} Distribution')
        ax.set_xlabel(f'{info["name"]} ({info["unit"]})')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{channel_name}_combined_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined distribution plot saved to: {plot_path}")


def plot_boxplots_with_dots(df, channel_name='VREF'):
    """
    Create a combined plot with three boxplots (frequency, bandwidth, AUC) 
    with individual subject dots overlaid.
    """
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'Spectral Parameters Distribution - {channel_name} Channel\n(N={len(df)} subjects)', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = df[metric].dropna()
        
        # Create boxplot
        box_plot = ax.boxplot(data, patch_artist=True, widths=0.6)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Add individual subject dots
        np.random.seed(42)  # For reproducible jitter
        x_jitter = np.random.normal(1, 0.04, size=len(data))  # Small jitter around x=1
        ax.scatter(x_jitter, data, alpha=0.6, color='red', s=30, zorder=3, label='Subjects')
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_xticklabels([''])  # Remove x-axis labels since we only have one box
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    boxplot_path = output_dir / f"{channel_name}_boxplots_with_dots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot with subject dots saved to: {boxplot_path}")


def load_and_combine_subject_data(subjects):
    """
    Load data from all subjects and combine into a single DataFrame.
    """
    all_subject_data = []
    for subject in subjects:
        subject_data = load_subject_data(subject)
        if subject_data is not None:
            all_subject_data.append(subject_data)
    
    if not all_subject_data:
        print("No subject data available for topography")
        return None
    
    return pd.concat(all_subject_data, ignore_index=True)


def filter_eeg_channels(combined_data):
    """
    Filter out non-EEG channels and calculate channel averages across subjects.
    """
    # Hardcoded specific channel exclusions for problematic channels
    specific_exclusions = {
        'EL3006': ['E202', 'E88'],
        'EL3005': ['E9']
    }
    
    excluded_patterns = ['EMG', 'EOG', 'ECG']
    eeg_data = combined_data[~combined_data['channel'].str.contains('|'.join(excluded_patterns), case=False, na=False)]
    
    # Apply hardcoded specific exclusions
    for subject, excluded_channels in specific_exclusions.items():
        if subject in combined_data['subject'].values:
            eeg_data = eeg_data[~((eeg_data['subject'] == subject) & (eeg_data['channel'].isin(excluded_channels)))]
            print(f"  ℹ Topography: Excluded specific channels {excluded_channels} for subject {subject}")
    
    if len(eeg_data) == 0:
        print("No EEG channels found for topography")
        return None
    
    # Calculate average values across all subjects for each channel
    channel_averages = eeg_data.groupby('channel')[['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']].mean().reset_index()
    return channel_averages


def create_electrode_montage(channel_names):
    """
    Create MNE montage and filter channels to those with known positions.
    """
    try:
        montage = mne.channels.make_standard_montage('EGI_256')
        # Filter montage to only include channels we have
        available_channels = [ch for ch in channel_names if ch in montage.ch_names]
        
        if len(available_channels) < 10:  # Need minimum channels for meaningful topography
            print(f"Not enough standard electrode positions found ({len(available_channels)})")
            print("Available channels:", available_channels[:10], "...")
            return None, None
            
        return montage, available_channels
        
    except Exception as e:
        print(f"Could not create montage: {e}")
        return None, None


def setup_electrode_info(channel_averages):
    """
    Create electrode montage, filter data to known positions, and create MNE info object.
    """
    # Get channel names
    channel_names = channel_averages['channel'].tolist()
    
    montage, available_channels = create_electrode_montage(channel_names)
    if montage is None:
        return None, None, None
    
    # Filter data to channels with known positions
    channel_averages_filtered = channel_averages[channel_averages['channel'].isin(available_channels)]
    channel_names_filtered = available_channels
    
    # Create MNE info object
    info = mne.create_info(ch_names=channel_names_filtered, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    
    return info, channel_averages_filtered, channel_names_filtered


def create_mne_info(channel_names, montage):
    """
    Create MNE Info object with electrode positions.
    
    Args:
        channel_names (list): List of channel names
        montage: MNE montage object
        
    Returns:
        mne.Info: Info object for topographical plotting
    """
    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    return info


def extract_channel_values(channel_averages, channel_names, metric):
    """
    Extract values for a specific metric from channel averages.
    
    Args:
        channel_averages (pd.DataFrame): Averaged values per channel
        channel_names (list): List of channel names
        metric (str): Metric name to extract
        
    Returns:
        np.array: Values for the specified metric
    """
    values = []
    for ch in channel_names:
        ch_data = channel_averages[channel_averages['channel'] == ch]
        if not ch_data.empty:
            values.append(ch_data[metric].iloc[0])
        else:
            values.append(np.nan)
    
    return np.array(values)


def plot_single_topography(values, info, ax, metric_info):
    """
    Plot a single topographical map with normalization and colorbar.
    
    Args:
        values (np.array): Values to plot
        info (mne.Info): MNE info object
        ax: Matplotlib axis
        metric_info (dict): Dictionary with 'name' and 'unit' keys
        
    Returns:
        float: Mean value used for normalization
    """
    valid_mask = ~np.isnan(values)
    
    if np.sum(valid_mask) == 0:
        print(f"No valid data for {metric_info['name']}")
        return None
        
    # Normalize by mean across all electrodes
    mean_value = np.nanmean(values)
    normalized_values = values / mean_value
    
    # Determine color scale limits based on normalized data range
    vmin = np.nanmin(normalized_values)
    vmax = np.nanmax(normalized_values)
    
    # Plot topography
    im, _ = mne.viz.plot_topomap(normalized_values, info, axes=ax, show=False,
                               cmap='RdBu_r', vlim=(vmin, vmax), contours=6)
    
    # Add colorbar and title
    ax.set_title(f'{metric_info["name"]}\n(Mean = {mean_value:.3f} {metric_info["unit"]})', fontsize=12)
    
    # Add colorbar to the right of each subplot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Normalized Value\n(/ mean)', fontsize=10)
    
    return mean_value


def create_and_save_topography_figure(subjects, metrics, info, channel_averages, channel_names):
    """Create figure with all three topographical plots and save to file."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    
    # Handle single subject vs multiple subjects
    if len(subjects) == 1:
        subject_name = subjects[0]
        title = f'Topographical Distribution - Subject {subject_name}\n(Normalized by mean across channels)'
        output_dir = Path(subject_name)
        filename = f"{subject_name}_topographies_normalized.png"
    else:
        title = f'Topographical Distribution - Average Across {len(subjects)} Subjects\n(Normalized by mean across channels)'
        output_dir = Path("distribution_analysis")
        filename = f"topographies_average_all_subjects_normalized.png"
    
    fig.suptitle(title, fontsize=14)
    
    for i, (metric, metric_info) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Extract values for this metric
        values = extract_channel_values(channel_averages, channel_names, metric)
        
        # Plot topography
        mean_value = plot_single_topography(values, info, ax, metric_info)
        if mean_value is None:
            continue
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(exist_ok=True)
    topo_path = output_dir / filename
    plt.savefig(topo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topographical plots saved to: {topo_path}")


def plot_topographies(subjects):
    """
    Create topographical plots for the three spectral parameters.
    Shows spatial distribution across electrodes, with values averaged across all subjects
    and normalized by mean across all electrodes.
    Plots are arranged vertically: Peak Frequency, Bandwidth, AUC.
    """
    # Step 1: Load and combine data from all subjects
    combined_data = load_and_combine_subject_data(subjects)
    if combined_data is None:
        return
    
    # Step 2: Filter to EEG channels and calculate averages
    channel_averages = filter_eeg_channels(combined_data)
    if channel_averages is None:
        return
    
    # Step 3: Setup electrode info (montage, filtering, MNE info)
    info, channel_averages, channel_names = setup_electrode_info(channel_averages)
    if info is None:
        return
    
    # Step 4: Define metrics to plot
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'}, 
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    # Step 5: Create and save topography figure
    create_and_save_topography_figure(subjects, metrics, info, channel_averages, channel_names)


def print_summary_statistics(df, channel_name='VREF'):
    """
    Print comprehensive summary statistics for all spectral metrics.
    Includes count, mean, std, quartiles, min/max for each metric.
    """
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS - {channel_name} Channel")
    print(f"{'='*60}")
    
    metrics = ['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']
    
    for metric in metrics:
        data = df[metric].dropna()
        print(f"\n{metric.replace('avg_', '').replace('_', ' ').title()}:")
        print(f"  Min: {data.min():.4f}")
        print(f"  25%: {data.quantile(0.25):.4f}")
        print(f"  Median: {data.median():.4f}")
        print(f"  75%: {data.quantile(0.75):.4f}")
        print(f"  Max: {data.max():.4f}")


def plot_subject_all_channels_distributions(subjects):
    """
    Plot distributions of peak frequency, bandwidth, and AUC for all channels 
    within each subject individually. Creates separate plots for each subject.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to analyze
    """
    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    
    for subject in subjects:
        print(f"\nProcessing distributions for subject {subject}...")
        
        # Load subject data
        subject_data = load_subject_data(subject)
        if subject_data is None:
            print(f"  ✗ No data found for subject {subject}")
            continue
        
        # Filter out non-EEG channels only
        eeg_data = subject_data[~subject_data['channel'].str.contains('EMG', case=False, na=False)]
        
        if len(eeg_data) == 0:
            print(f"  ✗ No EEG channels found for subject {subject}")
            continue
        
        # Create figure with 3 subplots for the 3 metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'All Channels Distribution - Subject {subject}\n(N={len(eeg_data)} channels)', fontsize=16)
        
        for i, (metric, info) in enumerate(metrics.items()):
            ax = axes[i]
            data = eeg_data[metric].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{info["name"]}')
                continue
            
            # Calculate optimal number of bins
            n_bins = calculate_optimal_bins(data)
            
            # Create histogram with KDE
            sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
            
            # Calculate statistics
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            
            # Add statistical lines
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Median: {median_val:.3f}')
            
            # Formatting
            ax.set_title(f'{info["name"]}', fontsize=14)
            ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # Save plot in the subject's own directory
        subject_output_dir = Path(subject)
        subject_output_dir.mkdir(exist_ok=True)
        plot_path = subject_output_dir / f"{subject}_all_channels_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ All channels distribution plot saved to: {plot_path}")
    
    print(f"\nCompleted all-channels distribution analysis for {len(subjects)} subjects")


def load_and_process_all_channels_data(subjects):
    """
    Shared function to load, filter, and average all channels data across subjects.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to analyze
        
    Returns:
    --------
    tuple: (subject_averages_df, total_channels, valid_subjects)
        - subject_averages_df: DataFrame with subject averages across channels
        - total_channels: Total number of channels processed
        - valid_subjects: Number of subjects with valid data
    """
    # Hardcoded specific channel exclusions
    specific_exclusions = {
        'EL3006': ['E202', 'E88'],
        'EL3005': ['E9']
    }
    
    print(f"\nProcessing data from all channels across {len(subjects)} subjects...")
    
    # Collect all data
    all_data = []
    total_channels = 0
    
    for subject in subjects:
        subject_data = load_subject_data(subject)
        if subject_data is None:
            continue
        
        # Filter out non-EEG channels
        eeg_data = subject_data[~subject_data['channel'].str.contains('EMG', case=False, na=False)]
        
        # Apply hardcoded specific exclusions
        if subject in specific_exclusions:
            excluded_for_subject = specific_exclusions[subject]
            eeg_data = eeg_data[~eeg_data['channel'].isin(excluded_for_subject)]
            print(f"  ℹ {subject}: Excluded specific channels {excluded_for_subject}")
        
        if len(eeg_data) > 0:
            eeg_data['subject'] = subject  # Add subject column
            all_data.append(eeg_data)
            total_channels += len(eeg_data)
            print(f"  ✓ {subject}: {len(eeg_data)} channels")
    
    if not all_data:
        print("No valid data found for analysis")
        return None, 0, 0
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Average across channels within each subject
    subject_averages = combined_df.groupby('subject')[['avg_peak_frequency', 'avg_bandwidth', 'avg_auc']].mean().reset_index()
    
    print(f"  ✓ Processed {len(subject_averages)} subjects with {total_channels} total channels")
    
    return subject_averages, total_channels, len(subject_averages)


def plot_boxplots_all_channels_all_subjects(subjects):
    """
    Create boxplots for all channels across all subjects with individual data points.
    Similar to plot_boxplots_with_dots but for all channels combined instead of single channel.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to analyze
    """
    # Use shared data processing function
    subject_averages, total_channels, valid_subjects = load_and_process_all_channels_data(subjects)
    
    if subject_averages is None:
        return

    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'All Channels Averaged Per Subject - Spectral Parameters Distribution\n'
                 f'{len(subjects)} subjects, {total_channels} total channels', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = subject_averages[metric].dropna()
        
        # Create boxplot
        box_plot = ax.boxplot(data, patch_artist=True, widths=0.6)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Add individual subject dots
        np.random.seed(42)  # For reproducible jitter
        x_jitter = np.random.normal(1, 0.04, size=len(data))  # Small jitter around x=1
        ax.scatter(x_jitter, data, alpha=0.6, color='red', s=30, zorder=3, label='Subjects')
        
        # Set x-axis limits based on data range with some padding
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        padding = data_range * 0.05  # 5% padding on each side
        ax.set_ylim(data_min - padding, data_max + padding)
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_ylabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_xticklabels([''])  # Remove x-axis labels since we only have one box
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    boxplot_path = output_dir / "all_channels_all_subjects_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All channels boxplots saved to: {boxplot_path}")


def plot_combined_all_channels_distributions(subjects):
    """
    Plot combined distributions across all channels and all subjects.
    Creates one plot showing the overall distribution of each metric across 
    all subjects (averaged within each subject across all channels).
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs to analyze
    """
    # Use shared data processing function
    subject_averages, total_channels, valid_subjects = load_and_process_all_channels_data(subjects)
    
    if subject_averages is None:
        return

    metrics = {
        'avg_peak_frequency': {'name': 'Peak Frequency', 'unit': 'Hz'},
        'avg_bandwidth': {'name': 'Bandwidth', 'unit': 'Hz'},
        'avg_auc': {'name': 'Area Under Curve', 'unit': 'AUC'}
    }
    
    # Create combined plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'All Channels Averaged Per Subject - Distribution\n'
                 f'{valid_subjects} subjects, {total_channels} total channels', fontsize=16)
    
    for i, (metric, info) in enumerate(metrics.items()):
        ax = axes[i]
        data = subject_averages[metric].dropna()
        
        # Calculate optimal number of bins
        n_bins = calculate_optimal_bins(data)
        
        # Create histogram with KDE
        sns.histplot(data, kde=True, ax=ax, alpha=0.7, bins=n_bins, stat='count')
        
        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        # Add statistical lines
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f'Median: {median_val:.3f}')
        
        
        # Set x-axis limits based on data range with some padding
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        padding = data_range * 0.05  # 5% padding on each side
        ax.set_xlim(data_min - padding, data_max + padding)
        
        # Formatting
        ax.set_title(f'{info["name"]}', fontsize=14)
        ax.set_xlabel(f'{info["name"]} ({info["unit"]})', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSTD = {std_val:.3f}\nMedian = {median_val:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save combined plot
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "all_channels_subject_avg_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All channels (subject averages) distribution plot saved to: {plot_path}")


def plot_group_spectral_power(subjects, target_channel='VREF', smoothing_window=5, aggregate_channels=False):
    """
    Plot smoothed mean spectral power for the group.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    target_channel : str
        Specific channel to analyze (used when aggregate_channels=False)
    smoothing_window : int
        Window size for smoothing (default: 5)
    aggregate_channels : bool
        If True, average across all channels per subject
        If False, analyze only the target_channel
    """
    from scipy.ndimage import uniform_filter1d
    
    # Hardcoded specific channel exclusions
    specific_exclusions = {
        'EL3006': ['E202', 'E88'],
        'EL3005': ['E9']
    }
    
    if aggregate_channels:
        print(f"\nCollecting spectral data for {len(subjects)} subjects (aggregating across all channels)...")
    else:
        print(f"\nCollecting spectral data for {len(subjects)} subjects (channel: {target_channel})...")
    
    all_spectral_data = []
    subject_data = {}
    for subject in subjects:
        if aggregate_channels:
            # Load all channels for this subject and average them
            subject_dir = Path(subject)
            channel_data = []
            
            # Find all spectral power CSV files for this subject
            spectral_files = list(subject_dir.glob(f"{subject}_*_output/{subject}_*_spectral_power.csv"))
            
            for spectral_file in spectral_files:
                # Extract channel name from file path
                channel = spectral_file.stem.split('_')[1]  # e.g., EL3004_VREF_spectral_power -> VREF
                
                # Skip EMG channels
                if 'EMG' in channel:
                    continue
                
                # Apply hardcoded specific exclusions
                if subject in specific_exclusions and channel in specific_exclusions[subject]:
                    continue
                
                try:
                    df_channel = pd.read_csv(spectral_file)
                    channel_data.append(df_channel)
                except Exception as e:
                    print(f"  ✗ Error loading {spectral_file}: {e}")
            
            if not channel_data:
                print(f"Warning: No valid channel data found for subject {subject}")
                continue
            
            # Average across channels for this subject
            try:
                # Find common frequency range across all channels
                common_freqs = channel_data[0]['frequency'].values
                all_powers = []
                
                for df_channel in channel_data:
                    # Interpolate to common frequency grid if needed
                    if not np.array_equal(df_channel['frequency'].values, common_freqs):
                        interp_func = interp1d(df_channel['frequency'], df_channel['mean_power'], 
                                              bounds_error=False, fill_value='extrapolate')
                        power = interp_func(common_freqs)
                    else:
                        power = df_channel['mean_power'].values
                    all_powers.append(power)
                
                # Average across channels
                subject_avg_power = np.mean(all_powers, axis=0)
                
                # Create averaged DataFrame for this subject
                df_subject = pd.DataFrame({
                    'frequency': common_freqs,
                    'mean_power': subject_avg_power,
                    'std_power': np.zeros_like(subject_avg_power)  # Not meaningful after averaging
                })
                
                subject_data[subject] = df_subject
                all_spectral_data.append(df_subject)
                print(f"  ✓ Loaded and averaged {len(channel_data)} channels for subject {subject}")
                
            except Exception as e:
                print(f"  ✗ Error averaging channels for subject {subject}: {e}")
        
        else:
            # Load single channel data
            spectral_pattern = f"{subject}/{subject}_{target_channel}_output/{subject}_{target_channel}_spectral_power.csv"
            spectral_file = Path(spectral_pattern)
            
            if not spectral_file.exists():
                print(f"Warning: No spectral data found for subject {subject} at {spectral_file}")
                continue
                
            try:
                df = pd.read_csv(spectral_file)
                subject_data[subject] = df
                all_spectral_data.append(df)
            except Exception as e:
                print(f"  ✗ Error loading data for subject {subject}: {e}")
    
    if not all_spectral_data:
        print("No valid spectral data found!")
        return
    
    # Find common frequency range
    min_freq = max(df['frequency'].min() for df in all_spectral_data)
    max_freq = min(df['frequency'].max() for df in all_spectral_data)
    
    # Create common frequency grid
    freq_resolution = min(df['frequency'].diff().dropna().min() for df in all_spectral_data)
    common_freqs = np.arange(min_freq, max_freq, freq_resolution)
    
    # Interpolate all subjects to common frequency grid
    interpolated_powers = []
    for subject, df in subject_data.items():
        interp_func = interp1d(df['frequency'], df['mean_power'], 
                              bounds_error=False, fill_value='extrapolate')
        interpolated_power = interp_func(common_freqs)
        interpolated_powers.append(interpolated_power)
    
    # Convert to numpy array and compute group statistics
    power_matrix = np.array(interpolated_powers)
    group_mean = np.mean(power_matrix, axis=0)
    group_std = np.std(power_matrix, axis=0)
    
    # Apply smoothing
    if smoothing_window > 1:
        group_mean_smooth = uniform_filter1d(group_mean, size=smoothing_window)
        group_std_smooth = uniform_filter1d(group_std, size=smoothing_window)
        smooth_label = f" (smoothed, window={smoothing_window})"
    else:
        group_mean_smooth = group_mean
        group_std_smooth = group_std
        smooth_label = ""
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main line: group mean
    plt.plot(common_freqs, group_mean_smooth, 'b-', linewidth=2.5, label=f'Group Mean{smooth_label}')
    
    # Standard deviation bands with dotted lines
    plt.plot(common_freqs, group_mean_smooth + group_std_smooth, 'b:', linewidth=1.5, alpha=0.7, label='+1 SD')
    plt.plot(common_freqs, group_mean_smooth - group_std_smooth, 'b:', linewidth=1.5, alpha=0.7, label='-1 SD')
    
    # Shaded area for standard deviation
    plt.fill_between(common_freqs, 
                     group_mean_smooth - group_std_smooth,
                     group_mean_smooth + group_std_smooth,
                     alpha=0.2, color='blue', label='±1 SD region')
    
    # Formatting
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power (AU)', fontsize=12)
    
    if aggregate_channels:
        title = f'Group-Level Infraslow Fluctuations of Sigma Power (ISFS)\n' \
                f'Mean across {len(subjects)} subjects, averaged across all electrodes'
        filename = f"group_ISFS_spectral_power_all_channels.png"
    else:
        title = f'Group-Level Infraslow Fluctuations of Sigma Power (ISFS)\n' \
                f'Mean across {len(subjects)} subjects, channel {target_channel}'
        filename = f"group_ISFS_spectral_power_{target_channel}.png"
    
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("distribution_analysis")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Group spectral plot saved to: {plot_path}")
    print(f"✓ Analyzed {len(subjects)} subjects")
    print(f"✓ Frequency range: {common_freqs[0]:.4f} - {common_freqs[-1]:.4f} Hz")
    print(f"✓ Smoothing applied: {smoothing_window}-point window" if smoothing_window > 1 else "✓ No smoothing applied")


def main():
    """
    Main function to run distribution analysis across all subjects.
    Analyzes distributions of spectral metrics (peak frequency, bandwidth, AUC)
    for specific electrodes and generates plots and statistics.
    """
    subjects = get_all_subjects(f"{BASE_DIR}/control_clean/")
    if not subjects:
        print("No subjects found!")
        return
    
    target_channel = 'VREF'
    print(f"\nAnalyzing distributions for channel: {target_channel}")
    
    # for subject in all_subjects:
        # subjects = [subject]
    df = collect_all_data(subjects, target_channel)
    if df is None:
        print("No data to analyze!")
        return
    
    # plot_distributions(df, target_channel)
    # plot_combined_distributions(df, target_channel)
    # plot_boxplots_with_dots(df, target_channel)
    
    # Plot topography for all subjects with normalization
    plot_topographies(subjects)
    
    # Generate all-channels distribution plots
    print("\n" + "="*60)
    print("GENERATING ALL-CHANNELS DISTRIBUTION PLOTS")
    print("="*60)
    
    # Individual subject all-channels distributions
    # plot_subject_all_channels_distributions(subjects)
    
    # Combined all-channels distribution across all subjects
    # plot_combined_all_channels_distributions(subjects)
    
    # All channels boxplots
    # plot_boxplots_all_channels_all_subjects(subjects)
    
    # Generate both spectral plots
    print("\n" + "="*60)
    print("GENERATING SPECTRAL POWER PLOTS")
    print("="*60)
    
    # Single channel plot
    # plot_group_spectral_power(subjects, target_channel, aggregate_channels=False)
    
    # All channels aggregated plot
    # plot_group_spectral_power(subjects, target_channel, smoothing_window=3, aggregate_channels=True)
    
    # print_summary_statistics(df, target_channel)
    
    if len(subjects) > 1:
        output_dir = Path("distribution_analysis")
        output_dir.mkdir(exist_ok=True)
        data_path = output_dir / f"{target_channel}_combined_data.csv"
        df.to_csv(data_path, index=False)
        print(f"\nCombined dataset saved to: {data_path}")

if __name__ == "__main__":
    main()