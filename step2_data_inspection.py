from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
import gc
import mne
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import numpy as np
from sleepeegpy.dashboard import create_dashboard
from config import BASE_DIR
from utils import get_all_subjects, find_subject_fif_file

mne.set_log_level("error")


def process_single_subject(
    subject_id,
    eeg_path,
    output_dir,
    bad_channels_path=None,
    annotations_path=None,
    hypno_path=None,
    hypno_freq=1/30,
    hypno_psd_pick=["Cz"],
):
    """ Process a single subject and create dashboard. """
    print(f"Processing subject: {subject_id}")
    print(f"{'='*60}")
    print(f"  EEG file: {eeg_path.name} ✓")
    print(f"  Hypnogram: {hypno_path.name} {'✓' if hypno_path and hypno_path.exists() else '✗ (will skip)'}")
    print(f"  Bad channels: {bad_channels_path.name} {'✓' if bad_channels_path and bad_channels_path.exists() else '✗ (will skip)'}")
    print(f"  Annotations: {annotations_path.name} {'✓' if annotations_path and annotations_path.exists() else '✗ (will skip)'}")
    try:
        create_dashboard(subject_id, eeg_path, hypno_path if (hypno_path and hypno_path.exists()) else None,
            hypno_freq if (hypno_path and hypno_path.exists()) else None, output_dir=output_dir,
            hypno_psd_pick=hypno_psd_pick,
            path_to_bad_channels=bad_channels_path if (bad_channels_path and bad_channels_path.exists()) else None,
            path_to_annotations=annotations_path if (annotations_path and annotations_path.exists()) else None,
        )
        print(f" ✓ Dashboard saved to: {output_dir / f'dashboard_{subject_id}.png'}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {subject_id}: {str(e)}")
        return False


def plot_psd_by_sleep_stage(
    subject_id,
    eeg_path,
    hypno_path,
    channel_name,
    output_dir=None,
    fmin=0,
    fmax=25,
    hypno_freq=1/30,
    n_fft=2048,
    n_per_seg=768,
    n_overlap=512,
    window="hamming",
):
    """
    Plot PSD for each sleep stage with different colored lines.
    
    Implementation matches sleepeegpy's approach:
    - Upsamples hypnogram to EEG sampling rate
    - Finds continuous regions for each stage
    - Computes weighted average of PSDs across regions
    
    Args:
        subject_id: Subject identifier
        eeg_path: Path to the EEG file
        hypno_path: Path to hypnogram file
        channel_name: Name of the channel to analyze (e.g., 'Cz', 'E101')
        output_dir: Directory to save the plot (if None, displays only)
        fmin: Minimum frequency to plot (Hz)
        fmax: Maximum frequency to plot (Hz)
        hypno_freq: Sampling frequency of hypnogram in Hz (default: 1/30 for 30-sec epochs)
        n_fft: FFT length for frequency resolution (default: 2048)
        n_per_seg: Segment length in samples (default: 768, matches sleepeegpy)
        n_overlap: Overlapping samples between segments (default: 512, 67% overlap)
        window: Window function for tapering (default: 'hamming')
    
    Returns:
        dict: {stage_name: (frequencies, psd_values)} or None if error
    """
    from scipy import ndimage
    from more_itertools import collapse
    
    print(f"Plotting PSD by sleep stage for {subject_id}, channel: {channel_name}")
    print(f"{'='*60}")
    print(f"  Welch parameters: n_fft={n_fft}, n_per_seg={n_per_seg}, n_overlap={n_overlap}, window={window}")
    
    # Sleep stage mapping and colors
    stage_mapping = {
        0: 'Wake',
        1: 'N1', 
        2: 'N2',
        3: 'N3',
        4: 'REM',
        -1: 'Unknown'
    }
    
    stage_colors = {
        'Wake': '#1f77b4',  # blue
        'N1': '#ff7f0e',    # orange
        'N2': '#2ca02c',    # green
        'N3': '#d62728',    # red
        'REM': '#9467bd',   # purple
        'Unknown': '#7f7f7f'  # gray
    }
    
    try:
        # Load EEG data
        print(f"  Loading EEG data...")
        raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
        
        # Check if channel exists
        if channel_name not in raw.ch_names:
            print(f"  ✗ Channel '{channel_name}' not found in data")
            print(f"  Available channels: {raw.ch_names[:10]}...")
            return None
        
        # Check if channel is marked as bad
        if channel_name in raw.info['bads']:
            print(f"  ✗ Warning: Channel '{channel_name}' is marked as bad!")
            print(f"  Bad channels: {raw.info['bads']}")
        
        # Load hypnogram
        print(f"  Loading hypnogram...")
        hypno = np.loadtxt(hypno_path, dtype=int)
        print(f"  Hypnogram loaded: {len(hypno)} samples at {hypno_freq} Hz")
        
        # Get sampling frequency of EEG
        sfreq = raw.info['sfreq']
        
        # Upsample hypnogram to match EEG sampling rate (like sleepeegpy does)
        print(f"  Upsampling hypnogram to {sfreq} Hz...")
        repeats = int(sfreq / hypno_freq)
        if not (sfreq / hypno_freq).is_integer():
            raise ValueError(f"EEG sfreq ({sfreq}) / hypno_freq ({hypno_freq}) must be a whole number")
        
        hypno_up = np.repeat(hypno, repeats)
        
        # Fit hypnogram to data length (crop or pad if needed)
        npts_data = raw.n_times
        if len(hypno_up) < npts_data:
            print(f"  Warning: Hypnogram shorter than data, padding with last value")
            hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
        elif len(hypno_up) > npts_data:
            print(f"  Warning: Hypnogram longer than data, cropping to match")
            hypno_up = hypno_up[:npts_data]
        
        print(f"  Upsampled hypnogram: {len(hypno_up)} samples")
        
        # Get unique stages present in the data
        unique_stages = np.unique(hypno_up)
        print(f"  Sleep stages found: {[stage_mapping.get(s, f'Unknown({s})') for s in unique_stages]}")
        
        # Get data for the channel - CRITICAL: reject bad annotations by marking them as NaN
        # This matches sleepeegpy's approach
        data = raw.get_data(
            picks=[channel_name],
            reject_by_annotation="NaN"  # Mark bad epochs as NaN
        )  # Shape: (1, n_times)
        
        # Count total valid (non-NaN) samples for percentage calculation
        n_samples_valid = np.count_nonzero(~np.isnan(data), axis=1)[0]
        print(f"  Valid samples (non-bad): {n_samples_valid}/{npts_data} ({n_samples_valid/npts_data*100:.1f}%)")
        
        # Compute PSD for each sleep stage (following sleepeegpy's approach)
        stage_psds = {}
        
        for stage_code in unique_stages:
            if stage_code == -1:  # Skip unknown stages
                continue
                
            stage_name = stage_mapping.get(stage_code, f'Stage{stage_code}')
            
            # Create boolean mask for this stage
            stage_mask = hypno_up == stage_code
            
            # Find continuous regions with this sleep stage
            labeled, n_regions = ndimage.label(stage_mask)
            regions = list(collapse(ndimage.find_objects(labeled)))
            
            if len(regions) == 0:
                continue
            
            print(f"  Processing {stage_name}: {len(regions)} continuous regions, {stage_mask.sum() / sfreq:.1f} seconds total")
            
            # Compute PSD for each region and weighted average (like sleepeegpy)
            psds_list, weights = [], []
            n_samples_total = 0
            
            for region in regions:
                region_data = data[:, region]
                
                # Count valid (non-NaN) samples in this region
                n_valid_samples = np.count_nonzero(~np.isnan(region_data))
                
                # Skip regions with too few valid samples for reliable FFT
                # Need at least n_fft valid samples
                if n_valid_samples < n_fft:
                    continue
                
                # Compute PSD for this region using Welch's method
                # psd_array_welch handles NaN values automatically
                # Using sleepeegpy dashboard parameters for smoother plots
                psds, freqs = mne.time_frequency.psd_array_welch(
                    region_data,
                    sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=n_fft,
                    n_per_seg=n_per_seg,
                    n_overlap=n_overlap,
                    window=window,
                    verbose=False
                )
                
                # Check if PSD is valid (not all NaN)
                if np.all(np.isnan(psds)):
                    continue
                
                psds_list.append(psds[0])  # Remove channel dimension
                weights.append(n_valid_samples)  # Weight by valid samples
                n_samples_total += n_valid_samples
            
            if len(psds_list) == 0:
                print(f"  ✗ Warning: {stage_name} has no valid regions (all regions have too many bad epochs)")
                continue
            
            # Weighted average across regions (matching sleepeegpy)
            # If there are NaNs in PSD, mask them for proper averaging
            psds_array = np.array(psds_list)
            if np.any(np.isnan(psds_array)):
                masked_psds = np.ma.masked_array(psds_array, np.isnan(psds_array))
                avg_psd = np.ma.average(masked_psds, weights=weights, axis=0)
                avg_psd = avg_psd.filled(np.nan)  # Convert back to regular array
            else:
                avg_psd = np.average(psds_array, weights=weights, axis=0)
            
            # Calculate percentage of VALID (non-bad) data for this stage
            # This matches sleepeegpy's approach
            percent = round(n_samples_total / n_samples_valid * 100, 2)
            
            stage_psds[stage_name] = {
                'freqs': freqs,
                'psd': avg_psd,
                'percent': percent,
                'n_samples': n_samples_total
            }
        
        # Create plot (matching sleepeegpy's style)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot PSD for each stage
        for stage_name in ['Wake', 'N1', 'N2', 'N3', 'REM']:
            if stage_name in stage_psds:
                freqs = stage_psds[stage_name]['freqs']
                psd = stage_psds[stage_name]['psd']
                percent = stage_psds[stage_name]['percent']
                
                # Convert from V²/Hz to µV²/Hz (dB) - matching sleepeegpy
                psd_db = 10 * np.log10(10**12 * psd)
                
                ax.plot(
                    freqs, 
                    psd_db, 
                    linewidth=2, 
                    color=stage_colors[stage_name],
                    label=f'{stage_name} ({percent}%)',
                    alpha=0.8
                )
        
        # Styling to match sleepeegpy
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel(r'$\mu V^{2}/Hz$ (dB)', fontsize=12)
        ax.set_title(f'PSD - Subject: {subject_id}, Channel: {channel_name}', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(fmin, fmax)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save or display
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f'{subject_id}_{channel_name}_psd_by_stage.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ PSD plot saved to: {output_path}")
            plt.close(fig)
            # Force garbage collection of figure to free memory
            del fig, ax
        else:
            plt.show()
        
        print(f"  ✓ PSD by sleep stage computation complete")
        return stage_psds
        
    except Exception as e:
        print(f"  ✗ Error computing PSD by sleep stage: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Ensure cleanup even if error occurs
        plt.close('all')
        gc.collect()


def process_all_subjects(
    data_dir,
    output_dir,
    hypno_dir,
    hypno_suffix="_hypno.txt",
    bad_channels_suffix="_bad_channels.txt",
    annotations_suffix="_annotations.txt",
    hypno_freq=1/30,  # 30-second epochs
    hypno_psd_pick=["Cz"],
):
    """
    Process all subjects in a directory and create dashboards.
    
    Expects structure:
        data_dir/
        ├── subject001/
        │   ├── subject001.fif
        │   ├── subject001_bad_channels.txt
        │   └── subject001_annotations.txt
        └── subject002/
            ├── subject002.fif
            ├── subject002_bad_channels.txt
            └── subject002_annotations.txt
    
    Hypnograms are in a separate flat directory:
        hypno_dir/
        ├── subject001_hypno.txt
        └── subject002_hypno.txt
    
    Args:
        data_dir: Directory containing subject subdirectories with EEG files
        output_dir: Base directory for saving dashboards
        hypno_dir: Directory containing hypnogram files (flat structure)
        hypno_suffix: Suffix for hypnogram files (e.g., "_hypno.txt")
        hypno_freq: Sampling frequency of hypnogram in Hz (1/30 for 30-sec epochs)
        hypno_psd_pick: Channel(s) for spectrogram visualization
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    hypno_dir = Path(hypno_dir)
    
    # Find all EEG files recursively in subject subdirectories
    # Pattern: data_dir/subject_folder/subject.fif
    all_files = sorted(data_dir.glob("**/*.fif"))
    
    # Filter out files ending with -1, -2, etc. and group by subject directory
    subject_files = defaultdict(list)
    for fif_file in all_files:
        # Skip files ending with a dash followed by digits (e.g., -1, -2, -10)
        if not re.search(r'-\d+$', fif_file.stem):
            subject_dir = fif_file.parent
            subject_files[subject_dir].append(fif_file)
    
    # Select the file with the longest name for each subject
    eeg_files = []
    for subject_dir, files in subject_files.items():
        if len(files) == 1:
            eeg_files.append(files[0])
        else:
            # Multiple files: choose the one with the longest filename
            longest_file = max(files, key=lambda f: len(f.name))
            eeg_files.append(longest_file)
    
    print(f"\nFound {len(all_files)} total .fif files")
    print(f"Selected {len(eeg_files)} files (one per subject, longest name after filtering)")
    print(f"Hypnogram directory: {hypno_dir}")
    
    # Process each subject
    successful_count = 0
    failed_count = 0
    
    for eeg_file in eeg_files:
        # Extract subject code from filename (remove extension)
        subject_id = str(eeg_file.stem).split("_")[0] 
        if subject_id != "DG1":
            continue
        
        subject_folder = eeg_file.parent
        bad_channels_path = subject_folder / (subject_id + bad_channels_suffix)
        annotations_path = subject_folder / (subject_id + annotations_suffix)
        hypno_path = hypno_dir / (subject_id + hypno_suffix)
        
        success = process_single_subject(
            subject_id=subject_id,
            eeg_path=eeg_file,
            output_dir=output_dir,
            bad_channels_path=bad_channels_path,
            annotations_path=annotations_path,
            hypno_path=hypno_path,
            hypno_freq=hypno_freq,
            hypno_psd_pick=hypno_psd_pick,
        )
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}")


def plot_psd_all_subjects(
    data_dir,
    hypno_dir,
    output_base_dir,
    channels_list,
    hypno_freq=1,
    hypno_suffix=".txt",
    fmin=0,
    fmax=25,
    n_fft=2048,
    n_per_seg=768,
    n_overlap=512,
    window="hamming",
):
    """
    Plot PSD by sleep stage for all subjects and multiple channels.
    Args:
        n_fft: FFT length for frequency resolution
        n_per_seg: Segment length in samples
        n_overlap: Overlapping samples between segments
        window: Window function for tapering
    """
    output_base_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n{'='*80}")
    print(f"Starting PSD analysis for all subjects")
    print(f"{'='*80}")
    
    subject_ids = get_all_subjects(data_dir)
    if not subject_ids:
        print("No subjects found!")
        return
    
    # Track statistics
    stats = {
        'total_subjects': len(subject_ids),
        'successful_subjects': 0,
        'failed_subjects': 0,
        'successful_channels': 0,
        'failed_channels': 0,
        'missing_hypno': 0,
        'missing_fif': 0,
        'missing_hypno_list': [],
        'missing_fif_list': [],
    }
    
    for idx, subject_id in enumerate(subject_ids, 1):
        print(f"\n[{idx}/{len(subject_ids)}] Processing subject: {subject_id}")
        print(f"{'-'*80}")
        subject_dir = data_dir / subject_id
        fif_path = find_subject_fif_file(subject_dir)
        if not fif_path:
            print(f"  ✗ No .fif file found for {subject_id}")
            stats['failed_subjects'] += 1
            stats['missing_fif'] += 1
            stats['missing_fif_list'].append(subject_id)
            continue
        
        hypno_path = hypno_dir / f"{subject_id}{hypno_suffix}"
        if not hypno_path.exists():
            print(f"  ✗ Hypnogram not found: {hypno_path}")
            stats['failed_subjects'] += 1
            stats['missing_hypno'] += 1
            stats['missing_hypno_list'].append(subject_id)
            continue
        
        psds_dir = output_base_dir / "psds"
        psds_dir.mkdir(exist_ok=True, parents=True)
        subject_success = True
        for channel in channels_list:
            print(f"\n  Channel: {channel}")
            try:
                result = plot_psd_by_sleep_stage(subject_id, fif_path, hypno_path, channel, psds_dir, 
                                                 fmin, fmax, hypno_freq, n_fft, n_per_seg, n_overlap, window)
                
                if result is not None:
                    stats['successful_channels'] += 1
                    print(f"    ✓ Successfully plotted PSD for {channel}")
                else:
                    stats['failed_channels'] += 1
                    subject_success = False
                    print(f"    ✗ Failed to plot PSD for {channel}")
                    
            except Exception as e:
                stats['failed_channels'] += 1
                subject_success = False
                print(f"    ✗ Error processing {channel}: {str(e)}")
        
        # Clean up memory after processing each subject
        gc.collect()
        
        if subject_success:
            stats['successful_subjects'] += 1
            print(f"\n  ✓ Successfully completed all channels for {subject_id}")
        else:
            stats['failed_subjects'] += 1
            print(f"\n  ⚠ Some channels failed for {subject_id}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total subjects processed: {stats['total_subjects']}")
    print(f"  ✓ Successful: {stats['successful_subjects']}")
    print(f"  ✗ Failed: {stats['failed_subjects']}")
    print(f"    - Missing .fif file: {stats['missing_fif']}")
    if stats['missing_fif_list']:
        print(f"      Subjects: {', '.join(stats['missing_fif_list'])}")
    print(f"    - Missing hypnogram: {stats['missing_hypno']}")
    if stats['missing_hypno_list']:
        print(f"      Subjects: {', '.join(stats['missing_hypno_list'])}")
    print(f"\nChannel statistics:")
    print(f"  Total channel plots: {stats['successful_channels'] + stats['failed_channels']}")
    print(f"  ✓ Successful: {stats['successful_channels']}")
    print(f"  ✗ Failed: {stats['failed_channels']}")
    print(f"\nOutput saved to: {output_base_dir}")
    print(f"{'='*80}\n")


def find_roi_channels(
    subject_id,
    eeg_path,
    hypno_path,
    hypno_freq=1,
    n_channels=3,
    sigma_band=(12, 16),
    output_dir=None,
):
    """
    Find best ROI channels for spindle analysis based on N2 sigma power topography.
    
    Strategy:
    1. Restrict to clean N2 epochs (and optionally N3)
    2. Make sigma-band topography (12-16 Hz average power)
    3. Pick top 2-3 sensors near canonical central/centroparietal and in compact cluster
    4. Prefer Cz-ish, C3-ish, C4-ish or central + 2 centroparietal neighbors
    5. Sanity check: sigma power high in N2 but NOT in wake/REM
    
    Args:
        subject_id: Subject identifier
        eeg_path: Path to EEG .fif file
        hypno_path: Path to hypnogram file
        hypno_freq: Hypnogram sampling frequency (Hz)
        n_channels: Number of channels to return (default: 3)
        sigma_band: Frequency band for sigma (default: 12-16 Hz)
        output_dir: Optional directory to save topography plot
    
    Returns:
        list: Selected channel names (e.g., ['E11', 'E12', 'E20'])
    """
    print(f"\n{'='*80}")
    print(f"Finding ROI channels for: {subject_id}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading EEG...")
    raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
    
    print(f"Loading hypnogram...")
    hypno = np.loadtxt(hypno_path, dtype=int)
    
    # Upsample hypnogram
    sfreq = raw.info['sfreq']
    repeats = int(sfreq / hypno_freq)
    hypno_up = np.repeat(hypno, repeats)
    
    # Fit to data length
    npts_data = raw.n_times
    if len(hypno_up) < npts_data:
        hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
    elif len(hypno_up) > npts_data:
        hypno_up = hypno_up[:npts_data]
    
    # Get EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    eeg_channels = [raw.ch_names[i] for i in eeg_picks]
    print(f"Analyzing {len(eeg_channels)} EEG channels")
    
    # Create masks for N2 and Wake/REM
    n2_mask = (hypno_up == 2)
    wake_rem_mask = (hypno_up == 0) | (hypno_up == 4)
    
    n2_duration = n2_mask.sum() / sfreq
    wake_rem_duration = wake_rem_mask.sum() / sfreq
    print(f"N2: {n2_duration:.1f}s ({n2_duration/60:.1f} min)")
    print(f"Wake/REM: {wake_rem_duration:.1f}s ({wake_rem_duration/60:.1f} min)")
    
    # Filter to sigma band
    print(f"Filtering to sigma ({sigma_band[0]}-{sigma_band[1]} Hz)...")
    raw_sigma = raw.copy().filter(l_freq=sigma_band[0], h_freq=sigma_band[1], picks='eeg', verbose=False)
    
    # Get data with bad epoch rejection
    data = raw_sigma.get_data(picks=eeg_picks, reject_by_annotation="NaN")
    
    # Compute RMS power for each channel
    channel_stats = {}
    
    for i, ch_name in enumerate(eeg_channels):
        ch_data = data[i, :]
        
        # N2 sigma power
        n2_data = ch_data[n2_mask]
        n2_valid = n2_data[~np.isnan(n2_data)]
        n2_power = np.sqrt(np.mean(n2_valid**2)) if len(n2_valid) > 0 else 0
        
        # Wake/REM sigma power
        wake_rem_data = ch_data[wake_rem_mask]
        wake_rem_valid = wake_rem_data[~np.isnan(wake_rem_data)]
        wake_rem_power = np.sqrt(np.mean(wake_rem_valid**2)) if len(wake_rem_valid) > 0 else 0
        
        # Ratio (higher = more N2-specific)
        ratio = n2_power / wake_rem_power if wake_rem_power > 0 else 0
        
        channel_stats[ch_name] = {
            'n2_power': n2_power,
            'wake_rem_power': wake_rem_power,
            'ratio': ratio,
            'score': n2_power * ratio  # Combined score
        }
    
    # Sort by score
    sorted_channels = sorted(channel_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Select top N channels
    selected = [ch for ch, _ in sorted_channels[:n_channels]]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Top {n_channels} ROI Channels:")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Channel':<12}{'N2 Power':<12}{'Wake/REM':<12}{'Ratio':<10}{'Score':<10}")
    print(f"{'-'*80}")
    
    for rank, (ch, stats) in enumerate(sorted_channels[:n_channels], 1):
        print(f"{rank:<6}{ch:<12}{stats['n2_power']:<12.3f}{stats['wake_rem_power']:<12.3f}"
              f"{stats['ratio']:<10.2f}{stats['score']:<10.2f}")
    
    # Show next 5 as runner-ups
    if len(sorted_channels) > n_channels:
        print(f"\nRunner-ups:")
        for rank, (ch, stats) in enumerate(sorted_channels[n_channels:n_channels+5], n_channels+1):
            print(f"{rank:<6}{ch:<12}{stats['n2_power']:<12.3f}{stats['wake_rem_power']:<12.3f}"
                  f"{stats['ratio']:<10.2f}{stats['score']:<10.2f}")
    
    # Optional: Create topography plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nCreating topography...")
        n2_power_array = np.array([channel_stats[ch]['n2_power'] for ch in eeg_channels])
        
        info = mne.pick_info(raw.info, eeg_picks)
        evoked = mne.EvokedArray(n2_power_array[:, np.newaxis], info, tmin=0)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im, _ = mne.viz.plot_topomap(
            evoked.data[:, 0], evoked.info, axes=ax, show=False,
            cmap='viridis', vlim=(0, np.percentile(n2_power_array, 95)), contours=6
        )
        
        # Mark selected channels
        selected_indices = [eeg_channels.index(ch) for ch in selected]
        pos = mne.channels.layout._find_topomap_coords(evoked.info, picks=selected_indices)
        ax.plot(pos[:, 0], pos[:, 1], 'g*', markersize=15, markeredgecolor='black', 
                markeredgewidth=1.5, label='Selected')
        
        ax.set_title(f'N2 Sigma Power ({sigma_band[0]}-{sigma_band[1]} Hz)\n{subject_id}', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='RMS Power (µV)', shrink=0.8)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        
        plot_path = output_dir / f'{subject_id}_roi_topography.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    print(f"{'='*80}\n")
    return selected


def validate_channels_spatial_consistency(
    subject_id,
    eeg_path,
    hypno_path,
    hypno_freq=1,
    correlation_threshold=0.3,
    amplitude_threshold=3.0,
    spectral_threshold=3.0,
    output_dir=None,
):
    """
    Validate channels using neighbor-consistency checks on clean N2 data.
    
    Strategy (powerful for dense EGI nets):
    1. Correlation with neighbors: channels with unusually low correlation are suspicious
    2. RMS amplitude vs neighbors: persistently higher/lower amplitude is suspicious
    3. Spectral similarity vs neighbors: weird spectrum relative to neighbors is suspicious
    
    This catches "looks fine in raw but spatially inconsistent" channels that ruin topomaps.
    
    Args:
        subject_id: Subject identifier
        eeg_path: Path to EEG .fif file
        hypno_path: Path to hypnogram file
        hypno_freq: Hypnogram sampling frequency (Hz)
        correlation_threshold: Min correlation with neighbors (default: 0.3)
        amplitude_threshold: Max z-score for amplitude deviation (default: 3.0)
        spectral_threshold: Max z-score for spectral deviation (default: 3.0)
        output_dir: Optional directory to save validation plots
    
    Returns:
        dict: {
            'bad_correlation': list of channels,
            'bad_amplitude': list of channels,
            'bad_spectral': list of channels,
            'all_bad': list of all problematic channels
        }
    """
    print(f"\n{'='*80}")
    print(f"Validating spatial consistency for: {subject_id}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading EEG...")
    raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
    
    print(f"Loading hypnogram...")
    hypno = np.loadtxt(hypno_path, dtype=int)
    
    # Upsample hypnogram
    sfreq = raw.info['sfreq']
    repeats = int(sfreq / hypno_freq)
    hypno_up = np.repeat(hypno, repeats)
    
    # Fit to data length
    npts_data = raw.n_times
    if len(hypno_up) < npts_data:
        hypno_up = np.pad(hypno_up, (0, npts_data - len(hypno_up)), mode='edge')
    elif len(hypno_up) > npts_data:
        hypno_up = hypno_up[:npts_data]
    
    # Get EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    eeg_channels = [raw.ch_names[i] for i in eeg_picks]
    print(f"Analyzing {len(eeg_channels)} EEG channels")
    
    # Extract clean N2 data
    n2_mask = (hypno_up == 2)
    n2_duration = n2_mask.sum() / sfreq
    print(f"Clean N2: {n2_duration:.1f}s ({n2_duration/60:.1f} min)")
    
    if n2_duration < 60:
        print("⚠ Warning: Less than 60s of N2. Results may be unreliable.")
    
    # Get N2 data with bad epoch rejection
    data = raw.get_data(picks=eeg_picks, reject_by_annotation="NaN")
    n2_data = data[:, n2_mask]
    
    # Get channel positions for neighbor finding
    montage = raw.get_montage()
    if montage is None:
        print("⚠ Warning: No montage found. Cannot compute spatial neighbors.")
        return None
    
    # Build adjacency matrix (neighbors within ~4cm for EGI)
    print(f"Computing spatial neighbors...")
    adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type='eeg')
    adjacency = adjacency.toarray()
    
    # === 1. CORRELATION CHECK ===
    print(f"\n{'='*80}")
    print(f"1. Neighbor Correlation Analysis")
    print(f"{'='*80}")
    
    correlation_scores = {}
    
    for i, ch_name in enumerate(eeg_channels):
        ch_data = n2_data[i, :]
        ch_valid = ch_data[~np.isnan(ch_data)]
        
        if len(ch_valid) < 100:  # Skip if too little data
            correlation_scores[ch_name] = 0
            continue
        
        # Find neighbors
        neighbor_indices = np.where(adjacency[i, :])[0]
        
        if len(neighbor_indices) == 0:
            correlation_scores[ch_name] = 0
            continue
        
        # Compute correlation with each neighbor
        correlations = []
        for neighbor_idx in neighbor_indices:
            neighbor_data = n2_data[neighbor_idx, :]
            neighbor_valid = neighbor_data[~np.isnan(neighbor_data)]
            
            # Find common valid samples
            valid_mask = ~(np.isnan(ch_data) | np.isnan(neighbor_data))
            if valid_mask.sum() < 100:
                continue
            
            corr = np.corrcoef(ch_data[valid_mask], neighbor_data[valid_mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Average correlation with neighbors
        avg_corr = np.mean(correlations) if len(correlations) > 0 else 0
        correlation_scores[ch_name] = avg_corr
    
    # Find channels with low correlation
    bad_correlation = [ch for ch, corr in correlation_scores.items() if corr < correlation_threshold]
    
    print(f"Correlation threshold: {correlation_threshold}")
    print(f"Channels with low neighbor correlation: {len(bad_correlation)}")
    if len(bad_correlation) > 0 and len(bad_correlation) <= 20:
        print(f"  {', '.join(bad_correlation)}")
    
    # === 2. AMPLITUDE CHECK ===
    print(f"\n{'='*80}")
    print(f"2. RMS Amplitude Analysis")
    print(f"{'='*80}")
    
    amplitude_scores = {}
    
    for i, ch_name in enumerate(eeg_channels):
        ch_data = n2_data[i, :]
        ch_valid = ch_data[~np.isnan(ch_data)]
        
        if len(ch_valid) < 100:
            amplitude_scores[ch_name] = {'rms': 0, 'z_score': 0}
            continue
        
        ch_rms = np.sqrt(np.mean(ch_valid**2))
        
        # Find neighbors
        neighbor_indices = np.where(adjacency[i, :])[0]
        
        if len(neighbor_indices) == 0:
            amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': 0}
            continue
        
        # Compute RMS for neighbors
        neighbor_rms_list = []
        for neighbor_idx in neighbor_indices:
            neighbor_data = n2_data[neighbor_idx, :]
            neighbor_valid = neighbor_data[~np.isnan(neighbor_data)]
            if len(neighbor_valid) < 100:
                continue
            neighbor_rms = np.sqrt(np.mean(neighbor_valid**2))
            neighbor_rms_list.append(neighbor_rms)
        
        if len(neighbor_rms_list) == 0:
            amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': 0}
            continue
        
        # Z-score relative to neighbors
        neighbor_mean = np.mean(neighbor_rms_list)
        neighbor_std = np.std(neighbor_rms_list)
        z_score = (ch_rms - neighbor_mean) / neighbor_std if neighbor_std > 0 else 0
        
        amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': z_score}
    
    # Find channels with extreme amplitude
    bad_amplitude = [ch for ch, stats in amplitude_scores.items() 
                     if abs(stats['z_score']) > amplitude_threshold]
    
    print(f"Amplitude z-score threshold: ±{amplitude_threshold}")
    print(f"Channels with extreme amplitude: {len(bad_amplitude)}")
    if len(bad_amplitude) > 0 and len(bad_amplitude) <= 20:
        for ch in bad_amplitude:
            print(f"  {ch}: z={amplitude_scores[ch]['z_score']:.2f}")
    
    # === 3. SPECTRAL SIMILARITY CHECK ===
    print(f"\n{'='*80}")
    print(f"3. Spectral Similarity Analysis")
    print(f"{'='*80}")
    
    # Compute PSD for all channels
    print(f"Computing PSDs...")
    psds, freqs = mne.time_frequency.psd_array_welch(
        n2_data, sfreq, fmin=1, fmax=40, n_fft=2048, 
        n_per_seg=512, n_overlap=256, verbose=False
    )
    
    spectral_scores = {}
    
    for i, ch_name in enumerate(eeg_channels):
        ch_psd = psds[i, :]
        
        if np.all(np.isnan(ch_psd)):
            spectral_scores[ch_name] = {'distance': 0, 'z_score': 0}
            continue
        
        # Find neighbors
        neighbor_indices = np.where(adjacency[i, :])[0]
        
        if len(neighbor_indices) == 0:
            spectral_scores[ch_name] = {'distance': 0, 'z_score': 0}
            continue
        
        # Compute spectral distance to each neighbor (Euclidean in log space)
        distances = []
        for neighbor_idx in neighbor_indices:
            neighbor_psd = psds[neighbor_idx, :]
            if np.all(np.isnan(neighbor_psd)):
                continue
            
            # Log transform for better comparison
            ch_log = np.log10(ch_psd + 1e-12)
            neighbor_log = np.log10(neighbor_psd + 1e-12)
            
            # Euclidean distance
            dist = np.sqrt(np.mean((ch_log - neighbor_log)**2))
            distances.append(dist)
        
        if len(distances) == 0:
            spectral_scores[ch_name] = {'distance': 0, 'z_score': 0}
            continue
        
        # Average distance
        avg_distance = np.mean(distances)
        
        # Z-score: how unusual is this distance?
        all_distances = [spectral_scores.get(ch, {}).get('distance', 0) 
                        for ch in eeg_channels[:i]]
        if len(all_distances) > 10:
            global_mean = np.mean([d for d in all_distances if d > 0])
            global_std = np.std([d for d in all_distances if d > 0])
            z_score = (avg_distance - global_mean) / global_std if global_std > 0 else 0
        else:
            z_score = 0
        
        spectral_scores[ch_name] = {'distance': avg_distance, 'z_score': z_score}
    
    # Recompute z-scores globally after all channels processed
    all_distances = [stats['distance'] for stats in spectral_scores.values() if stats['distance'] > 0]
    if len(all_distances) > 0:
        global_mean = np.mean(all_distances)
        global_std = np.std(all_distances)
        for ch_name in spectral_scores:
            dist = spectral_scores[ch_name]['distance']
            spectral_scores[ch_name]['z_score'] = (dist - global_mean) / global_std if global_std > 0 else 0
    
    # Find channels with weird spectra
    bad_spectral = [ch for ch, stats in spectral_scores.items() 
                   if stats['z_score'] > spectral_threshold]
    
    print(f"Spectral z-score threshold: {spectral_threshold}")
    print(f"Channels with weird spectra: {len(bad_spectral)}")
    if len(bad_spectral) > 0 and len(bad_spectral) <= 20:
        for ch in bad_spectral:
            print(f"  {ch}: z={spectral_scores[ch]['z_score']:.2f}")
    
    # === SUMMARY ===
    all_bad = list(set(bad_correlation + bad_amplitude + bad_spectral))
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Bad correlation: {len(bad_correlation)}")
    print(f"Bad amplitude: {len(bad_amplitude)}")
    print(f"Bad spectral: {len(bad_spectral)}")
    print(f"Total unique bad channels: {len(all_bad)}")
    if len(all_bad) > 0:
        print(f"\nAll problematic channels: {', '.join(sorted(all_bad))}")
    
    # Save results
    results = {
        'bad_correlation': bad_correlation,
        'bad_amplitude': bad_amplitude,
        'bad_spectral': bad_spectral,
        'all_bad': all_bad,
        'correlation_scores': correlation_scores,
        'amplitude_scores': amplitude_scores,
        'spectral_scores': spectral_scores,
    }
    
    # Optional: Create visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nCreating validation plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Correlation scores
        corr_values = [correlation_scores.get(ch, 0) for ch in eeg_channels]
        axes[0].hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(correlation_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {correlation_threshold}')
        axes[0].set_xlabel('Average Neighbor Correlation')
        axes[0].set_ylabel('Number of Channels')
        axes[0].set_title(f'Correlation Analysis\n{len(bad_correlation)} bad channels')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Amplitude z-scores
        amp_z_values = [amplitude_scores.get(ch, {}).get('z_score', 0) for ch in eeg_channels]
        axes[1].hist(amp_z_values, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(amplitude_threshold, color='red', linestyle='--', 
                       label=f'Threshold: ±{amplitude_threshold}')
        axes[1].axvline(-amplitude_threshold, color='red', linestyle='--')
        axes[1].set_xlabel('Amplitude Z-Score (vs neighbors)')
        axes[1].set_ylabel('Number of Channels')
        axes[1].set_title(f'Amplitude Analysis\n{len(bad_amplitude)} bad channels')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Spectral z-scores
        spec_z_values = [spectral_scores.get(ch, {}).get('z_score', 0) for ch in eeg_channels]
        axes[2].hist(spec_z_values, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(spectral_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {spectral_threshold}')
        axes[2].set_xlabel('Spectral Dissimilarity Z-Score')
        axes[2].set_ylabel('Number of Channels')
        axes[2].set_title(f'Spectral Analysis\n{len(bad_spectral)} bad channels')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f'{subject_id}_channel_validation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    print(f"{'='*80}\n")
    return results


def save_validation_results_to_csv(all_results, output_dir):
    """
    Save channel validation results to CSV files (one per subject).
    Each file has rows=channels, columns=metrics for that subject.
    
    Args:
        all_results: Dict mapping subject_id to validation results
        output_dir: Directory to save CSV files
    """
    print(f"\n{'='*80}")
    print(f"Saving validation results to CSV files")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    total_channels = 0
    total_bad = 0
    for subject_id, results in all_results.items():
        if results is None:
            print(f"⚠ Skipping {subject_id}: No results")
            continue
        
        correlation_scores = results.get('correlation_scores', {})
        amplitude_scores = results.get('amplitude_scores', {})
        spectral_scores = results.get('spectral_scores', {})
        
        # Get all channels
        all_channels = set(correlation_scores.keys()) | set(amplitude_scores.keys()) | set(spectral_scores.keys())
        
        # Create rows for this subject
        rows = []
        for channel in sorted(all_channels):
            row = {
                'channel': channel,
                'correlation': correlation_scores.get(channel, 0),
                'amplitude_rms': amplitude_scores.get(channel, {}).get('rms', 0),
                'amplitude_zscore': amplitude_scores.get(channel, {}).get('z_score', 0),
                'spectral_distance': spectral_scores.get(channel, {}).get('distance', 0),
                'spectral_zscore': spectral_scores.get(channel, {}).get('z_score', 0),
                'bad_correlation': channel in results.get('bad_correlation', []),
                'bad_amplitude': channel in results.get('bad_amplitude', []),
                'bad_spectral': channel in results.get('bad_spectral', []),
                'is_bad': channel in results.get('all_bad', []),
            }
            rows.append(row)
        
        # Create DataFrame for this subject
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = output_dir / f"{subject_id}_channel_validation.csv"
        df.to_csv(csv_path, index=False)
        
        # Update totals
        n_channels = len(df)
        n_bad = df['is_bad'].sum()
        percent_bad = (n_bad / n_channels * 100) if n_channels > 0 else 0
        
        total_channels += n_channels
        total_bad += n_bad
        
        print(f"✓ {subject_id}: {n_channels} channels, {n_bad} bad ({percent_bad:.1f}%) → {csv_path.name}")
    
    # Calculate average percentage
    avg_percent_bad = (total_bad / total_channels * 100) if total_channels > 0 else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"CSV Export Summary:")
    print(f"{'='*80}")
    print(f"  Files saved: {len(all_results)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Average bad channels: {avg_percent_bad:.1f}%")
    print(f"{'='*80}\n")


def main():
    """Main function to run PSD analysis for all subjects."""
    DATA_DIR = Path(BASE_DIR) / "control_clean"
    HYPNO_DIR = Path(BASE_DIR) / "HC_hypno"
    OUTPUT_DIR = Path("young_control")
    HYPNO_FREQ = 1  # 1-second epochs
    CHANNELS_TO_ANALYZE = ["E89", "E130"]
    
    # plot_psd_all_subjects(DATA_DIR, HYPNO_DIR, OUTPUT_DIR, CHANNELS_TO_ANALYZE, HYPNO_FREQ)
    
    # Single subject example
    subjects = get_all_subjects(DATA_DIR)
    print(f"Found {len(subjects)} subjects: {subjects}")
    all_results = {}
    for sub in subjects[2:]:
        try:
            eeg_path = f"{DATA_DIR}/{sub}/{sub}_196-head-ch_resample250_filtered_scored_bad-epochs_bad-channels_avg-ref.fif"
            hypno_path = f"{HYPNO_DIR}/{sub}.txt"
            output_dir = f"young_control/channels_validation/"
            # plot_psd_by_sleep_stage(sub, eeg_path, hypno_path, "E101", output_dir, hypno_freq=1)
            # channels = find_roi_channels(sub, eeg_path, hypno_path, output_dir=output_dir)
            results = validate_channels_spatial_consistency(sub, eeg_path, hypno_path, output_dir=output_dir)
            all_results[sub] = results
        except Exception as e:
            print(f"Error processing subject {sub}: {str(e)}")
    
    # Save all results to CSV (one file per subject)
    save_validation_results_to_csv(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(all_results)} subjects")
    print(f"CSV files saved to: {output_dir}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
