import mne
import os
import shutil
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to import config and functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import BASE_DIR
from utils.utils import find_subject_fif_file
from utils.utils import merge_consecutive_annotations, print_annotation_summary, get_all_subjects
from sleepeegpy.dashboard import create_dashboard


mne.set_log_level("error")

def save_annotations_to_txt(subject_id, raw_path):
    """Save raw file annotations to txt file using MNE's built-in save method."""
    raw = mne.io.read_raw(raw_path, preload=False)
    
    if len(raw.annotations) == 0:
        print(f"  {subject_id}: No annotations found")
        return
    
    # Save to control_clean/SUB/SUB_cleaned_annotations.txt
    output_file = Path(f"{BASE_DIR}/control_clean/{subject_id}/{subject_id}_cleaned_annotations.txt")
    
    # Use MNE's built-in save method
    raw.annotations.save(output_file, overwrite=True)
    
    print(f"  {subject_id}: Saved {len(raw.annotations)} annotations to {output_file}")


def export_all_annotations():
    """Export annotations from all subjects in control_clean."""
    main_dir = f"{BASE_DIR}/control_clean/"
    
    if not os.path.exists(main_dir):
        print(f"Main directory not found: {main_dir}")
        return
    
    subject_dirs = [d for d in os.listdir(main_dir) 
                   if os.path.isdir(os.path.join(main_dir, d))]
    
    print(f"Found {len(subject_dirs)} subjects")
    print("="*60)
    
    for subject_id in subject_dirs:
        raw_path = find_subject_fif_file(subject_id)
        
        if raw_path:
            save_annotations_to_txt(subject_id, raw_path)
        else:
            print(f"  {subject_id}: No .fif file found")
    
    print("="*60)
    print("✓ Export complete")


def compare_annotation_files():
    """
    Compare original and cleaned annotation files for all subjects.
    - BAD annotations: Check if each cleaned BAD is contained within an original BAD
    - Stage annotations: Check if they remain exactly the same
    """
    main_dir = f"{BASE_DIR}/control_clean/"
    
    if not os.path.exists(main_dir):
        print(f"Main directory not found: {main_dir}")
        return
    
    subject_dirs = [d for d in os.listdir(main_dir) 
                   if os.path.isdir(os.path.join(main_dir, d))]
    
    issues_found = []
    subject_stats = []
    
    for subject_id in subject_dirs:
        orig_file = Path(f"{main_dir}/{subject_id}/{subject_id}_annotations.txt")
        cleaned_file = Path(f"{main_dir}/{subject_id}/{subject_id}_cleaned_annotations.txt")
        
        # Skip if either file doesn't exist
        if not orig_file.exists() or not cleaned_file.exists():
            continue
        
        # Load annotations using MNE
        orig_annotations = mne.read_annotations(orig_file)
        cleaned_annotations = mne.read_annotations(cleaned_file)
        
        # Sum BAD annotation durations for this subject
        orig_bad_duration = sum(duration for duration, desc in zip(orig_annotations.duration, orig_annotations.description) 
                                if 'bad' in desc.lower())
        cleaned_bad_duration = sum(duration for duration, desc in zip(cleaned_annotations.duration, cleaned_annotations.description) 
                                   if 'bad' in desc.lower())
        
        # Store stats for this subject
        subject_stats.append((subject_id, orig_bad_duration, cleaned_bad_duration))
        
        subject_issues = []
        
        # Check BAD annotations - each cleaned BAD should be contained in an original BAD
        for i, (onset, duration, desc) in enumerate(zip(cleaned_annotations.onset, 
                                                         cleaned_annotations.duration, 
                                                         cleaned_annotations.description)):
            if 'bad' in desc.lower():
                # Find if this BAD segment is contained within any original BAD with same name
                start_time = onset
                end_time = onset + duration
                
                found_container = False
                for (orig_onset, orig_duration, orig_desc) in zip(orig_annotations.onset,
                                                                                 orig_annotations.duration,
                                                                                 orig_annotations.description):
                    if orig_desc == desc:  # Same BAD type
                        orig_start = orig_onset
                        orig_end = orig_onset + orig_duration
                        
                        # Check if cleaned segment is contained within original using np.allclose for floating point comparison
                        if (np.allclose(orig_start, start_time, atol=1e-6) or orig_start <= start_time) and \
                           (np.allclose(end_time, orig_end, atol=1e-6) or end_time <= orig_end):
                            found_container = True
                            break
                
                if not found_container:
                    # import ipdb; ipdb.set_trace()
                    subject_issues.append(f"BAD '{desc}' at {start_time:.2f}s not contained in original")
        
        # Check stage annotations - should be exactly the same
        stage_names = ['NREM1', 'NREM2', 'NREM3', 'REM', 'Wake', 'W', 'N1', 'N2', 'N3', 'R']
        
        orig_stages = [(onset, duration, desc) for onset, duration, desc 
                       in zip(orig_annotations.onset, orig_annotations.duration, orig_annotations.description)
                       if any(stage in desc for stage in stage_names)]
        
        cleaned_stages = [(onset, duration, desc) for onset, duration, desc 
                          in zip(cleaned_annotations.onset, cleaned_annotations.duration, cleaned_annotations.description)
                          if any(stage in desc for stage in stage_names)]
        
        if len(orig_stages) != len(cleaned_stages):
            subject_issues.append(f"Stage count mismatch: {len(orig_stages)} orig vs {len(cleaned_stages)} cleaned")
        else:
            # Compare each stage annotation using np.allclose for floating point comparison
            for (o_onset, o_dur, o_desc), (c_onset, c_dur, c_desc) in zip(orig_stages, cleaned_stages):
                if not (np.allclose(o_onset, c_onset, atol=1e-6) and np.allclose(o_dur, c_dur, atol=1e-6) and o_desc == c_desc):
                    subject_issues.append(f"Stage mismatch: orig '{o_desc}' at {o_onset:.2f}s vs cleaned '{c_desc}' at {c_onset:.2f}s")
                    break  # Only report first mismatch per subject
        
        if subject_issues:
            issues_found.append((subject_id, subject_issues))
        else:
            # Print BAD duration stats for this subject
            percentage = (cleaned_bad_duration / orig_bad_duration * 100) if orig_bad_duration > 0 else 0
            print(f"✓ {subject_id}: BAD {cleaned_bad_duration:.1f}s / {orig_bad_duration:.1f}s ({percentage:.1f}%)")
    
    # Print final summary
    if issues_found:
        print(f"\nFound issues in {len(issues_found)} subjects:")
        for subject_id, issues in issues_found:
            print(f"\n  {subject_id}:")
            for issue in issues:
                print(f"    - {issue}")


def find_detrended_bouts_files(base_dir="RD43_MA_Hann_N2"):
    """
    Search through all subdirectories in base_dir for analysis_summary.txt files
    and find those where 'Low-Frequency Detrended Bouts' is not 0.
    
    Parameters:
    -----------
    base_dir : str
        Directory to search (default: "RD43_MA_Hann_N2")
    
    Returns:
    --------
    list : List of tuples (file_path, detrended_count)
    """
    base_path = Path(base_dir)
    results = []
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return results
    
    # Find all .txt files in subdirectories
    txt_files = list(base_path.glob("*_output/*.txt"))
    
    print(f"Searching through {len(txt_files)} text files in {base_dir}...")
    print("="*60)
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
                
            # Look for the line with "Low-Frequency Detrended Bouts:"
            for line in content.split('\n'):
                if "Low-Frequency Detrended Bouts:" in line:
                    # Extract the number
                    parts = line.split(':')
                    if len(parts) >= 2:
                        count_str = parts[1].strip()
                        try:
                            count = int(count_str)
                            if count != 0:
                                results.append((str(txt_file), count))
                                print(f"Found: {txt_file.name} - Detrended bouts: {count}")
                        except ValueError:
                            print(f"Warning: Could not parse count in {txt_file.name}: '{count_str}'")
                    break
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    print("="*60)
    if results:
        print(f"\n✓ Found {len(results)} files with non-zero detrended bouts:")
        for file_path, count in results:
            print(f"  - {Path(file_path).name}: {count} bouts")
    else:
        print("✓ No files found with non-zero detrended bouts (all are 0)")
    
    return results


def analyze_no_spindles_by_channel(notes_file="notes/notes.txt"):
    """
    Parse notes.txt for 'No spindles found' entries and group by channel.
    Shows how many subjects had no spindles for each channel.
    """
    from collections import defaultdict
    
    notes_path = Path(notes_file)
    if not notes_path.exists():
        print(f"Notes file not found: {notes_file}")
        return
    
    # Dictionary to store: channel -> set of subjects
    channel_subjects = defaultdict(set)
    
    with open(notes_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Look for lines like: "SUBJECT - No spindles found in channel CHANNEL"
            if " - No spindles found in channel " in line:
                parts = line.split(" - No spindles found in channel ")
                if len(parts) == 2:
                    subject = parts[0].strip()
                    channel = parts[1].strip()
                    channel_subjects[channel].add(subject)
    
    if not channel_subjects:
        print("No 'No spindles found' entries found in notes.txt")
        return
    
    # Sort channels by number of subjects (descending), then by channel name
    sorted_channels = sorted(channel_subjects.items(), 
                            key=lambda x: (-len(x[1]), x[0]))
    
    print("="*80)
    print("Channels with No Spindles Found (grouped by channel)")
    print("="*80)
    print(f"{'Channel':<10} {'Count':<8} Subjects")
    print("-"*80)
    
    for channel, subjects in sorted_channels:
        subject_list = sorted(subjects)
        subjects_str = ", ".join(subject_list)
        print(f"{channel:<10} {len(subjects):<8} ({subjects_str})")
    
    print("-"*80)
    print(f"Total unique channels with no spindles: {len(channel_subjects)}")
    print(f"Total unique subjects affected: {len(set().union(*channel_subjects.values()))}")
    print("="*80)


def compute_correlation_scores(n2_data, eeg_channels, adjacency, correlation_threshold, manual_bad_channels):
    """
    Compute correlation scores between each channel and its spatial neighbors.
    
    Args:
        n2_data: N2 data array (channels x timepoints)
        eeg_channels: List of channel names
        adjacency: Adjacency matrix (channels x channels)
        correlation_threshold: Threshold for flagging bad channels
        manual_bad_channels: Set of manually marked bad channels to exclude
    
    Returns:
        tuple: (correlation_scores dict, bad_correlation list, zero_variance_channels list)
    """
    print(f"\n{'='*80}")
    print(f"1. Neighbor Correlation Analysis")
    print(f"{'='*80}")
    
    correlation_scores = {}
    zero_variance_channels = []  # Track channels with flat signals
    min_neighbors_required = 2  # Minimum number of good neighbors needed
    
    for i, ch_name in enumerate(eeg_channels):
        # Skip manually bad channels
        if ch_name in manual_bad_channels:
            correlation_scores[ch_name] = None  # Mark as skipped
            continue
        
        ch_data = n2_data[i, :]
        ch_valid = ch_data[~np.isnan(ch_data)]
        
        if len(ch_valid) < 100:  # Skip if too little data
            correlation_scores[ch_name] = 0
            continue
        
        # Check for zero variance in the channel itself (flat signal)
        if np.std(ch_valid) == 0:
            correlation_scores[ch_name] = None
            zero_variance_channels.append(ch_name)
            continue
        
        # Find neighbors, excluding manually bad ones
        neighbor_indices = np.where(adjacency[i, :])[0]
        good_neighbor_indices = [idx for idx in neighbor_indices 
                                if eeg_channels[idx] not in manual_bad_channels]
        
        if len(good_neighbor_indices) < min_neighbors_required:
            correlation_scores[ch_name] = None  # Not enough good neighbors
            continue
        
        # Compute correlation with each good neighbor
        correlations = []
        for neighbor_idx in good_neighbor_indices:
            neighbor_data = n2_data[neighbor_idx, :]
            
            # Find common valid samples
            valid_mask = ~(np.isnan(ch_data) | np.isnan(neighbor_data))
            if valid_mask.sum() < 100:
                continue
            
            # Check for zero variance (prevents division by zero warning)
            ch_valid_samples = ch_data[valid_mask]
            neighbor_valid_samples = neighbor_data[valid_mask]
            
            if np.std(ch_valid_samples) == 0 or np.std(neighbor_valid_samples) == 0:
                continue  # Skip if either channel has no variance
            
            corr = np.corrcoef(ch_valid_samples, neighbor_valid_samples)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Average correlation with neighbors
        if len(correlations) >= min_neighbors_required:
            avg_corr = np.mean(correlations)
            correlation_scores[ch_name] = avg_corr
        else:
            correlation_scores[ch_name] = None  # Not enough valid correlations
    
    # Find channels with low correlation (excluding None/skipped)
    bad_correlation = [ch for ch, corr in correlation_scores.items() 
                      if corr is not None and corr < correlation_threshold]
    
    # Count unreliable channels (excluding manually bad channels AND zero-variance channels)
    unreliable = [ch for ch, corr in correlation_scores.items() 
                 if corr is None and ch not in manual_bad_channels and ch not in zero_variance_channels]
    
    print(f"Correlation threshold: {correlation_threshold}")
    print(f"Channels with low neighbor correlation: {len(bad_correlation)}")
    if len(bad_correlation) > 0 and len(bad_correlation) <= 20:
        print(f"  {', '.join(bad_correlation)}")
    if len(zero_variance_channels) > 0:
        print(f"Zero-variance channels (flat signal, likely reference): {len(zero_variance_channels)}")
        if len(zero_variance_channels) <= 10:
            print(f"  {', '.join(zero_variance_channels)}")
    if len(unreliable) > 0:
        print(f"Unreliable results (too few good neighbors): {len(unreliable)}")
        if len(unreliable) <= 10:
            print(f"  {', '.join(unreliable)}")
    
    return correlation_scores, bad_correlation, zero_variance_channels


def compute_amplitude_scores(n2_data, eeg_channels, adjacency, amplitude_threshold, manual_bad_channels):
    """
    Compute RMS amplitude z-scores relative to spatial neighbors.
    
    Args:
        n2_data: N2 data array (channels x timepoints)
        eeg_channels: List of channel names
        adjacency: Adjacency matrix (channels x channels)
        amplitude_threshold: Z-score threshold for flagging bad channels
        manual_bad_channels: Set of manually marked bad channels to exclude
    
    Returns:
        tuple: (amplitude_scores dict, bad_amplitude list)
    """
    print(f"\n{'='*80}")
    print(f"2. RMS Amplitude Analysis")
    print(f"{'='*80}")
    
    amplitude_scores = {}
    min_neighbors_required = 2  # Minimum number of good neighbors needed
    
    for i, ch_name in enumerate(eeg_channels):
        # Skip manually bad channels
        if ch_name in manual_bad_channels:
            amplitude_scores[ch_name] = {'rms': None, 'z_score': None}
            continue
        
        ch_data = n2_data[i, :]
        ch_valid = ch_data[~np.isnan(ch_data)]
        
        if len(ch_valid) < 100:
            amplitude_scores[ch_name] = {'rms': 0, 'z_score': 0}
            continue
        
        ch_rms = np.sqrt(np.mean(ch_valid**2))
        
        # Find neighbors, excluding manually bad ones
        neighbor_indices = np.where(adjacency[i, :])[0]
        good_neighbor_indices = [idx for idx in neighbor_indices 
                                if eeg_channels[idx] not in manual_bad_channels]
        
        if len(good_neighbor_indices) < min_neighbors_required:
            amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': None}
            continue
        
        # Compute RMS for good neighbors
        neighbor_rms_list = []
        for neighbor_idx in good_neighbor_indices:
            neighbor_data = n2_data[neighbor_idx, :]
            neighbor_valid = neighbor_data[~np.isnan(neighbor_data)]
            if len(neighbor_valid) < 100:
                continue
            neighbor_rms = np.sqrt(np.mean(neighbor_valid**2))
            neighbor_rms_list.append(neighbor_rms)
        
        if len(neighbor_rms_list) < min_neighbors_required:
            amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': None}
            continue
        
        # Z-score relative to neighbors
        neighbor_mean = np.mean(neighbor_rms_list)
        neighbor_std = np.std(neighbor_rms_list)
        z_score = (ch_rms - neighbor_mean) / neighbor_std if neighbor_std > 0 else 0
        
        amplitude_scores[ch_name] = {'rms': ch_rms, 'z_score': z_score}
    
    # Find channels with extreme amplitude (excluding None/skipped)
    bad_amplitude = [ch for ch, stats in amplitude_scores.items() 
                     if stats['z_score'] is not None and abs(stats['z_score']) > amplitude_threshold]
    
    # Count unreliable channels (excluding manually bad channels)
    unreliable = [ch for ch, stats in amplitude_scores.items() 
                 if stats['z_score'] is None and ch not in manual_bad_channels]
    
    print(f"Amplitude z-score threshold: ±{amplitude_threshold}")
    print(f"Channels with extreme amplitude: {len(bad_amplitude)}")
    if len(bad_amplitude) > 0 and len(bad_amplitude) <= 20:
        for ch in bad_amplitude:
            print(f"  {ch}: z={amplitude_scores[ch]['z_score']:.2f}")
    if len(unreliable) > 0:
        print(f"Unreliable results (too few good neighbors): {len(unreliable)}")
        if len(unreliable) <= 10:
            print(f"  {', '.join(unreliable)}")
    
    return amplitude_scores, bad_amplitude


def compute_spectral_scores(n2_data, eeg_channels, adjacency, sfreq, spectral_threshold, manual_bad_channels):
    """
    Compute spectral dissimilarity scores between each channel and its neighbors.
    
    Args:
        n2_data: N2 data array (channels x timepoints)
        eeg_channels: List of channel names
        adjacency: Adjacency matrix (channels x channels)
        sfreq: Sampling frequency
        spectral_threshold: Z-score threshold for flagging bad channels
        manual_bad_channels: Set of manually marked bad channels to exclude
    
    Returns:
        tuple: (spectral_scores dict, bad_spectral list)
    """
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
    min_neighbors_required = 2  # Minimum number of good neighbors needed
    
    for i, ch_name in enumerate(eeg_channels):
        # Skip manually bad channels
        if ch_name in manual_bad_channels:
            spectral_scores[ch_name] = {'distance': None, 'z_score': None}
            continue
        
        ch_psd = psds[i, :]
        
        if np.all(np.isnan(ch_psd)):
            spectral_scores[ch_name] = {'distance': 0, 'z_score': 0}
            continue
        
        # Find neighbors, excluding manually bad ones
        neighbor_indices = np.where(adjacency[i, :])[0]
        good_neighbor_indices = [idx for idx in neighbor_indices 
                                if eeg_channels[idx] not in manual_bad_channels]
        
        if len(good_neighbor_indices) < min_neighbors_required:
            spectral_scores[ch_name] = {'distance': None, 'z_score': None}
            continue
        
        # Compute spectral distance to each good neighbor (Euclidean in log space)
        distances = []
        for neighbor_idx in good_neighbor_indices:
            neighbor_psd = psds[neighbor_idx, :]
            if np.all(np.isnan(neighbor_psd)):
                continue
            
            # Log transform for better comparison
            ch_log = np.log10(ch_psd + 1e-12)
            neighbor_log = np.log10(neighbor_psd + 1e-12)
            
            # Euclidean distance
            dist = np.sqrt(np.mean((ch_log - neighbor_log)**2))
            distances.append(dist)
        
        if len(distances) < min_neighbors_required:
            spectral_scores[ch_name] = {'distance': None, 'z_score': None}
            continue
        
        # Average distance
        avg_distance = np.mean(distances)
        spectral_scores[ch_name] = {'distance': avg_distance, 'z_score': 0}  # z_score computed later
    
    # Recompute z-scores globally after all channels processed (excluding manual bads and None values)
    all_distances = [stats['distance'] for stats in spectral_scores.values() 
                    if stats['distance'] is not None and stats['distance'] > 0]
    
    if len(all_distances) > 0:
        global_mean = np.mean(all_distances)
        global_std = np.std(all_distances)
        for ch_name in spectral_scores:
            dist = spectral_scores[ch_name]['distance']
            if dist is not None and dist > 0:
                spectral_scores[ch_name]['z_score'] = (dist - global_mean) / global_std if global_std > 0 else 0
            elif dist is None:
                spectral_scores[ch_name]['z_score'] = None
    
    # Find channels with weird spectra (excluding None/skipped)
    bad_spectral = [ch for ch, stats in spectral_scores.items() 
                   if stats['z_score'] is not None and stats['z_score'] > spectral_threshold]
    
    # Count unreliable channels (excluding manually bad channels)
    unreliable = [ch for ch, stats in spectral_scores.items() 
                 if stats['z_score'] is None and ch not in manual_bad_channels]
    
    print(f"Spectral z-score threshold: {spectral_threshold}")
    print(f"Channels with weird spectra: {len(bad_spectral)}")
    if len(bad_spectral) > 0 and len(bad_spectral) <= 20:
        for ch in bad_spectral:
            print(f"  {ch}: z={spectral_scores[ch]['z_score']:.2f}")
    if len(unreliable) > 0:
        print(f"Unreliable results (too few good neighbors): {len(unreliable)}")
        if len(unreliable) <= 10:
            print(f"  {', '.join(unreliable)}")
    
    return spectral_scores, bad_spectral


def validate_channels_spatial_consistency(
    subject_id,
    eeg_path,
    hypno_path,
    hypno_freq=1,
    correlation_thresh=0.3,
    amplitude_thresh=3.0,
    spectral_thresh=3.0,
    output_dir=None,
):
    """
    Validate channels using neighbor-consistency checks on clean N2 data.
    Strategy (powerful for dense EGI nets):
    1. Correlation with neighbors: channels with unusually low correlation are suspicious
    2. RMS amplitude vs neighbors: persistently higher/lower amplitude is suspicious
    3. Spectral similarity vs neighbors: weird spectrum relative to neighbors is suspicious
    This catches "looks fine in raw but spatially inconsistent" channels that ruin topomaps.
    
    Important: Manually marked bad channels are excluded from all calculations.
    """
    print(f"\n{'='*80}")
    print(f"Validating spatial consistency for: {subject_id}")
    print(f"{'='*80}")
    
    # Load manually marked bad channels
    eeg_path = Path(eeg_path)
    bad_channels_file = eeg_path.parent / f"{subject_id}_bad_channels.txt"
    manual_bad_channels = set()
    
    if bad_channels_file.exists():
        with open(bad_channels_file, 'r') as f:
            manual_bad_channels = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(manual_bad_channels)} manually marked bad channels from: {bad_channels_file.name}")
        if len(manual_bad_channels) > 0 and len(manual_bad_channels) <= 20:
            print(f"  {', '.join(sorted(manual_bad_channels))}")
    else:
        print(f"No bad channels file found: {bad_channels_file.name}")
    
    # Load raw data
    raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
    
    # Load cleaned annotations if available
    annotations_file = eeg_path.parent / f"{subject_id}_cleaned_annotations.txt"
    if annotations_file.exists():
        try:
            annotations = mne.read_annotations(annotations_file)
            raw.set_annotations(annotations)
        except Exception as e:
            print(f"⚠ Warning: Could not load annotations from {annotations_file.name}: {str(e)}")
    
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
    
    # Run three validation checks
    correlation_scores, bad_correlation, zero_variance_channels = compute_correlation_scores(
        n2_data, eeg_channels, adjacency, correlation_thresh, manual_bad_channels
    )
    amplitude_scores, bad_amplitude = compute_amplitude_scores(
        n2_data, eeg_channels, adjacency, amplitude_thresh, manual_bad_channels
    )
    spectral_scores, bad_spectral = compute_spectral_scores(
        n2_data, eeg_channels, adjacency, sfreq, spectral_thresh, manual_bad_channels
    )
    
    # === SUMMARY ===
    all_bad = list(set(bad_correlation + bad_amplitude + bad_spectral))
    
    # Convert zero_variance_channels to a set for efficient lookup
    zero_variance_set = set(zero_variance_channels)
    
    # Count unreliable channels (too few good neighbors, excluding manually bad AND zero-variance)
    unreliable_corr = [ch for ch, score in correlation_scores.items() 
                      if score is None and ch not in manual_bad_channels and ch not in zero_variance_set]
    unreliable_amp = [ch for ch, stats in amplitude_scores.items() 
                     if stats.get('z_score') is None and ch not in manual_bad_channels and ch not in zero_variance_set]
    unreliable_spec = [ch for ch, stats in spectral_scores.items() 
                      if stats.get('z_score') is None and ch not in manual_bad_channels and ch not in zero_variance_set]
    all_unreliable = list(set(unreliable_corr + unreliable_amp + unreliable_spec))
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Manually marked bad (excluded from analysis): {len(manual_bad_channels)}")
    print(f"Zero-variance channels (flat signal, likely reference): {len(zero_variance_channels)}")
    print(f"Bad correlation: {len(bad_correlation)}")
    print(f"Bad amplitude: {len(bad_amplitude)}")
    print(f"Bad spectral: {len(bad_spectral)}")
    print(f"Total unique bad channels (newly detected): {len(all_bad)}")
    print(f"Unreliable (too few good neighbors): {len(all_unreliable)}")
    if len(zero_variance_channels) > 0 and len(zero_variance_channels) <= 10:
        print(f"\nZero-variance channels: {', '.join(sorted(zero_variance_channels))}")
    if len(all_bad) > 0:
        print(f"\nNewly detected problematic channels: {', '.join(sorted(all_bad))}")
    if len(all_unreliable) > 0 and len(all_unreliable) <= 15:
        print(f"Unreliable channels (inspect manually): {', '.join(sorted(all_unreliable))}")
    
    # Save results
    results = {
        'manual_bad_channels': list(manual_bad_channels),
        'zero_variance_channels': zero_variance_channels,
        'bad_correlation': bad_correlation,
        'bad_amplitude': bad_amplitude,
        'bad_spectral': bad_spectral,
        'all_bad': all_bad,
        'unreliable': all_unreliable,
        'correlation_scores': correlation_scores,
        'amplitude_scores': amplitude_scores,
        'spectral_scores': spectral_scores,
    }
    
    # Optional: Create visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nCreating validation plots...")
        
        # Count unreliable channels per metric (excluding manually bad AND zero-variance channels)
        unreliable_corr_count = sum(1 for ch in eeg_channels 
                                   if correlation_scores.get(ch) is None 
                                   and ch not in manual_bad_channels 
                                   and ch not in zero_variance_set)
        unreliable_amp_count = sum(1 for ch in eeg_channels 
                                  if amplitude_scores.get(ch, {}).get('z_score') is None 
                                  and ch not in manual_bad_channels 
                                  and ch not in zero_variance_set)
        unreliable_spec_count = sum(1 for ch in eeg_channels 
                                   if spectral_scores.get(ch, {}).get('z_score') is None 
                                   and ch not in manual_bad_channels 
                                   and ch not in zero_variance_set)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Correlation scores (exclude None values for unreliable channels)
        corr_values = [correlation_scores.get(ch, 0) for ch in eeg_channels 
                      if correlation_scores.get(ch) is not None]
        axes[0].hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(correlation_thresh, color='red', linestyle='--', label=f'Threshold: {correlation_thresh}')
        axes[0].set_xlabel('Average Neighbor Correlation')
        axes[0].set_ylabel('Number of Channels')
        axes[0].set_title(f'Correlation Analysis\n{len(bad_correlation)} bad | {unreliable_corr_count} unreliable')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Amplitude z-scores (exclude None values for unreliable channels)
        amp_z_values = [amplitude_scores.get(ch, {}).get('z_score', 0) for ch in eeg_channels 
                       if amplitude_scores.get(ch, {}).get('z_score') is not None]
        axes[1].hist(amp_z_values, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(amplitude_thresh, color='red', linestyle='--', label=f'Threshold: ±{amplitude_thresh}')
        axes[1].axvline(-amplitude_thresh, color='red', linestyle='--')
        axes[1].set_xlabel('Amplitude Z-Score (vs neighbors)')
        axes[1].set_ylabel('Number of Channels')
        axes[1].set_title(f'Amplitude Analysis\n{len(bad_amplitude)} bad | {unreliable_amp_count} unreliable')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Spectral z-scores (exclude None values for unreliable channels)
        spec_z_values = [spectral_scores.get(ch, {}).get('z_score', 0) for ch in eeg_channels 
                        if spectral_scores.get(ch, {}).get('z_score') is not None]
        axes[2].hist(spec_z_values, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(spectral_thresh, color='red', linestyle='--', label=f'Threshold: {spectral_thresh}')
        axes[2].set_xlabel('Spectral Dissimilarity Z-Score')
        axes[2].set_ylabel('Number of Channels')
        axes[2].set_title(f'Spectral Analysis\n{len(bad_spectral)} bad | {unreliable_spec_count} unreliable')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add overall summary text at the bottom
        summary_text = (f'Subject: {subject_id} | '
                       f'Manual bad: {len(manual_bad_channels)} | '
                       f'Zero-variance: {len(zero_variance_channels)} | '
                       f'Unreliable: {len(all_unreliable)} | '
                       f'Newly detected bad: {len(all_bad)}')
        
        # Adjust layout to make room for summary text at bottom
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom (5%)
        
        # Add summary text below the plots
        fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
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
                'manually_bad': channel in results.get('manual_bad_channels', []),
                'zero_variance': channel in results.get('zero_variance_channels', []),
                'correlation': correlation_scores.get(channel, 0),
                'amplitude_rms': amplitude_scores.get(channel, {}).get('rms', 0),
                'amplitude_zscore': amplitude_scores.get(channel, {}).get('z_score', 0),
                'spectral_distance': spectral_scores.get(channel, {}).get('distance', 0),
                'spectral_zscore': spectral_scores.get(channel, {}).get('z_score', 0),
                'bad_correlation': channel in results.get('bad_correlation', []),
                'bad_amplitude': channel in results.get('bad_amplitude', []),
                'bad_spectral': channel in results.get('bad_spectral', []),
                'is_bad': channel in results.get('all_bad', []),
                'unreliable': channel in results.get('unreliable', []),
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


def old_main():
    dir = f"{BASE_DIR}/elderly_control/"
    subjects = get_all_subjects(dir)
    errs = []
    for sub in subjects:
        # results = validate_channels_spatial_consistency(sub, eeg_path, hypno_path, output_dir=output_dir)
        # all_results[sub] = results
        
        print(f"\nProcessing subject: {sub}")
        sub_dir = f"{dir}/{sub}/saved_raw/CleaningPipe/"
        fif_path = find_subject_fif_file(sub, sub_dir)
        if not fif_path:
            errs.append(f"No .fif file found for subject {sub}")
            continue
        
        raw = mne.io.read_raw(fif_path, preload=False)
        merge_consecutive_annotations(raw, inplace=True)
        print_annotation_summary(raw)
        
        output_dir = Path(f"{dir}/{sub}")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = output_dir / f"{sub}_cleaned_hypno_annotations.txt"
        
        raw.annotations.save(annotations_path, overwrite=True)
        print(f"  ✓ Saved cleaned annotations: {annotations_path}")

    # Save all results to CSV (one file per subject)
    # save_validation_results_to_csv(all_results, output_dir)
    print("\n" + "="*60)
    if errs:
        print(f"Errors encountered ({len(errs)}):")
        for err in errs:
            print(f"  - {err}")
    else:
        print("✓ All subjects processed successfully!")
    print("="*60)


## from older version of step2:
def load_bads_from_csv(csv_path):
    # === 2. Load validation results from CSV ===
    validation_bad_channels = set()
    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                
                # Check if 'is_bad' column exists
                if 'is_bad' not in df.columns:
                    print(f"⚠ Warning: CSV does not have 'is_bad' column. Available columns: {list(df.columns)}")
                else:
                    # Extract channels where is_bad == True
                    validation_bad_channels = set(df[df['is_bad'] == True]['channel'].tolist())
                    print(f"✓ Loaded {len(validation_bad_channels)} bad channels from validation CSV")
                    if len(validation_bad_channels) > 0 and len(validation_bad_channels) <= 20:
                        print(f"  {', '.join(sorted(validation_bad_channels))}")
            except Exception as e:
                print(f"⚠ Warning: Could not load validation CSV: {str(e)}")
        else:
            print(f"⚠ Warning: CSV file not found: {csv_path}")


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


def analyze_n2_outliers_across_subjects(csv_dir):
    """
    Analyze N2 sigma power outliers across all subjects and visualize on EGI 256 montage.
    
    Args:
        csv_dir: Directory path containing CSV files with N2_Sigma_Power_Outliers column
    
    Returns:
        dict: Channel name -> count of how many CSV files it appears in as outlier
    """
    csv_dir = Path(csv_dir)
    
    if not csv_dir.exists():
        print(f"Directory not found: {csv_dir}")
        return {}
    
    print(f"{'='*80}")
    print(f"Analyzing N2 Outliers Across Subjects")
    print(f"{'='*80}")
    print(f"Directory: {csv_dir}\n")
    
    # Find all CSV files in the directory (top-level only)
    csv_files = list(csv_dir.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {csv_dir}")
        return {}
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Dictionary to count channel occurrences
    channel_counts = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if the column exists
            if 'N2_Sigma_Power_Outliers' not in df.columns:
                print(f"⚠ Skipping {csv_file.name}: Column 'N2_Sigma_Power_Outliers' not found")
                continue
            
            # Get outlier channels (skip empty cells)
            outliers = df['N2_Sigma_Power_Outliers'].dropna()
            outliers = outliers[outliers != '']  # Remove empty strings
            
            # Count each unique channel in this file
            for channel in outliers.unique():
                channel = str(channel).strip()
                if channel:  # Make sure it's not empty
                    channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            print(f"✓ {csv_file.name}: {len(outliers)} outlier entries, {len(outliers.unique())} unique channels")
            
        except Exception as e:
            print(f"✗ Error reading {csv_file.name}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Total unique outlier channels: {len(channel_counts)}")
    print(f"Files processed: {len(csv_files)}")
    
    if len(channel_counts) > 0:
        # Sort by count (descending)
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop outlier channels (by frequency):")
        print(f"{'-'*80}")
        print(f"{'Channel':<12} {'Count':<10} {'Percentage':<10}")
        print(f"{'-'*80}")
        for channel, count in sorted_channels[:20]:  # Show top 20
            percentage = (count / len(csv_files)) * 100
            print(f"{channel:<12} {count:<10} {percentage:.1f}%")
        
        if len(sorted_channels) > 20:
            print(f"... and {len(sorted_channels) - 20} more channels")
    
    # Create topographic plot
    print(f"\n{'='*80}")
    print(f"Creating topographic plot...")
    print(f"{'='*80}")
    
    try:
        # Create a standard EGI 256 montage
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        
        # Get channel names from montage
        montage_ch_names = montage.ch_names
        
        # Create info object with EGI 256 channels
        info = mne.create_info(ch_names=montage_ch_names, sfreq=250, ch_types='eeg')
        info.set_montage(montage)
        
        # Create a dummy evoked object for plotting
        # Set all channels to 0 (gray), outliers to 1 (red)
        data = np.zeros(len(montage_ch_names))
        
        # Mark outlier channels as 1
        outlier_channels = set(channel_counts.keys())
        for i, ch_name in enumerate(montage_ch_names):
            if ch_name in outlier_channels:
                data[i] = 1
        
        evoked = mne.EvokedArray(data[:, np.newaxis], info, tmin=0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Plot topomap with binary colormap (gray for normal, red for outliers)
        im, _ = mne.viz.plot_topomap(
            evoked.data[:, 0],
            evoked.info,
            axes=ax,
            show=False,
            cmap='Greys',
            vlim=(0, 1),
            contours=0,
            sensors=True,
            names=None,
        )
        
        # Overlay red markers for outlier channels with labels
        if len(outlier_channels) > 0:
            outlier_indices = [i for i, ch in enumerate(montage_ch_names) if ch in outlier_channels]
            pos = mne.channels.layout._find_topomap_coords(evoked.info, picks=outlier_indices)
            
            # Plot red dots
            ax.scatter(pos[:, 0], pos[:, 1], c='red', s=100, marker='o', 
                      edgecolors='darkred', linewidths=1.5, alpha=0.8, zorder=10,
                      label=f'N2 Outliers ({len(outlier_channels)})')
            
            # Add channel labels slightly below the dots
            for idx, (x, y) in zip(outlier_indices, pos):
                ch_name = montage_ch_names[idx]
                
                # Add text with tiny downward offset - centered horizontally
                ax.text(x, y - 0.003, ch_name,  # Tiny downward shift
                       fontsize=7, fontweight='bold', color='darkred',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='darkred', alpha=0.8, linewidth=0.8),
                       zorder=11)
        
        ax.set_title(f'N2 Sigma Power Outliers Across {len(csv_files)} Subjects\n'
                    f'{len(outlier_channels)} channels flagged as outliers',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        if len(outlier_channels) > 0:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = csv_dir / 'n2_outliers_topography.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved topographic plot: {plot_path}")
        
    except Exception as e:
        print(f"✗ Error creating topographic plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*80}\n")
    
    return channel_counts


def main():
    # channel_counts = analyze_n2_outliers_across_subjects("young_control/sigma_boxplot/")
    # print(channel_counts)

    mne.set_log_level('WARNING')
    dir_path = Path("F:/gennadiy/ella_processed/")
    subjects =  [f for f in dir_path.iterdir() if f.is_dir()]
    for sub in subjects:
        # fif_path = find_subject_fif_file(sub/Path("CleaningPipe"))
        fif_path = dir_path / sub / "CleaningPipe" / "cleaned_raw.fif"
        print(f"{'*' * 10} {sub.name} {'*' * 10}")
        raw = mne.io.read_raw_fif(fif_path, preload=False)


        # 1. Check the filenames attribute (often contains the path to the original data)
        print(f"Associated Filenames: {raw.filenames}")

        # 2. Check the 'description' field in info (sometimes contains processing history)
        print(f"Description: {raw.info['description']}")
        print(f"{sub.name}: {len(raw.ch_names)} channels")



if __name__ == "__main__":
    # export_all_annotations()    
    # compare_annotation_files()
    # find_detrended_bouts_files("RD43_MA_Hann_N2")
    # analyze_no_spindles_by_channel()
    main()