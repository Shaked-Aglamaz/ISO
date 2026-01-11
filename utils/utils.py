import os
import glob
import re
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import BASE_DIR


def find_subject_fif_file(subject_dir, max_length=True):
    """Find appropriate .fif file for subject."""
    if not os.path.exists(subject_dir):
        print(f"Subject folder not found: {subject_dir}")
        return None

    all_files = glob.glob(os.path.join(subject_dir, "*"))
    base_fif_pattern = re.compile(r'.*\.fif$')  # ends with .fif
    numbered_fif_pattern = re.compile(r'.*-\d+\.fif$')  # ends with -X.fif
    base_files = [f for f in all_files 
                  if base_fif_pattern.match(os.path.basename(f)) 
                  and not numbered_fif_pattern.match(os.path.basename(f))]
    
    if base_files:
        if max_length:
            raw_path = max(base_files, key=lambda x: len(os.path.basename(x)))
        else:
            raw_path = min(base_files, key=lambda x: len(os.path.basename(x)))
        print(f"Using file: {os.path.basename(raw_path)}")
        return raw_path
    else:
        return None
    

def plot_sigma_envelope(self, times, data, amplitude_envelope, start_time, duration):
    spindle_mask = (self.spindles_df['Start'] >= start_time) & (self.spindles_df['Start'] <= start_time + duration)
    curr_spindles = self.spindles_df[spindle_mask]
    
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, data, color='black', lw=1, label="Sigma", alpha=0.7)
    ax.plot(times, amplitude_envelope, label="Sigma Envelope", color="red", linewidth=2)
    
    for i, (_, row) in enumerate(curr_spindles.iterrows()):
        label = "Spindle" if i == 0 else None
        ax.axvspan(row['Start'], row['Start'] + row['Duration'], color='blue', alpha=0.3, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f"Envelope of {self.target_channel} ({self.low_freq}-{self.high_freq} Hz)")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(data.min() - 10, data.max() + 10)
    
    xticks = np.arange(start_time, start_time + duration + 1)
    ax.set_xticks(xticks)
    xtick_labels = [str(tick) if tick % 5 == 0 else '' for tick in xticks]
    ax.set_xticklabels(xtick_labels)
    
    plt.legend()
    plt.tight_layout()
    plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channel}_sigma_envelope.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Segment plot saved to: {plot_path}")


def compare_raw_and_file_annotations(raw, annotations_file):
    """Compare annotations embedded in raw file with annotations from file."""
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return
    
    # Get annotations from raw file
    raw_annot = raw.annotations
    
    # Load annotations from file
    file_annot = mne.read_annotations(annotations_file)
    
    # Compare counts
    print(f"Raw file annotations: {len(raw_annot)}")
    print(f"Text file annotations: {len(file_annot)}")
    
    if len(raw_annot) != len(file_annot):
        print("⚠️ Different number of annotations!")
        return False
    
    # Compare each annotation
    all_match = True
    
    for i, (r_onset, r_dur, r_desc) in enumerate(zip(raw_annot.onset, raw_annot.duration, raw_annot.description)):
        f_onset = file_annot.onset[i]
        f_dur = file_annot.duration[i]
        f_desc = file_annot.description[i]
        
        # Use np.allclose for floating point comparison with 1 second tolerance
        onset_match = np.allclose(r_onset, f_onset, atol=1.0)  # 1 second tolerance
        dur_match = np.allclose(r_dur, f_dur, atol=1.0)  # 1 second tolerance
        desc_match = r_desc == f_desc
        
        if not (onset_match and dur_match and desc_match):
            print(f"Mismatch at index {i}:")
            print(f"  Raw: {r_desc} at {r_onset:.10f}s, duration {r_dur:.10f}s")
            print(f"  File: {f_desc} at {f_onset:.10f}s, duration {f_dur:.10f}s")
            print(f"  Onset diff: {abs(r_onset - f_onset):.2e}s")
            print(f"  Duration diff: {abs(r_dur - f_dur):.2e}s")
            print(f"  Description match: {desc_match}")
            all_match = False
    
    if all_match:
        print("✓ All annotations match!")
        return True
    else:
        print("✗ Some annotations don't match")
        return False


def get_all_subjects(main_dir):
    """Get list of all subject IDs from directory."""
    if not os.path.exists(main_dir):
        print(f"Main directory not found: {main_dir}")
        return []
    
    subject_dirs = [d for d in os.listdir(main_dir) 
                   if os.path.isdir(os.path.join(main_dir, d))]
    
    print(f"Found {len(subject_dirs)} subjects: {subject_dirs}")
    return subject_dirs


def load_google_sheet():
    """Load the main subject data from Google Sheets"""
    sheet_url = "https://docs.google.com/spreadsheets/d/1BE0Yu-wECLe0NkdIIvxGjYKNL84FRgUE/edit?gid=768201376#gid=768201376"
    # Convert to CSV export URL
    csv_url = sheet_url.replace('/edit?gid=', '/export?format=csv&gid=')
    
    df = pd.read_csv(csv_url)
    return df


def merge_consecutive_annotations(raw, inplace=True, tolerance=0.001):
    """
    Merge consecutive annotations of the same type into single annotations.
    """
    if not hasattr(raw, 'annotations') or len(raw.annotations) == 0:
        print("⚠ No annotations found to merge.")
        return raw if inplace else None
    
    orig_annot = raw.annotations
    n_original = len(orig_annot)
    
    # Convert annotations to list of dictionaries for easier manipulation
    annot_list = []
    for onset, duration, description in zip(orig_annot.onset, 
                                            orig_annot.duration, 
                                            orig_annot.description):
        annot_list.append({
            'onset': float(onset),
            'duration': float(duration),
            'description': str(description),
            'end': float(onset + duration)
        })
    
    # Sort by onset time
    annot_list.sort(key=lambda x: x['onset'])
    
    # Merge consecutive annotations of the same type
    merged = []
    i = 0
    
    while i < len(annot_list):
        current = annot_list[i].copy()
        
        # Look ahead for consecutive annotations of the same type
        j = i + 1
        while j < len(annot_list):
            next_annot = annot_list[j]
            
            # Check if:
            # 1. Same description (type)
            # 2. Consecutive or overlapping (within tolerance)
            is_same_type = next_annot['description'] == current['description']
            is_consecutive = next_annot['onset'] <= current['end'] + tolerance
            
            if is_same_type and is_consecutive:
                # Merge: extend the end time to include next annotation
                current['end'] = max(current['end'], next_annot['end'])
                current['duration'] = current['end'] - current['onset']
                j += 1
            else:
                # Different type or gap too large, stop merging this segment
                break
        
        merged.append(current)
        i = j  # Continue from the next unmerged annotation
    
    # Create new MNE Annotations object
    onsets = np.array([a['onset'] for a in merged])
    durations = np.array([a['duration'] for a in merged])
    descriptions = np.array([a['description'] for a in merged])
    
    new_annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=orig_annot.orig_time
    )
    
    # Print summary
    n_merged = len(new_annotations)
    reduction = n_original - n_merged
    reduction_pct = (reduction / n_original * 100) if n_original > 0 else 0
    
    print(f"✓ Merged {n_original} → {n_merged} annotations "
          f"(reduced by {reduction}, {reduction_pct:.1f}%)")
    
    # Apply changes
    if inplace:
        raw.set_annotations(new_annotations)
        return raw
    else:
        return new_annotations


def print_annotation_summary(raw):
    """
    Print a summary of annotations in the raw object.
    """
    if not hasattr(raw, 'annotations') or len(raw.annotations) == 0:
        print("No annotations found.")
        return
    
    from collections import Counter
    
    annot = raw.annotations
    print(f"\nTotal annotations: {len(annot)}")
    print(f"Time span: {annot.onset[0]:.2f}s to {annot.onset[-1] + annot.duration[-1]:.2f}s")
    
    # Count by type
    type_counts = Counter(annot.description)
    print(f"\nAnnotations by type:")
    for desc, count in sorted(type_counts.items()):
        # Calculate total duration for this type
        total_dur = sum(dur for dur, d in zip(annot.duration, annot.description) if d == desc)
        avg_dur = total_dur / count if count > 0 else 0
        print(f"  {desc:20s}: {count:5d} annotations, "
              f"total {total_dur:8.1f}s, avg {avg_dur:6.2f}s each")
