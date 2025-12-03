import os
import glob
import re
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import BASE_DIR


def find_subject_fif_file(subject_id):
    """Find appropriate .fif file for subject."""
    folder_path = f"{BASE_DIR}/control_clean/{subject_id}/"
    if not os.path.exists(folder_path):
        print(f"Subject folder not found: {folder_path}")
        return None
    
    all_files = glob.glob(os.path.join(folder_path, "*"))
    base_fif_pattern = re.compile(r'.*\.fif$')  # ends with .fif
    numbered_fif_pattern = re.compile(r'.*-\d+\.fif$')  # ends with -X.fif
    base_files = [f for f in all_files 
                  if base_fif_pattern.match(os.path.basename(f)) 
                  and not numbered_fif_pattern.match(os.path.basename(f))]
    
    if base_files:
        raw_path = max(base_files, key=lambda x: len(os.path.basename(x)))
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