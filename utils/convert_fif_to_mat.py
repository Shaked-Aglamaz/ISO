import mne
import numpy as np
import hdf5storage
import os
import scipy.io as sio

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import BASE_DIR

# --- 1. CONFIGURATION ---
subject = 'EL3002'
sub_dir = f"{BASE_DIR}/control_clean/{subject}"
file_path = f"{sub_dir}/{subject}_196-head-ch_resample250_filtered_scored_bad-epochs_bad-channels_avg-ref.fif"
output_path = f"{sub_dir}/{subject}_old_Preprocessed1.mat"
EPOCH_SEC = 30  # Standard sleep epoch length

# Mapping rules based on your specific labels
# Zurich format: Wake=0, N1=-1, N2=-2, N3=-3, REM=-5
STAGE_MAP = {
    'WAKE': 0, 
    'N1': -1,
    'NREM1': -1, 
    'N2': -2,
    'NREM2': -2, 
    'N3': -3,
    'NREM3': -3, 
    'REM': -5
}

# --- 2. LOAD DATA ---
print(f"Loading {file_path}...")
raw = mne.io.read_raw_fif(file_path, preload=True)

data = raw.get_data()  # Shape: (channels, samples)
sfreq = raw.info['sfreq']
n_channels = len(raw.ch_names)
n_samples = data.shape[1]
n_epochs = int(np.floor(n_samples / (sfreq * EPOCH_SEC)))

# --- 3. INITIALIZE ARRAYS ---
visnum = np.zeros(n_epochs)  # Default all to 0 (Wake)
artndxn = np.ones((n_channels, n_epochs))  # Default all to 1 (Clean)

# --- 4. DURATION-AWARE STAGE MAPPING ---
print("Mapping sleep stages and durations...")
for ann in raw.annotations:
    desc = ann['description']
    onset = ann['onset']
    duration = ann['duration']
    
    # Calculate epoch range
    start_ep = int(np.floor(onset / EPOCH_SEC))
    end_ep = int(np.floor((onset + duration) / EPOCH_SEC))
    
    # A. Handle Sleep Stages
    stage_val = None
    for key, val in STAGE_MAP.items():
        if key in desc.upper():
            stage_val = val
            break
            
    if stage_val is not None:
        # Fill all epochs covered by this duration
        visnum[max(0, start_ep) : min(end_ep + 1, n_epochs)] = stage_val

    # B. Handle Artifacts (BAD_ACQ_SKIP)
    if 'BAD' in desc.upper():
        # Mark artndxn as 0 (Bad) for all channels during these epochs
        # This prevents the Zurich script from using these segments
        artndxn[:, max(0, start_ep) : min(end_ep + 1, n_epochs)] = 0
        
        # Optional: Hard-zero the raw data for these segments
        sample_start = int(onset * sfreq)
        sample_end = int((onset + duration) * sfreq)
        data[:, sample_start:sample_end] = 0

# --- 5. PREPARE FOR MATLAB (v7.3) ---
# Organize into the 'EEG' struct format expected by Zurich scripts
dtype = [('labels', 'O')]
chanlocs_array = np.array([(name,) for name in raw.ch_names], dtype=dtype)
mat_content = {
    'EEG': {
        'data': (data * 1e6).astype('float32'),  # float32 saves 50% RAM in MATLAB
        'srate': float(sfreq),
        'nbchan': int(n_channels),
        'visnum': visnum.reshape(1, -1), # Must be 1xN row vector
        'artndxn': artndxn.astype('float32'),
        'epochl': float(EPOCH_SEC),
        'interpolatedch': np.array([]).astype('float32'),
        'chanlocs': chanlocs_array,
    }
}

# --- 6. SAVE CLEAN FILE ---
if os.path.exists(output_path):
    os.remove(output_path)  # Critical: Avoids "file signature not found" error

print(f"Saving to {output_path}...")
try:
    # oned_as='row' ensures 1D arrays look like [1 2 3] in MATLAB, not [1;2;3]
    hdf5storage.savemat(output_path, mat_content, format='7.3', oned_as='row')
    print("Successfully saved! You can now run the MATLAB diagnostic.")
    print(f"Final N2 Epoch Count: {np.sum(visnum == -2)} ({np.sum(visnum == -2)*0.5} mins)")
except Exception as e:
    print(f"Save failed: {e}")