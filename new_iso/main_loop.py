import sys
from pathlib import Path

# Add parent directory to path to import new_iso module
sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import numpy as np
import pandas as pd
from new_iso.mult_chan import extract_clean_sleep_bouts
from new_iso.morlet import calculate_gabor_wavelet
from new_iso.isfs_presence import extract_isfs_parameters
from utils.utils import find_subject_fif_file

# --- Configuration ---
frequency_steps = np.arange(13, 16.2, 0.2) 
source_folders = [Path(r'I:\Shaked\ISO_data\control_clean')]
EPOCH_LEN = 30

for folder in source_folders:
    for subject_path in [f for f in folder.iterdir() if f.is_dir()][1:2]:
        fif_path = find_subject_fif_file(subject_path)
        if not fif_path:
            continue
            
        raw = mne.io.read_raw_fif(fif_path, preload=False)
        raw.pick(['E8', 'E9'])
        raw.load_data()
        
        # 1 & 2. Extract Bouts directly from the raw object without manual epoching
        n2_bout_data, bout_metadata = extract_clean_sleep_bouts(raw)
        
        # Skip if no valid bouts found
        if n2_bout_data.size == 0 or n2_bout_data.shape[1] == 0:
            print(f"Skipping {subject_path.name} - no valid N2 bouts found.")
            continue

        # --- Continuing inside the Subject Loop, after extract_clean_sleep_bouts ---

        # Containers for this subject's channel results
        pf_all_channels = []
        bw_all_channels = []
        auc_all_channels = []
        pp_all_channels = []

        # Loop through each channel in the extracted bout data
        # n2_bout_data shape is (channels, samples)
        for ch_idx in range(n2_bout_data.shape[0]):
            print(f"Processing Channel {ch_idx}...")

            # 1. Calculate the Wavelet Transform (Gabor-Morlet)
            # This gives us a (frequencies x samples) complex matrix
            complex_transform = calculate_gabor_wavelet(
                data=n2_bout_data[ch_idx, :], 
                sampling_rate=raw.info['sfreq'],
                min_freq=frequency_steps[0], 
                max_freq=frequency_steps[-1], 
                freq_res=0.2
            )
            
            # 2. Extract the Amplitude Timecourse
            # Equivalent to MATLAB: powertimecourse = mean(abs(powertimecourse))'
            # We take the absolute (amplitude) and average across the frequency bins (axis 0)
            amplitude_envelope = np.mean(np.abs(complex_transform), axis=0)
            
            # 3. Extract ISFS Parameters (PF, BW, AUC, PP)
            # Using the relative power condition logic
            pf, bw, auc, pp = extract_isfs_parameters(
                power_timecourse=amplitude_envelope, 
                bout_locations=bout_metadata, 
                sampling_rate=raw.info['sfreq']
            )
            
            # Append results
            pf_all_channels.append(pf)
            bw_all_channels.append(bw)
            auc_all_channels.append(auc)
            pp_all_channels.append(pp)

        # 4. Save results in CSV format
        results_df = pd.DataFrame({
            'channel': raw.ch_names,
            'peak_frequency': pf_all_channels,
            'bandwidth': bw_all_channels,
            'auc': auc_all_channels,
            'peak_amplitude': pp_all_channels,
        })
        
        output_name = f"ISFS_results_{subject_path.name}.csv"
        
        # Write metadata header
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write(f"# Subject: {subject_path.name}\n")
            f.write(f"# Number of channels: {len(raw.ch_names)}\n")
            f.write(f"# Channel names: {', '.join(raw.ch_names)}\n")
            f.write(f"# Number of bouts: {bout_metadata.shape[1]}\n")
            f.write(f"# Sampling rate: {raw.info['sfreq']} Hz\n")
            f.write(f"# Frequency range: {frequency_steps[0]}-{frequency_steps[-1]} Hz\n")
            f.write("#\n")
        
        # Append the dataframe
        results_df.to_csv(output_name, mode='a', index=False)
        
        print(f"✓ Saved results to {output_name}")
        print(f"  {len(raw.ch_names)} channels, {bout_metadata.shape[1]} bouts")
        print(f"Finished processing Subject: {subject_path.name}")
        print(f"{'='*60}\n")
