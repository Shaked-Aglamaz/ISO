import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import mne
import numpy as np
import pandas as pd
from new_iso.mult_chan import extract_clean_sleep_bouts
from new_iso.morlet import calculate_gabor_wavelet
from new_iso.isfs_presence import extract_isfs_parameters
from new_iso.visualization import plot_bout_fft, plot_mean_spectrum_with_fit, plot_all_bouts_overlay
from utils.utils import find_subject_fif_file


frequency_steps = np.arange(13, 16.2, 0.2) 
source_folder = Path(r'I:\Shaked\ISO_data\control_clean')


def save_bouts_info(bout_metadata, s_freq, subject_id, output_dir):
    bout_details = []
    for bout_idx in range(bout_metadata.shape[1]):
        start_idx, end_idx = bout_metadata[:, bout_idx].astype(int)
        start_time_sec = start_idx / s_freq
        end_time_sec = end_idx / s_freq
        duration_sec = (end_idx - start_idx + 1) / s_freq

        bout_details.append({
            'Bout_Number': bout_idx,
            'Start_Time_sec': start_time_sec,
            'End_Time_sec': end_time_sec,
            'Duration_sec': duration_sec,
            'Start_Index': start_idx,
            'End_Index': end_idx
        })
    
    bout_details_df = pd.DataFrame(bout_details)
    bout_details_file = output_dir / f"{subject_id}_all_bouts_details.csv"
    bout_details_df.to_csv(bout_details_file, index=False)
    print(f"✓ Saved bout details: {bout_details_file.name} ({len(bout_details)} bouts)\n")


def save_analysis_summary(subject_id, channel_name, plot_data, pf, bw, auc, pp, s_freq, frequency_steps, output_dir):
    summary_txt_path = output_dir / f"{subject_id}_{channel_name}_analysis_summary.txt"
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"ISFS Analysis Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Subject: {subject_id}\n")
        f.write(f"Channel: {channel_name}\n\n")
        f.write(f"Data Processing:\n")
        f.write(f"  Sigma frequency range: {frequency_steps[0]:.1f} - {frequency_steps[-1]:.1f} Hz\n")
        f.write(f"  Sampling rate: {s_freq:.1f} Hz\n")
        f.write(f"  Number of N2 bouts: {plot_data['num_bouts']}\n")
        f.write(f"  Bouts with outliers removed (0-0.006 Hz): {plot_data['n_bouts_with_outliers']}\n\n")
        
        if plot_data['fitted_params'] is not None:
            amplitude, mu, sigma = plot_data['fitted_params']
            f.write(f"Gaussian Fitting Results:\n")
            f.write(f"  ✓ ISFS detected (fit successful)\n")
            f.write(f"  Peak Frequency (μ): {pf:.6f} Hz\n")
            f.write(f"  Bandwidth (2σ): {bw:.6f} Hz\n")
            f.write(f"  Peak Amplitude: {pp:.6f} AU\n")
            f.write(f"  Area Under Curve (AUC): {auc:.6f}\n")
            f.write(f"  Sigma (σ): {sigma:.6f} Hz\n")
            f.write(f"  Detection threshold: {plot_data['threshold']:.6f} AU\n")
        else:
            f.write(f"Gaussian Fitting Results:\n")
            f.write(f"  ✗ ISFS not detected (fit failed)\n")
            if plot_data['failure_reason'] is not None:
                f.write(f"  Failure reason: {plot_data['failure_reason']}\n")
        
        f.write(f"\n{'='*60}\n")
    print(f"  ✓ Analysis summary saved: {summary_txt_path.name}")


def save_channel_results(subject_id, channel_results, output_dir, s_freq, n_bouts, n_channels):
    results_df = pd.DataFrame(channel_results)
    summary_file = output_dir / f"{subject_id}_all_channels_summary.csv"
    
    # Calculate statistics (only count successful detections)
    channels_with_isfs = results_df['peak_frequency'].notna().sum()
    channels_excluded = n_channels - channels_with_isfs
    percentage_with_isfs = (channels_with_isfs / n_channels * 100)

    # Write header with metadata (matching step3_spectral format)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# Subject: {subject_id}\n")
        f.write(f"# ISFS Detection Summary:\n")
        f.write(f"#   Total valid channels analyzed: {n_channels}\n")
        f.write(f"#   Channels with ISFS detected: {channels_with_isfs} ({percentage_with_isfs:.1f}%)\n")
        f.write(f"#   Channels excluded (no ISFS): {channels_excluded} ({100-percentage_with_isfs:.1f}%)\n")
        f.write(f"#   Number of bouts: {n_bouts}\n")
        f.write(f"#   Sampling rate: {s_freq} Hz\n")
        f.write(f"#   Frequency range: {frequency_steps[0]}-{frequency_steps[-1]} Hz\n")
        f.write("#\n")
    
    # Append the dataframe
    results_df.to_csv(summary_file, mode='a', index=False)
    print(f"\n{'='*60}")
    print(f"SUMMARY: Saved results for ALL {n_channels} channels to {summary_file}")
    print(f"         ISFS detected: {channels_with_isfs}/{n_channels} ({percentage_with_isfs:.1f}%)")
    print(f"         Failed: {channels_excluded}/{n_channels} ({100-percentage_with_isfs:.1f}%)")
    print(f"{'='*60}\n")


def analyze_channel(channel_name, subject_id, output_dir, channel_data, s_freq, bout_metadata, n_bouts):
    print(f"Processing Channel {channel_name}...")
    
    channel_output_dir = output_dir / f"{subject_id}_{channel_name}_output"
    channel_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calculate the Wavelet Transform (Gabor-Morlet) -> (frequencies x samples) complex matrix
    complex_transform = calculate_gabor_wavelet(channel_data, s_freq, frequency_steps[0], frequency_steps[-1], freq_res=0.2)
    
    # 2. Extract the Amplitude Timecourse, absolute(amplitude) averaged across frequency bins (axis 0)
    amplitude_envelope = np.mean(np.abs(complex_transform), axis=0)
    
    # 3. Extract ISFS Parameters (PF, BW, AUC, PP) over relative power
    (pf, bw, auc, pp), plot_data = extract_isfs_parameters(amplitude_envelope, bout_metadata, s_freq)

    # Plot individual bout FFTs (save in channel folder)
    for bout_idx in range(n_bouts):
        plot_bout_fft(plot_data['frequencies'], plot_data['all_bout_spectra'][bout_idx], bout_idx, subject_id, channel_name, channel_output_dir)
    
    # Plot mean spectrum with Gaussian fit (save in channel folder)
    plot_mean_spectrum_with_fit(plot_data, subject_id, channel_name, channel_output_dir)
    
    # Plot all bouts overlay (save in channel folder)
    plot_all_bouts_overlay(plot_data['frequencies'], plot_data['all_bout_spectra'], plot_data['mean_power_no_baseline'], subject_id, channel_name, channel_output_dir)
    
    print(f"  ✓ Plots saved to {channel_output_dir}")
    
    # Save spectral power CSV (mean ± std across bouts)
    # Calculate std across bouts (without baseline correction)
    all_bout_spectra_array = np.array(plot_data['all_bout_spectra'])
    std_power = np.nanstd(all_bout_spectra_array, axis=0)

    spectral_df = pd.DataFrame({'frequency': plot_data['frequencies'], 'mean_power': plot_data['mean_power_no_baseline'], 'std_power': std_power})
    spectral_csv_path = channel_output_dir / f"{subject_id}_{channel_name}_spectral_power.csv"
    spectral_df.to_csv(spectral_csv_path, index=False)
    print(f"  ✓ Spectral power CSV saved: {spectral_csv_path.name}")
    
    # Save individual bout FFT power matrix (each row is a bout)
    bout_fft_df = pd.DataFrame(plot_data['power_fft_matrix'], columns=[f'freq_{freq:.6f}' for freq in plot_data['frequencies']])
    bout_fft_df.insert(0, 'bout_number', range(n_bouts))
    bout_fft_csv_path = channel_output_dir / f"{subject_id}_{channel_name}_bout_fft_power.csv"
    bout_fft_df.to_csv(bout_fft_csv_path, index=False)
    print(f"  ✓ Bout FFT power CSV saved: {bout_fft_csv_path.name}")
    
    # Save analysis summary text file
    save_analysis_summary(subject_id, channel_name, plot_data, pf, bw, auc, pp, s_freq, frequency_steps, channel_output_dir)
    
    # Return parameters and failure reason (None if successful)
    return pf, bw, auc, pp, plot_data['failure_reason']


def main():
    errors = {}
    for subject_path in [f for f in source_folder.iterdir() if f.is_dir()]:
        try:
            fif_path = find_subject_fif_file(subject_path)
            if not fif_path:
                continue
                
            subject_id = subject_path.name
            raw = mne.io.read_raw_fif(fif_path, preload=True)
            s_freq = raw.info['sfreq']
            
            # Extract Bouts directly from the raw object without manual epoching
            # n2_bout_data: (n_channels, n_samples) - concatenated N2 bout data for all channels
            # bout_metadata: (2, n_bouts) - start/end indices for each bout in concatenated data
            n2_bout_data, bout_metadata = extract_clean_sleep_bouts(raw)
            if n2_bout_data.size == 0 or n2_bout_data.shape[1] == 0:
                print(f"Skipping {subject_id} - no valid N2 bouts found.")
                continue

            n_bouts = bout_metadata.shape[1]
            pf_all_channels, bw_all_channels, auc_all_channels, pp_all_channels = [], [], [], []
            failure_reasons = []  # Track failure reasons for all channels
            subject_output_dir = Path(f"new_iso_results/{subject_id}")
            subject_output_dir.mkdir(parents=True, exist_ok=True)
            
            save_bouts_info(bout_metadata, s_freq, subject_id, subject_output_dir)

            # Loop through each channel in the extracted bout data
            for ch_idx in range(n2_bout_data.shape[0]):
                channel_name = raw.ch_names[ch_idx]
                channel_data = n2_bout_data[ch_idx, :]
                pf, bw, auc, pp, failure_reason = analyze_channel(channel_name, subject_id, subject_output_dir, channel_data, s_freq, bout_metadata, n_bouts)
                
                # Append results
                pf_all_channels.append(pf)
                bw_all_channels.append(bw)
                auc_all_channels.append(auc)
                pp_all_channels.append(pp)
                failure_reasons.append(failure_reason)

            # 4. Save channel results (all channels, successful and failed)
            channel_results = []
            for i, ch_name in enumerate(raw.ch_names):
                channel_results.append({
                    'channel': ch_name,
                    'peak_frequency': pf_all_channels[i],
                    'bandwidth': bw_all_channels[i],
                    'auc': auc_all_channels[i],
                    'peak_amplitude': pp_all_channels[i],
                    'bandwidth_sigma': bw_all_channels[i] / 2 if not np.isnan(bw_all_channels[i]) else np.nan,
                    'failure_reason': failure_reasons[i]
                })
            
            # Save in subject root directory (always save, includes both successful and failed channels)
            save_channel_results(subject_id, channel_results, subject_output_dir, s_freq, n_bouts, n2_bout_data.shape[0])
            
            print(f"Finished processing Subject: {subject_id}\n")
        
        except Exception as e:
            errors[subject_id] = str(e)
        
        finally:
            gc.collect()

    print(f"Number of errors: {len(errors)}\n")
    for subject_id, error_msg in errors.items():
        print(f"{subject_id}: {error_msg}\n")

if __name__ == "__main__":
    main()
