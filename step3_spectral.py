import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from pathlib import Path
import textwrap
import time
import glob
import os
import re
from config import BASE_DIR, FACE_ELECTRODES, NECK_ELECTRODES

mne.set_log_level("error")
mne.viz.set_browser_backend('qt')
mne.set_config('MNE_BROWSER_THEME', 'dark')
SUB_TO_HYPNO_PATH = {'31': 'HC_V1/EL3006', '26': 'HC_V1/RD43', '9': 'HK5'}


class BoutInfo:
    """
    Class to store information about an N2 sleep bout.
    """
    def __init__(self, start_time, end_time, bout_id):
        self.start_time = start_time
        self.end_time = end_time
        self.id = bout_id
        self.duration = end_time - start_time

    def __str__(self):
        return f"Bout {self.id}: {self.start_time:.1f}-{self.end_time:.1f}s ({self.duration:.1f}s)"


class SpindleAnalyzer:
    def __init__(self, subject_id, raw_path, low_freq=13, high_freq=16, target_channels=['VREF'], 
                output_dir=None, min_duration=280):
        self.subject_id = subject_id
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_channels = target_channels
        self.output_dir = output_dir if output_dir else f"{subject_id}/{subject_id}_{target_channels[0]}_output"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._add_to_gitignore(subject_id)
        self.raw_path = raw_path
        self.min_duration = min_duration
        
        self.target_raw = None
        self.spindles_df = None
        self.n2_bouts = []
        self.bout_errors = {}
        self.results_df = None
        
        # For relative spectral power computation
        self._frequencies = []
        self._all_power_spectra = []

    def _add_to_gitignore(self, subject_id):
        """Add subject directory to .gitignore if not already present."""
        gitignore_path = Path('.gitignore')
        subject_pattern = f"{subject_id}/"        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            if subject_pattern not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n{subject_pattern}")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(f"# Subject data directories\n{subject_pattern}")

    def load_raw_data(self):
        raw_data = mne.io.read_raw(self.raw_path)
        annotation_desc = set(raw_data.annotations.description)
        if "NREM2" not in annotation_desc:
            print(f"ERROR: No 'NREM2' annotations found in the data.")
            print(f"Available annotations: {annotation_desc}")
            exit(1)
        
        if "BAD" not in annotation_desc:
            print("WARNING: No 'BAD' annotations found in the data.")
            print(f"Available annotations: {annotation_desc}")
        
        self.target_raw = raw_data.pick_channels(self.target_channels)
        self.target_raw.load_data()

    def detect_spindles(self, plot_raw=False):
        if self.target_raw is None:
            self.load_raw_data()
            
        spindles = yasa.spindles_detect(self.target_raw, freq_sp=(self.low_freq, self.high_freq))
        if spindles:
            self.spindles_df = spindles.summary()
            spindle_annotations = mne.Annotations(
                self.spindles_df['Start'].values, 
                self.spindles_df['Duration'].values, 
                ['Spindle_' + channel for channel in self.spindles_df['Channel']],
                orig_time=self.target_raw.annotations.orig_time
            )
            annotations = self.target_raw.annotations + spindle_annotations
            self.target_raw.set_annotations(annotations)
            if plot_raw:
                self.target_raw.plot()

    def _extract_valid_segments(self, n2_start, n2_end, bad_segments):
        """Extract valid segments from an N2 period, excluding BAD overlaps, and create BoutInfo objects."""
        if len(bad_segments) == 0:
            if (n2_end - n2_start) >= self.min_duration:
                self.n2_bouts.append(BoutInfo(n2_start, n2_end, bout_id=len(self.n2_bouts)))
            return

        # Sort BAD segments and extract valid parts between them
        bad_segments = bad_segments[np.argsort(bad_segments[:, 0])]
        valid_segments = []
        current_pos = n2_start
        
        for bad_start, bad_end in bad_segments:
            if current_pos < bad_start:
                valid_segments.append((current_pos, bad_start))
            current_pos = max(current_pos, bad_end)
        
        if current_pos < n2_end:
            valid_segments.append((current_pos, n2_end))

        for start, end in valid_segments:
            if (end - start) >= self.min_duration:
                self.n2_bouts.append(BoutInfo(start, end, bout_id=len(self.n2_bouts)))

    def extract_n2_bouts(self):
        """Extract valid N2 segments, excluding overlaps with BAD annotations."""
        ann = self.target_raw.annotations
        n2_mask = (ann.description == "NREM2") & (ann.duration >= self.min_duration)
        bad_mask = ann.description == "BAD"
        n2_segments = np.column_stack([ann.onset[n2_mask], ann.onset[n2_mask] + ann.duration[n2_mask]])
        bad_segments = np.column_stack([ann.onset[bad_mask], ann.onset[bad_mask] + ann.duration[bad_mask]])
        
        for n2_start, n2_end in n2_segments:
            # Find overlapping BAD segments
            overlaps = (bad_segments[:, 0] < n2_end) & (bad_segments[:, 1] > n2_start)
            overlapping_bads = bad_segments[overlaps]
            self._extract_valid_segments(n2_start, n2_end, overlapping_bads)
        
        print("")
        print(f"Found {len(self.n2_bouts)} valid N2 bouts (>= {self.min_duration}s)")

    @staticmethod
    def compute_envelope(data):
        """
        Compute amplitude envelope using Hilbert transform and recenter around 0.
        """
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
        return amplitude_envelope
    
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
        ax.set_title(f"Envelope of {self.target_channels[0]} ({self.low_freq}-{self.high_freq} Hz)")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(data.min() - 10, data.max() + 10)
        
        xticks = np.arange(start_time, start_time + duration + 1)
        ax.set_xticks(xticks)
        xtick_labels = [str(tick) if tick % 5 == 0 else '' for tick in xticks]
        ax.set_xticklabels(xtick_labels)
        
        plt.legend()
        plt.tight_layout()
        plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_sigma_envelope.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Segment plot saved to: {plot_path}")

    @staticmethod
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def compute_fft_power_spectrum(self, amplitude_envelope, min_freq=0, max_freq=0.1):
        """
        Compute FFT power spectrum of amplitude envelope and calculate relative power.
        Relative power = power at each frequency / mean power across 0-0.1 Hz range.
        """
        sfreq = self.target_raw.info['sfreq']
        n_samples = amplitude_envelope.shape[0]
        freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
        fft_values = np.fft.rfft(amplitude_envelope)
        fft_power = np.abs(fft_values) ** 2
        fft_power *= 1e6  # to avoid really small numbers
        fft_power = fft_power / (n_samples / sfreq)  # normalized by length
        
        # Extract frequency range of interest
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        x = freqs[mask]
        y = fft_power[mask]
        
        # Calculate relative power: divide by mean power in the 0-0.1 Hz range
        y_relative = y / np.mean(y)
        return x, y_relative

    def fit_gaussian_to_spectrum(self, x, y):
        # Params order: peak, mu, sigma
        max_idx = np.argmax(y)
        peak_amplitude = y[max_idx]
        peak_frequency = x[max_idx]
        
        # Estimate sigma from data spread (use a reasonable fraction of frequency range)
        frequency_range = x[-1] - x[0]
        initial_sigma = frequency_range / 20  # Start with 5% of frequency range (more conservative)
        p0 = [peak_amplitude, peak_frequency, initial_sigma]
        fitted_params, _ = curve_fit(self.gaussian, x, y, p0=p0)
        mu = fitted_params[1]
        if mu < 0:
            raise ValueError(f"Fitted negative peak frequency {mu:.4f}")
        return fitted_params

    def analyze_single_bout(self, bout_info):
        """
        Analyze a single N2 bout: crop raw data, filter to sigma frequencies, compute envelope, FFT, and fit Gaussian.
        Returns bout result and power spectrum data for aggregation.
        """
        # Crop the raw data to bout times first, then filter
        bout_raw = self.target_raw.copy().crop(tmin=bout_info.start_time, tmax=bout_info.end_time)
        bout_sigma = bout_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        bout_data = bout_sigma.get_data()[0]  # Get first (and only) channel
        amplitude_envelope = self.compute_envelope(bout_data)
        x, y = self.compute_fft_power_spectrum(amplitude_envelope)

        fitted_params = None
        try:
            fitted_params = self.fit_gaussian_to_spectrum(x, y)
        except Exception as fit_error:
            self.bout_errors[bout_info] = fit_error
            error_msg = f"  ✗ {bout_info} analysis failed: {fit_error}"
            print(textwrap.fill(error_msg, width=110, subsequent_indent="    "))
        
        bout_result = self.plot_fft_power_spectrum_bout(x, y, fitted_params, bout_info)
        
        # Return both the bout result and the power spectrum data for aggregation
        return bout_result, (x, y)

    def plot_fft_power_spectrum_bout(self, x, y, fitted_params, bout_info, min_freq=0, max_freq=0.1):
        """Plot FFT relative power spectrum for a bout."""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f'Relative Power ({self.target_channels[0]})')
        
        bout_result = None
        if fitted_params is not None:
            _, mu, sigma = fitted_params
            x_fit = np.linspace(min_freq, max_freq, 500)
            y_fit = self.gaussian(x_fit, *fitted_params)
            x1 = mu - sigma
            x2 = mu + sigma
            mask = (x_fit >= x1) & (x_fit <= x2)
            bandwidth_height = self.gaussian(x1, *fitted_params)
            area = simpson(x=x_fit[mask], y=y_fit[mask])
            # TODO: think about the threshold
            
            bout_result = {
                'bout_id': bout_info.id,
                'start_time': bout_info.start_time,
                'end_time': bout_info.end_time,
                'duration': bout_info.duration,
                'peak_frequency': mu,
                'bandwidth': 2 * sigma,
                'auc': area
            }
            
            plt.plot(x_fit, y_fit, label=f'Gaussian fit (μ={mu:.3f}, STD={sigma:.3f})')
            plt.hlines(bandwidth_height, x1, x2, colors="purple", label="Bandwidth")
            plt.fill_between(x_fit[mask], y_fit[mask], color='skyblue', alpha=0.5, label='±1 STD Area')
            title_suffix = ""
        else:
            # Failed Gaussian fit - plot FFT only
            title_suffix = " - FAILED"

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative Power')
        plt.title(f'FFT Relative Power Spectrum - Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s){title_suffix}\n'
                  f'Duration: {bout_info.duration:.1f}s')
        plt.legend()
        plt.tight_layout()

        plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_bout_{bout_info.id}_fft_power.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        return bout_result

    def analyze_segment(self, start_time, duration):
        """For visualization: analyze and plot a specific time segment."""
        
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles()
        
        # Crop the raw data to segment times first, then filter
        end_time = start_time + duration
        segment_raw = self.target_raw.copy().crop(tmin=start_time, tmax=end_time)
        segment_sigma = segment_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        sigma_data, times = segment_sigma.get_data(return_times=True)
        sigma_data = sigma_data[0] * 1e6  # convert from V to µV
        amplitude_envelope = self.compute_envelope(sigma_data)
        self.plot_sigma_envelope(times, sigma_data, amplitude_envelope, start_time, duration)

    def analyze_all_bouts(self, plot_raw=False):
        """
        Analyze all valid N2 bouts in the recording.
        """
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles(plot_raw)
        
            # Check if spindle detection failed
            if self.spindles_df is None:
                message = f"{self.subject_id} - No spindles found in channel {self.target_channels[0]}"
                print(message)
                notes_dir = Path("notes")
                notes_dir.mkdir(exist_ok=True)
                notes_file = notes_dir / "notes.txt"
                with open(notes_file, 'a') as f:
                    f.write(f"{message}\n")
                return
        
        if not self.n2_bouts:
            self.extract_n2_bouts()
        
        if not self.n2_bouts:
            print("No valid N2 bouts found!")
            return
        
        bout_results = []
        for bout_info in self.n2_bouts:
            try:
                result, (freqs, power) = self.analyze_single_bout(bout_info)
                if result is not None:
                    bout_results.append(result)
                    self._all_power_spectra.append(power)
                    self._frequencies.append(freqs)
                    print(f"  ✓ {bout_info} analysis succeeded")
            except Exception as e:
                self.bout_errors[bout_info] = e
                print(f"  ✗ {bout_info} analysis failed: {e}")
        
        print(f"Completed analysis of {len(bout_results)}/{len(self.n2_bouts)} bouts\n")
        if bout_results:
            self.results_df = pd.DataFrame(bout_results)
        
    def compute_mean_spectral_power(self):
        """
        Compute the mean spectral power over all bouts.
        Uses interpolation to create a common frequency grid for averaging.
        """
        if not self._frequencies or not self._all_power_spectra:
            print("No power spectra available. Run analyze_all_bouts() first.")
            return None, None, None
        
        # Create common frequency grid with finest resolution for interpolation
        highest_min = max(freqs[0] for freqs in self._frequencies)
        lowest_max = min(freqs[-1] for freqs in self._frequencies)
        finest_resolution = min(freqs[1] - freqs[0] for freqs in self._frequencies)
        common_freqs = np.arange(highest_min, lowest_max, finest_resolution)
        
        # Interpolate all power spectra to common grid
        interpolated_spectra = []
        for freqs, power in zip(self._frequencies, self._all_power_spectra):
            interp_func = interp1d(freqs, power, bounds_error=False, fill_value='extrapolate')
            interpolated_power = interp_func(common_freqs)
            interpolated_spectra.append(interpolated_power)
        
        # Convert to numpy array and compute statistics
        power_matrix = np.array(interpolated_spectra)
        mean_power_spectrum = np.mean(power_matrix, axis=0)
        std_power_spectrum = np.std(power_matrix, axis=0)
        print(f"\nComputed mean spectral power from {len(interpolated_spectra)} bouts successfully.")
        
        return common_freqs, mean_power_spectrum, std_power_spectrum
    
    def plot_mean_spectral_power(self, frequencies=None, mean_power=None, std_power=None):
        """
        Plot the mean spectral power (averaged across all bouts).
        """
        if frequencies is None or mean_power is None:
            frequencies, mean_power, std_power = self.compute_mean_spectral_power()
            
        if frequencies is None:
            print("Cannot plot - no valid spectral data computed.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, mean_power, 'b-', linewidth=2, label='Mean Relative Power')
        
        if std_power is not None:
            plt.fill_between(frequencies, 
                           mean_power - std_power, 
                           mean_power + std_power, 
                           alpha=0.3, color='blue', 
                           label='±1 STD')
        
        if self.results_df is not None and not self.results_df.empty:
            mean_peak_freq = self.results_df['peak_frequency'].mean()
            plt.axvline(x=mean_peak_freq, color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean Peak Freq: {mean_peak_freq:.3f} Hz')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Power')
        plt.title(f'Mean Spectral Power - Subject {self.subject_id}\n'
                  f'across all N2 bouts ({self.low_freq}-{self.high_freq} Hz envelope)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_mean_spectral_power.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Save spectral data as CSV for later group analysis
        if frequencies is not None and mean_power is not None:
            assert len(frequencies) == len(mean_power), f"Length mismatch: frequencies={len(frequencies)}, mean_power={len(mean_power)}"
            
            if std_power is not None:
                assert len(std_power) == len(frequencies), f"Length mismatch: std_power={len(std_power)}, frequencies={len(frequencies)}"
                std_values = std_power
            else:
                std_values = np.zeros_like(mean_power)

            spectral_data = pd.DataFrame({'frequency': frequencies, 'mean_power': mean_power, 'std_power': std_values})
            csv_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_spectral_power.csv"
            spectral_data.to_csv(csv_path, index=False)
            print(f"Spectral data saved to: {csv_path}")
        
    def get_averaged_metrics(self):
        """Calculate averaged metrics across all successful bouts."""
        if self.results_df is None or self.results_df.empty:
            return None
        
        averaged_metrics = {
            'channel': self.target_channels[0],
            'avg_peak_frequency': self.results_df['peak_frequency'].mean(),
            'avg_bandwidth': self.results_df['bandwidth'].mean(),
            'avg_auc': self.results_df['auc'].mean(),
            'n_successful_bouts': len(self.results_df)
        }
        
        return averaged_metrics
        
    def get_summary(self):
        if self.results_df is None:
            print("No analysis results available. Run analyze_all_bouts() first.")
            return
        
        summary_lines = []
        summary_lines.append(f"Analysis Summary for Subject {self.subject_id}, Channel {self.target_channels[0]}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Target Channels: {self.target_channels}")
        summary_lines.append(f"Frequency Range: {self.low_freq}-{self.high_freq} Hz")
        summary_lines.append(f"Total Spindles Detected: {len(self.spindles_df) if self.spindles_df is not None else 0}")
        summary_lines.append(f"Total N2 Bouts Found: {len(self.n2_bouts)}")
        summary_lines.append(f"Failed Bouts: {len(self.bout_errors)}")
        summary_lines.append("")
        summary_lines.append("Per-Bout Analysis:")
        summary_lines.append(f"{'Bout':<6}{'Duration':<12}{'Peak Freq':<12}{'Bandwidth':<12}{'AUC':<10}")
        summary_lines.append("-" * 60)
        
        for _, row in self.results_df.iterrows():
            summary_lines.append(f"{row['bout_id']:<6}{row['duration']:<12.1f}{row['peak_frequency']:<12.3f}{row['bandwidth']:<12.3f}{row['auc']:<10.3f}")
        
        # Summary statistics using DataFrame
        summary_lines.append("")
        summary_lines.append("Summary Statistics:")
        summary_lines.append("-" * 60)
        summary_lines.append(f"Average bout duration: {self.results_df['duration'].mean():.1f}s (±{self.results_df['duration'].std():.1f})")
        summary_lines.append(f"Average peak frequency: {self.results_df['peak_frequency'].mean():.3f}Hz (±{self.results_df['peak_frequency'].std():.3f})")
        summary_lines.append(f"Average bandwidth: {self.results_df['bandwidth'].mean():.3f}Hz (±{self.results_df['bandwidth'].std():.3f})")
        summary_lines.append(f"Average AUC: {self.results_df['auc'].mean():.3f} (±{self.results_df['auc'].std():.3f})")
        
        for line in summary_lines:
            print(line)
        
        if self.bout_errors:
            summary_lines.append("")
            summary_lines.append("Analysis Errors:")
            summary_lines.append("-" * 60)
            for bout_info, error in self.bout_errors.items():
                error_type = type(error).__name__
                error_msg = str(error)
                summary_lines.append(f"Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s): {error_type} - {error_msg}")
        
        summary_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))

def analyze_all_channels(sub, raw_path):
    raw = mne.io.read_raw(raw_path, preload=False)
    excluded_channels = set(FACE_ELECTRODES + NECK_ELECTRODES)
    valid_channels = [ch for ch in raw.ch_names if ch not in excluded_channels and 'EMG' not in ch]
    
    channel_results = []
    all_bout_results = []
    for channel in valid_channels:
        print(f"\n{'='*60}")
        print(f"Analyzing subject: {sub}, channel: {channel}")
        print(f"{'='*60}")
        
        analyzer = SpindleAnalyzer(sub, raw_path, low_freq=13, high_freq=16, target_channels=[channel])
        analyzer.analyze_all_bouts()
        analyzer.get_summary()
        analyzer.plot_mean_spectral_power()
        avg_metrics = analyzer.get_averaged_metrics()
        if avg_metrics is not None:
            channel_results.append(avg_metrics)
        
        if analyzer.results_df is not None and not analyzer.results_df.empty:
            bout_df = analyzer.results_df.copy()
            cols = ['channel'] + bout_df.columns.tolist()
            bout_df['channel'] = channel
            bout_df = bout_df[cols]
            all_bout_results.append(bout_df)

    if channel_results:
        results_df = pd.DataFrame(channel_results)
        summary_file = f"{sub}/{sub}_all_channels_summary.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"\n{'='*60}")
        print(f"SUMMARY: Saved results for {len(channel_results)} channels to {summary_file}")
        print(f"{'='*60}")
    
    if all_bout_results:
        all_bouts_df = pd.concat(all_bout_results, ignore_index=True)
        all_bouts_file = f"{sub}/{sub}_all_bouts_details.csv"
        all_bouts_df.to_csv(all_bouts_file, index=False)
        print(f"DETAILED: Saved {len(all_bouts_df)} individual bout results to {all_bouts_file}")


def get_all_subjects(main_dir):
    """Get list of all subject IDs from the specified main directory."""
    if not os.path.exists(main_dir):
        print(f"Main directory not found: {main_dir}")
        return []
    
    subject_dirs = [d for d in os.listdir(main_dir) 
                   if os.path.isdir(os.path.join(main_dir, d))]
    
    print(f"Found {len(subject_dirs)} subjects: {subject_dirs}")
    return subject_dirs


def main():
    start_time = time.time()
    
    subject_dirs = get_all_subjects(f"{BASE_DIR}/control_clean/")
    if not subject_dirs:
        return
    
    processed_subjects = []
    failed_subjects = []
    
    # Process each subject
    for i, sub in enumerate(subject_dirs, 1):
        subject_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT {i}/{len(subject_dirs)}: {sub}")
        print(f"{'='*80}")
        
        try:
            folder_path = f"{BASE_DIR}/control_clean/{sub}/"
            
            all_files = glob.glob(os.path.join(folder_path, "*"))
            base_fif_pattern = re.compile(r'.*\.fif$')  # ends with .fif
            numbered_fif_pattern = re.compile(r'.*-\d+\.fif$')  # ends with -X.fif
            
            base_files = [f for f in all_files 
                          if base_fif_pattern.match(os.path.basename(f)) 
                          and not numbered_fif_pattern.match(os.path.basename(f))]
            
            if base_files:
                # Select the file with the longest filename
                raw_path = max(base_files, key=lambda x: len(os.path.basename(x)))
                print(f"Using file: {os.path.basename(raw_path)} (out of {len(base_files)} options)")
                
                analyze_all_channels(sub, raw_path)
                
                subject_end_time = time.time()
                subject_duration = subject_end_time - subject_start_time
                processed_subjects.append(sub)
                print(f"✓ Successfully processed subject {sub} in {subject_duration/60:.2f} minutes")
                
            else:
                subject_end_time = time.time()
                subject_duration = subject_end_time - subject_start_time
                print(f"✗ No base .fif file found in {folder_path} (checked in {subject_duration:.1f} seconds)")
                failed_subjects.append((sub, "No .fif file found"))
                
        except Exception as e:
            subject_end_time = time.time()
            subject_duration = subject_end_time - subject_start_time
            print(f"✗ Error processing subject {sub} after {subject_duration/60:.2f} minutes: {e}")
            failed_subjects.append((sub, str(e)))
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"🕒 Total time: {duration/60:.2f} minutes")
    print(f"✓ Successfully processed: {len(processed_subjects)}/{len(subject_dirs)} subjects")
    if processed_subjects:
        print(f"   {', '.join(processed_subjects)}")
    
    if failed_subjects:
        print(f"✗ Failed subjects: {len(failed_subjects)}")
        for sub, error in failed_subjects:
            print(f"   {sub}: {error}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()