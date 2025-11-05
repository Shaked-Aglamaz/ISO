import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
from scipy.signal import hilbert, detrend, filtfilt
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
    """N2 sleep bout information."""
    def __init__(self, start_time, end_time, bout_id):
        self.start_time = start_time
        self.end_time = end_time
        self.id = bout_id
        self.duration = end_time - start_time

    def __str__(self):
        return f"Bout {self.id}: {self.start_time:.1f}-{self.end_time:.1f}s ({self.duration:.1f}s)"


class SpindleAnalyzer:
    def __init__(self, subject_id, raw_path, low_freq=13, high_freq=16, target_channels=['VREF'], 
                output_dir=None, min_duration=280, apply_detrending=True, include_n3=False):
        self.subject_id = subject_id
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_channels = target_channels
        self.output_dir = output_dir if output_dir else f"{subject_id}/{subject_id}_{target_channels[0]}_output"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._add_to_gitignore(subject_id)
        self.raw_path = raw_path
        self.min_duration = min_duration
        self.apply_detrending = apply_detrending
        self.include_n3 = include_n3
        
        self.target_raw = None
        self.spindles_df = None
        self.n2_bouts = []
        self.bout_errors = {}
        self.results_df = None
        self.detrended_bouts = []
        
        # For relative spectral power computation
        self._frequencies = []
        self._all_power_spectra = []

    def _add_to_gitignore(self, subject_id):
        """Add subject directory to .gitignore."""
        gitignore_path = Path('.gitignore')
        subject_pattern = f"{subject_id}/"        
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if subject_pattern not in content:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{subject_pattern}")
        else:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(f"# Subject data directories\n{subject_pattern}")

    def load_raw_data(self):
        raw_data = mne.io.read_raw(self.raw_path)
        annotation_desc = set(raw_data.annotations.description)
        
        if self.include_n3:
            if "NREM2" not in annotation_desc and "NREM3" not in annotation_desc:
                print(f"ERROR: No 'NREM2' or 'NREM3' annotations found in the data.")
                print(f"Available annotations: {annotation_desc}")
                exit(1)
        else:
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
        """Extract valid segments from N2 period, excluding BAD overlaps."""
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
        """Extract valid N2 segments (and optionally N3) excluding BAD annotations."""
        ann = self.target_raw.annotations
        
        if self.include_n3:
            stage_mask = ((ann.description == "NREM2") | (ann.description == "NREM3")) & (ann.duration >= self.min_duration)
            stage_names = "N2/N3"
        else:
            stage_mask = (ann.description == "NREM2") & (ann.duration >= self.min_duration)
            stage_names = "N2"
            
        bad_mask = ann.description == "BAD"
        stage_segments = np.column_stack([ann.onset[stage_mask], ann.onset[stage_mask] + ann.duration[stage_mask]])
        bad_segments = np.column_stack([ann.onset[bad_mask], ann.onset[bad_mask] + ann.duration[bad_mask]])
        
        for stage_start, stage_end in stage_segments:
            overlaps = (bad_segments[:, 0] < stage_end) & (bad_segments[:, 1] > stage_start)
            overlapping_bads = bad_segments[overlaps]
            self._extract_valid_segments(stage_start, stage_end, overlapping_bads)
        
        print("")
        print(f"Found {len(self.n2_bouts)} valid {stage_names} bouts (>= {self.min_duration}s)")

    @staticmethod
    def compute_envelope(data):
        """Compute amplitude envelope using Hilbert transform and recenter around 0."""
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
    
    def _compute_basic_fft(self, amplitude_envelope):
        """Compute basic FFT power spectrum from amplitude envelope."""
        sfreq = self.target_raw.info['sfreq']
        n_samples = amplitude_envelope.shape[0]
        freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
        fft_values = np.fft.rfft(amplitude_envelope)
        fft_power = np.abs(fft_values) ** 2
        fft_power *= 1e6
        fft_power = fft_power / (n_samples / sfreq)
        return freqs, fft_power
    
    def detect_low_frequency_artifacts(self, amplitude_envelope):
        """Detect low-frequency artifacts and apply linear detrending if needed."""
        freqs, fft_power = self._compute_basic_fft(amplitude_envelope)
        
        low_freq_mask = (freqs >= 0) & (freqs <= 0.004)
        main_freq_mask = (freqs > 0.004) & (freqs <= 0.1)
        
        if not np.any(low_freq_mask) or not np.any(main_freq_mask):
            return amplitude_envelope, False
        
        max_power_low_freq = np.max(fft_power[low_freq_mask])
        max_power_main_range = np.max(fft_power[main_freq_mask])
        
        if max_power_low_freq > max_power_main_range:
            amplitude_envelope_detrended = detrend(amplitude_envelope, type='linear')
            return amplitude_envelope_detrended, True
        else:
            return amplitude_envelope, False
    
    def compute_fft_power_spectrum(self, amplitude_envelope, min_freq=0, max_freq=0.1, bout_id=None):
        """Compute FFT power spectrum with detrending (per-bout relative power only)."""
        
        if self.apply_detrending:
            amplitude_envelope, was_detrended = self.detect_low_frequency_artifacts(amplitude_envelope)
            if was_detrended and bout_id is not None:
                self.detrended_bouts.append(bout_id)
        
        freqs, fft_power = self._compute_basic_fft(amplitude_envelope)
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        x = freqs[mask]
        y = fft_power[mask]
        
        # Per-bout normalization: divide by bout's mean power
        y_relative = y / np.mean(y)
        return x, y_relative

    def fit_gaussian_to_spectrum(self, x, y):
        max_idx = np.argmax(y)
        peak_amplitude = y[max_idx]
        peak_frequency = x[max_idx]
        
        frequency_range = x[-1] - x[0]
        initial_sigma = frequency_range / 20
        p0 = [peak_amplitude, peak_frequency, initial_sigma]
        fitted_params, _ = curve_fit(self.gaussian, x, y, p0=p0)
        mu = fitted_params[1]
        if mu < 0:
            print(f"Warning: Gaussian fit produced negative peak frequency {mu:.4f}")
            return None
        return fitted_params

    def analyze_single_bout(self, bout_info):
        """Analyze single N2 bout: crop, filter sigma band, compute envelope, FFT."""
        bout_raw = self.target_raw.copy().crop(tmin=bout_info.start_time, tmax=bout_info.end_time)
        bout_sigma = bout_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        bout_data = bout_sigma.get_data()[0]
        amplitude_envelope = self.compute_envelope(bout_data)
        x, y = self.compute_fft_power_spectrum(amplitude_envelope, bout_id=bout_info.id)

        bout_result = {
            'bout_id': bout_info.id,
            'start_time': bout_info.start_time,
            'end_time': bout_info.end_time,
            'duration': bout_info.duration
        }
        
        self._save_bout_fft_data(x, y, bout_info)
        self.plot_bout_fft(x, y, bout_info)
        return bout_result, (x, y)

    def _save_bout_fft_data(self, frequencies, power, bout_info):
        """Save bout FFT data to CSV (relative power)."""
        fft_data = pd.DataFrame({
            'frequency': frequencies,
            'relative_power': power
        })
        
        csv_filename = f"{self.subject_id}_{self.target_channels[0]}_bout_{bout_info.id:02d}_fft.csv"
        csv_path = Path(self.output_dir) / csv_filename
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(f"# Bout ID: {bout_info.id}\n")
            f.write(f"# Start Time: {bout_info.start_time}s\n")
            f.write(f"# End Time: {bout_info.end_time}s\n")
            f.write(f"# Duration: {bout_info.duration}s\n")
            f.write("# \n")
        
        fft_data.to_csv(csv_path, mode='a', index=False)

    def plot_bout_fft(self, x, y, bout_info):
        """Plot FFT relative power spectrum for a bout."""
        title = (f'FFT Relative Power Spectrum - Bout {bout_info.id}\n'
                f'Time: {bout_info.start_time:.1f}-{bout_info.end_time:.1f}s, Duration: {bout_info.duration:.1f}s')
        filename_suffix = f"bout_{bout_info.id}_fft_power"
        self.plot_fft_spectrum(x, y, title, filename_suffix, show_gaussian=False)

    def plot_fft_spectrum(self, frequencies, power_data, title, filename_suffix, 
                         power_label=None, show_gaussian=False, show_baseline_info=False, 
                         baseline_power=None, show_popup=False):
        """Unified FFT spectrum plotting function for all scenarios."""
        if power_label is None:
            power_label = f'Relative Power ({self.target_channels[0]})'
        
        plt.figure(figsize=(10, 6))
        
        # 1. Main power spectrum
        plt.plot(frequencies, power_data, color='#2E86AB', linewidth=2, label=power_label)
        
        # 2. Baseline correction visualization (if requested)
        if show_baseline_info and baseline_power is not None:
            # Show baseline level line
            plt.axhline(y=baseline_power, color='#F18F01', linestyle='--', alpha=0.8, 
                       label=f'Baseline Level: {baseline_power:.3f} AU')
            # Show area to be subtracted
            plt.fill_between(frequencies, 0, baseline_power, 
                           color='#F18F01', alpha=0.3, 
                           label='Area to be Subtracted (Baseline Level)')
        
        # 3. Gaussian fit visualization (if requested and available)
        if show_gaussian and hasattr(self, 'fitted_params') and self.fitted_params is not None:
            peak_amplitude, mu, sigma = self.fitted_params
            x_fit = np.linspace(frequencies[0], frequencies[-1], 500)
            y_fit = self.gaussian(x_fit, *self.fitted_params)
            
            # Calculate AUC values
            x1 = mu - sigma
            x2 = mu + sigma
            mask = (x_fit >= x1) & (x_fit <= x2)
            bandwidth_height = self.gaussian(x1, *self.fitted_params)
            area = simpson(x=x_fit[mask], y=y_fit[mask])
            
            # Plot Gaussian components in order
            plt.plot(x_fit, y_fit, '#A23B72', linewidth=2, alpha=0.8, 
                    label=f'Gaussian Fit (μ={mu:.3f} Hz, σ={sigma:.3f})')
            plt.plot(mu, peak_amplitude, 'o', color='#A23B72', markersize=8, 
                    label=f'Peak: {peak_amplitude:.3f} AU')
            plt.hlines(bandwidth_height, x1, x2, colors="#F18F01", linewidth=2, label="Bandwidth")
            plt.fill_between(x_fit[mask], y_fit[mask], color='#A23B72', alpha=0.15, 
                           label=f'±1σ Area (AUC={area:.3f})')
        
        # 4. Standard plot formatting
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative Power (AU)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 5. Save and show
        plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_{filename_suffix}.png"
        plt.savefig(plot_path, dpi=300)
        
        if show_popup:
            plt.show()
        else:
            plt.close()
        
        return plot_path

    def _validate_spindles_and_bouts(self, plot_raw=False):
        """Helper function to validate spindle detection and N2 bouts before analysis."""
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles(plot_raw)
        
            # Check if spindle detection failed
            if self.spindles_df is None:
                message = f"{self.subject_id} - No spindles found in channel {self.target_channels[0]}"
                print(message)
                notes_dir = Path("notes")
                notes_dir.mkdir(exist_ok=True)
                notes_file = notes_dir / "notes.txt"
                with open(notes_file, 'a', encoding='utf-8') as f:
                    f.write(f"{message}\n")
                return False
        
        if not self.n2_bouts:
            self.extract_n2_bouts()
        
        if not self.n2_bouts:
            stage_desc = "N2/N3" if self.include_n3 else "N2"
            print(f"No valid {stage_desc} bouts found!")
            return False
        
        return True

    def analyze_all_bouts(self, plot_raw=False):
        """Analyze all valid N2 bouts in the recording."""
        if not self._validate_spindles_and_bouts(plot_raw):
            return
        
        bout_results = []
        for bout_info in self.n2_bouts:
            try:
                result, (freqs, power) = self.analyze_single_bout(bout_info)
                if result is not None:
                    bout_results.append(result)
                    self._all_power_spectra.append(power)
                    self._frequencies.append(freqs)
            except Exception as e:
                self.bout_errors[bout_info] = e
                print(f"  ✗ {bout_info} analysis failed: {e}")
        
        print(f"Completed analysis of {len(bout_results)}/{len(self.n2_bouts)} bouts")
        if bout_results:
            self.results_df = pd.DataFrame(bout_results)
        
        # Perform per-channel analysis: average, baseline correction, and Gaussian fitting
        if self._all_power_spectra:
            # Step 1: Average across bouts
            frequencies, mean_power, std_power = self.compute_mean_spectral_power()
            
            if frequencies is not None:
                # Step 2: Baseline correction (0.06-0.1 Hz)
                baseline_corrected = self.apply_baseline_correction(frequencies, mean_power)
                
                # Step 3: Gaussian fitting to baseline-corrected spectrum
                self.channel_result, self.fitted_params = self.fit_gaussian_to_mean_spectrum(
                    frequencies, baseline_corrected)
                
                # Store baseline-corrected data for plotting
                self._baseline_corrected_power = baseline_corrected
                
                print("✓ Per-channel analysis completed")
            else:
                print("✗ Could not compute mean spectral power")
        
    def compute_mean_spectral_power(self):
        """Compute mean spectral power over all bouts using interpolation."""
        if not self._frequencies or not self._all_power_spectra:
            print("No power spectra available. Run analyze_all_bouts() first.")
            return None, None, None
        
        # Create common frequency grid with finest resolution for interpolation
        highest_min = max(freqs[0] for freqs in self._frequencies)
        lowest_max = min(freqs[-1] for freqs in self._frequencies)
        finest_resolution = min(freqs[1] - freqs[0] for freqs in self._frequencies)
        common_freqs = np.arange(highest_min, lowest_max, finest_resolution)
        
        interpolated_spectra = []
        for freqs, power in zip(self._frequencies, self._all_power_spectra):
            interp_func = interp1d(freqs, power, bounds_error=False, fill_value='extrapolate')
            interpolated_power = interp_func(common_freqs)
            interpolated_spectra.append(interpolated_power)
        
        power_matrix = np.array(interpolated_spectra)
        mean_power_spectrum = np.mean(power_matrix, axis=0)
        std_power_spectrum = np.std(power_matrix, axis=0)
        
        return common_freqs, mean_power_spectrum, std_power_spectrum
    
    def apply_baseline_correction(self, frequencies, mean_power):
        """Apply baseline correction using 0.06-0.1 Hz range."""
        baseline_mask = (frequencies >= 0.06) & (frequencies <= 0.1)
        
        if not np.any(baseline_mask):
            print("Warning: No data in 0.06-0.1 Hz range for baseline correction")
            return mean_power
        
        baseline_power = np.mean(mean_power[baseline_mask])
        baseline_corrected = mean_power - baseline_power
        return baseline_corrected
    
    def fit_gaussian_to_mean_spectrum(self, frequencies, baseline_corrected_power):
        """Fit Gaussian to baseline-corrected mean spectrum (per-channel analysis)."""
        try:
            fitted_params = self.fit_gaussian_to_spectrum(frequencies, baseline_corrected_power)
            if fitted_params is None:
                failure_reason = "Invalid parameters (negative peak frequency)"
                print(f"Gaussian fitting failed - {failure_reason}")
                self.gaussian_fit_failure_reason = failure_reason
                return None, None
            
            peak_amplitude, peak_frequency, bandwidth_sigma = fitted_params
            
            # Calculate actual AUC of the fitted Gaussian (±1σ area)
            x_fit = np.linspace(frequencies[0], frequencies[-1], 500)
            y_fit = self.gaussian(x_fit, *fitted_params)
            x1 = peak_frequency - bandwidth_sigma
            x2 = peak_frequency + bandwidth_sigma
            mask = (x_fit >= x1) & (x_fit <= x2)
            actual_auc = simpson(x=x_fit[mask], y=y_fit[mask])
            
            # Store channel-level results
            channel_result = {
                'channel': self.target_channels[0],
                'peak_frequency': peak_frequency,
                'bandwidth': 2 * bandwidth_sigma,  # Convert sigma to full bandwidth
                'auc': actual_auc,  # Actual AUC of ±1σ area under Gaussian curve
                'peak_amplitude': peak_amplitude,
                'bandwidth_sigma': bandwidth_sigma
            }
            
            print(f"Channel {self.target_channels[0]} Gaussian fit:")
            print(f"  Peak frequency: {peak_frequency:.3f} Hz")
            print(f"  Bandwidth (2σ): {2 * bandwidth_sigma:.3f} Hz") 
            print(f"  Peak amplitude: {peak_amplitude:.3f}")
            
            return channel_result, fitted_params
            
        except Exception as e:
            failure_reason = f"Exception during fitting: {str(e)}"
            print(f"Failed to fit Gaussian to mean spectrum: {e}")
            self.gaussian_fit_failure_reason = failure_reason
            return None, None
    
    def plot_baseline_correction_preview(self, frequencies=None, mean_power=None):
        """Plot mean relative power before baseline correction, highlighting the subtraction area."""
        if frequencies is None or mean_power is None:
            frequencies, mean_power, _ = self.compute_mean_spectral_power()
            
        if frequencies is None:
            print("Cannot plot baseline preview - no valid spectral data computed.")
            return
        
        # Calculate baseline area (0.06-0.1 Hz)
        baseline_mask = (frequencies >= 0.06) & (frequencies <= 0.1)
        baseline_power = np.mean(mean_power[baseline_mask]) if np.any(baseline_mask) else 0
        
        title = (f'Pre-Baseline Correction - Subject {self.subject_id}\n'
                f'Channel {self.target_channels[0]} ({self.low_freq}-{self.high_freq} Hz envelope)')
        power_label = 'Mean Relative Power (Before Correction)'
        filename_suffix = "baseline_correction_preview"
        
        plot_path = self.plot_fft_spectrum(frequencies, mean_power, title, filename_suffix,
                                         power_label=power_label, show_gaussian=False, 
                                         show_baseline_info=True, baseline_power=baseline_power)
        print(f"Baseline correction preview saved to: {plot_path}")
    
    def plot_mean_spectral_power(self, frequencies=None, mean_power=None, std_power=None):
        """Plot mean spectral power with baseline correction and Gaussian fit."""
        if frequencies is None or mean_power is None:
            frequencies, mean_power, std_power = self.compute_mean_spectral_power()
            
        if frequencies is None:
            print("Cannot plot - no valid spectral data computed.")
            return
        
        # Use baseline-corrected data if available
        if hasattr(self, '_baseline_corrected_power'):
            plot_power = self._baseline_corrected_power
            power_label = 'Mean Relative Power (Baseline Corrected)'
            title_suffix = '(Baseline Corrected)'
        else:
            plot_power = mean_power
            power_label = 'Mean Relative Power'
            title_suffix = ''
        
        # Check if Gaussian fitting failed and add "failed" to title
        gaussian_failed = (not hasattr(self, 'fitted_params') or 
                          self.fitted_params is None or
                          hasattr(self, 'gaussian_fit_failure_reason'))
        
        if gaussian_failed:
            title_suffix += ' - Gaussian Fit Failed'
        
        title = (f'Mean Spectral Power - Subject {self.subject_id} {title_suffix}\n'
                f'Channel {self.target_channels[0]} ({self.low_freq}-{self.high_freq} Hz envelope)')
        filename_suffix = "mean_spectral_power"
        
        self.plot_fft_spectrum(frequencies, plot_power, title, filename_suffix,
                             power_label=power_label, show_gaussian=True)
        
        # Save the baseline-corrected data
        if frequencies is not None and plot_power is not None:
            if std_power is not None:
                std_values = std_power
            else:
                std_values = np.zeros_like(plot_power)

            spectral_data = pd.DataFrame({
                'frequency': frequencies, 
                'mean_power': plot_power, 
                'std_power': std_values
            })
            csv_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_spectral_power.csv"
            spectral_data.to_csv(csv_path, index=False)
            print(f"Spectral data (baseline corrected) saved to: {csv_path}")
        
    def get_averaged_metrics(self):
        """Return channel-level metrics from Gaussian fit to mean spectrum."""
        if hasattr(self, 'channel_result') and self.channel_result is not None:
            result = self.channel_result.copy()
            if self.results_df is not None:
                result['n_successful_bouts'] = len(self.results_df)
            else:
                result['n_successful_bouts'] = 0
            return result
        
        # If Gaussian fitting failed, return basic info
        if self.results_df is not None and not self.results_df.empty:
            return {
                'channel': self.target_channels[0],
                'n_successful_bouts': len(self.results_df),
                'gaussian_fit_failed': True
            }
        
        return None
        
    def get_summary(self):
        if self.results_df is None:
            print("No analysis results available. Run analyze_all_bouts() first.")
            return
        
        summary_lines = []
        summary_lines.append(f"Analysis Summary for Subject {self.subject_id}, Channel {self.target_channels[0]}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Frequency Range: {self.low_freq}-{self.high_freq} Hz")
        summary_lines.append(f"Total Spindles Detected: {len(self.spindles_df) if self.spindles_df is not None else 0}")
        summary_lines.append(f"Total N2 Bouts Found: {len(self.n2_bouts)}")
        summary_lines.append(f"Failed Bouts: {len(self.bout_errors)}")
        summary_lines.append(f"Low-Frequency Detrended Bouts: {len(self.detrended_bouts)}")
        if self.detrended_bouts:
            summary_lines.append(f"  Detrended Bout IDs: {', '.join(map(str, self.detrended_bouts))}")
        summary_lines.append("")
        summary_lines.append("Per-Bout Analysis (Spectral Power Only):")
        summary_lines.append(f"{'Bout':<6}{'Start Time':<12}{'End Time':<12}{'Duration':<10}")
        summary_lines.append("-" * 50)
        
        for _, row in self.results_df.iterrows():
            summary_lines.append(f"{row['bout_id']:<6}{row['start_time']:<12.1f}{row['end_time']:<12.1f}{row['duration']:<10.1f}")
        
        summary_lines.append("")
        summary_lines.append("Bout Duration Statistics:")
        summary_lines.append("-" * 60)
        summary_lines.append(f"Average bout duration: {self.results_df['duration'].mean():.1f}s (±{self.results_df['duration'].std():.1f})")
        summary_lines.append(f"Total analyzed time: {self.results_df['duration'].sum():.1f}s")
        
        # Add channel-level results if available
        if hasattr(self, 'channel_result') and self.channel_result is not None:
            summary_lines.append("")
            summary_lines.append("Channel-Level Gaussian Fit Results:")
            summary_lines.append("-" * 60)
            cr = self.channel_result
            summary_lines.append(f"Peak frequency: {cr['peak_frequency']:.3f} Hz")
            summary_lines.append(f"Bandwidth (2σ): {cr['bandwidth']:.3f} Hz")
            summary_lines.append(f"Peak amplitude: {cr['peak_amplitude']:.3f}")
            summary_lines.append(f"AUC proxy: {cr['auc']:.3f}")
        else:
            # Handle Gaussian fitting failure
            summary_lines.append("")
            summary_lines.append("Channel-Level Gaussian Fit Results:")
            summary_lines.append("-" * 60)
            summary_lines.append("STATUS: FAILED")
            if hasattr(self, 'gaussian_fit_failure_reason'):
                summary_lines.append(f"FAILURE REASON: {self.gaussian_fit_failure_reason}")
            else:
                summary_lines.append("FAILURE REASON: Unknown - no fitting attempted or results unavailable")
        
        if self.bout_errors:
            summary_lines.append("")
            summary_lines.append("Analysis Errors:")
            summary_lines.append("-" * 60)
            for bout_info, error in self.bout_errors.items():
                error_type = type(error).__name__
                error_msg = str(error)
                summary_lines.append(f"Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s): {error_type} - {error_msg}")
        
        summary_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channels[0]}_analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

def analyze_all_channels(sub, raw_path, apply_detrending=True, include_n3=False, subject_dir=None):
    raw = mne.io.read_raw(raw_path, preload=False)
    excluded_channels = set(FACE_ELECTRODES + NECK_ELECTRODES)
    valid_channels = [ch for ch in raw.ch_names if ch not in excluded_channels and 'EMG' not in ch]
    
    channel_results = []
    all_bout_results = []
    for channel in valid_channels:
        print(f"\n{'='*60}")
        print(f"Analyzing subject: {sub}, channel: {channel}")
        print(f"{'='*60}")
        
        output_dir = f"{subject_dir}/{sub}_{channel}_output" if subject_dir else None
        analyzer = run_channel_analysis(sub, raw_path, channel, output_dir, apply_detrending, include_n3)
        aggregate_channel_results(analyzer, channel, channel_results, all_bout_results)

    save_multi_channel_summary(sub, channel_results, all_bout_results)


def get_all_subjects(main_dir):
    """Get list of all subject IDs from directory."""
    if not os.path.exists(main_dir):
        print(f"Main directory not found: {main_dir}")
        return []
    
    subject_dirs = [d for d in os.listdir(main_dir) 
                   if os.path.isdir(os.path.join(main_dir, d))]
    
    print(f"Found {len(subject_dirs)} subjects: {subject_dirs}")
    return subject_dirs


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


def run_channel_analysis(subject_id, raw_path, channel, output_dir=None, apply_detrending=True, include_n3=False):
    """Run complete analysis pipeline for a single channel."""
    analyzer = SpindleAnalyzer(subject_id, raw_path, low_freq=13, high_freq=16, 
                             target_channels=[channel], output_dir=output_dir, 
                             apply_detrending=apply_detrending, include_n3=include_n3)
    analyzer.analyze_all_bouts()
    analyzer.get_summary()
    analyzer.plot_baseline_correction_preview()
    analyzer.plot_mean_spectral_power()
    return analyzer


def aggregate_channel_results(analyzer, channel_name, channel_results, all_bout_results):
    """Aggregate a channel's analysis results for multi-channel processing."""
    avg_metrics = analyzer.get_averaged_metrics()
    
    if avg_metrics is not None:
        channel_results.append(avg_metrics)
    
    if analyzer.results_df is not None and not analyzer.results_df.empty:
        bout_df = analyzer.results_df.copy()
        bout_df['channel'] = channel_name
        cols = ['channel'] + bout_df.columns.tolist()
        bout_df = bout_df[cols]
        all_bout_results.append(bout_df)


def save_multi_channel_summary(subject_id, channel_results, all_bout_results):
    """Save aggregated results from multi-channel analysis."""
    if channel_results:
        results_df = pd.DataFrame(channel_results)
        summary_file = f"{subject_id}/{subject_id}_all_channels_summary.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"\n{'='*60}")
        print(f"SUMMARY: Saved results for {len(channel_results)} channels to {summary_file}")
        print(f"{'='*60}")
    
    if all_bout_results:
        all_bouts_df = pd.concat(all_bout_results, ignore_index=True)
        all_bouts_file = f"{subject_id}/{subject_id}_all_bouts_details.csv"
        all_bouts_df.to_csv(all_bouts_file, index=False)
        print(f"DETAILED: Saved {len(all_bouts_df)} individual bout results to {all_bouts_file}")


def save_focused_analysis_results(analyzer, subject_id, channel_name, output_dir):
    """Save analysis results specifically for focused analysis."""
    avg_metrics = analyzer.get_averaged_metrics()
    
    if avg_metrics is not None:
        results_df = pd.DataFrame([avg_metrics])
        summary_file = f"{output_dir}/{subject_id}_{channel_name}_summary.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"✓ Saved summary results to {summary_file}")
    
    if analyzer.results_df is not None and not analyzer.results_df.empty:
        bout_df = analyzer.results_df.copy()
        bout_df['channel'] = channel_name
        bout_file = f"{output_dir}/{subject_id}_{channel_name}_bouts_details.csv"
        bout_df.to_csv(bout_file, index=False)
        print(f"✓ Saved {len(bout_df)} individual bout results to {bout_file}")


def process_all_subjects(apply_detrending=True, include_n3=False):
    """Process all subjects in the control_clean directory."""
    start_time = time.time()
    print(f"{'='*80}")
    print(f"Detrending: {'ENABLED' if apply_detrending else 'DISABLED'}")
    print(f"{'='*80}\n")
    
    # subject_dirs = get_all_subjects(f"{BASE_DIR}/control_clean/")
    # if not subject_dirs:
    #     return
    subject_dirs = ["RD43"]

    subject_dirs = ["RD43"]
    processed_subjects = []
    failed_subjects = []
    for i, sub in enumerate(subject_dirs):
        subject_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT {i}/{len(subject_dirs)}: {sub}")
        print(f"{'='*80}")
        
        try:
            raw_path = find_subject_fif_file(sub)
            if raw_path:
                subject_dir = f"{sub}_" # FILL IN!! 
                analyze_all_channels(sub, raw_path, apply_detrending, include_n3, subject_dir)
                subject_duration = time.time() - subject_start_time
                processed_subjects.append(sub)
                print(f"✓ Successfully processed subject {sub} in {subject_duration/60:.2f} minutes")
                
            else:
                subject_duration = time.time() - subject_start_time
                print(f"✗ No base .fif file found for subject {sub} (checked in {subject_duration:.1f} seconds)")
                failed_subjects.append((sub, "No .fif file found"))
                
        except Exception as e:
            subject_duration = time.time() - subject_start_time
            print(f"✗ Error processing subject {sub} after {subject_duration/60:.2f} minutes: {e}")
            failed_subjects.append((sub, str(e)))
    
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


def focused_electrode_analysis(target_subject, target_electrode, output_dir, apply_detrending=True, include_n3=False):
    """Perform focused analysis on single electrode of single subject."""
    start_time = time.time()
    print(f"\nFOCUSED ANALYSIS: Subject {target_subject}, Electrode {target_electrode}")
    print(f"Detrending: {'ENABLED' if apply_detrending else 'DISABLED'}")
    try:
        raw_path = find_subject_fif_file(target_subject)
        if raw_path:
            print(f"{'='*60}")
            analyzer = run_channel_analysis(target_subject, raw_path, target_electrode, output_dir, apply_detrending, include_n3)
            save_focused_analysis_results(analyzer, target_subject, target_electrode, output_dir)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n{'='*60}")
            print(f"✓ FOCUSED ANALYSIS COMPLETE in {duration:.2f} seconds")
            print(f"✓ Results saved in {output_dir}/ directory")
            print(f"{'='*60}")
            
        else:
            print(f"✗ No base .fif file found for subject {target_subject}")
            
    except Exception as e:
        print(f"✗ Error in focused analysis: {e}")
        raise


def main():
    """Main function to run spectral analysis."""
    
    process_all_subjects(apply_detrending=True, include_n3=True)  # OPTION 1: Process all subjects
    
    # focused_electrode_analysis(target_subject="RD43", target_electrode="E221", 
    #                           output_dir="tmp/no_detrending", apply_detrending=False)


if __name__ == "__main__":
    main()