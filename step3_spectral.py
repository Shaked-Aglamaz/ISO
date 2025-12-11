import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
from scipy.signal import hilbert, detrend
from scipy.signal.windows import hann
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from pathlib import Path
import time
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import BASE_DIR, FACE_ELECTRODES, NECK_ELECTRODES
from utils import find_subject_fif_file, get_all_subjects


mne.set_log_level("error")
mne.viz.set_browser_backend('qt')
mne.set_config('MNE_BROWSER_THEME', 'dark')


class BoutInfo:
    """N2 sleep bout information."""
    def __init__(self, start_time, end_time, bout_id):
        self.start_time = start_time
        self.end_time = end_time
        self.id = bout_id
        self.duration = end_time - start_time
        self.n_segments = 0

    def __str__(self):
        return f"Bout {self.id}: {self.start_time:.1f}-{self.end_time:.1f}s ({self.duration:.1f}s), Segments: {self.n_segments}"


class SpindleAnalyzer:
    def __init__(self, subject_id, raw_path, low_freq=13, high_freq=16, target_channel='VREF', 
                output_dir=None, min_duration=300, include_n3=False, annotations_path=None):
        self.subject_id = subject_id
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.target_channel = target_channel
        self.output_dir = output_dir if output_dir else f"{subject_id}/{subject_id}_{target_channel}_output"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._add_to_gitignore(Path(self.output_dir).parent)
        self.raw_path = raw_path
        self.min_duration = min_duration
        self.include_n3 = include_n3
        self.annotations_path = annotations_path
        
        self.target_raw = None
        self.spindles_df = None
        self.n2_bouts = []
        self.bout_errors = {}
        self.results_df = None
        self.detrended_bouts = []
        
        # For relative spectral power computation
        self._common_freqs = None
        self._all_power_spectra = []
        self.channel_result = None
        self.fitted_params = None

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
        if self.annotations_path.exists():
            cleaned_annotations = mne.read_annotations(self.annotations_path)
            raw_data.set_annotations(cleaned_annotations)
            print(f"✓ Loaded cleaned annotations from {self.annotations_path.name}")
        else:
            print(f"⚠ No cleaned annotations found at {self.annotations_path}, using original")
        
        annotation_desc = set(raw_data.annotations.description)
        has_n2 = any(stage in annotation_desc for stage in ["N2", "NREM2"])
        has_n3 = any(stage in annotation_desc for stage in ["N3", "NREM3"])
        if self.include_n3:
            if not has_n2 and not has_n3:
                print(f"ERROR: No 'N2'/'NREM2' or 'N3'/'NREM3' annotations found in the data.")
                print(f"Available annotations: {annotation_desc}")
                exit(1)
        else:
            if not has_n2:
                print(f"ERROR: No 'N2' or 'NREM2' annotations found in the data.")
                print(f"Available annotations: {annotation_desc}")
                exit(1)
        
        # Check for annotations containing "bad" (case-insensitive)
        bad_annotations = [desc for desc in annotation_desc if 'bad' in desc.lower()]
        if not bad_annotations:
            print("WARNING: No annotations containing 'bad' found in the data.")
            print(f"Available annotations: {annotation_desc}")
        
        self.target_raw = raw_data.pick_channels([self.target_channel])
        self.target_raw.load_data()

    def detect_spindles(self):
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
            stage_mask = ((ann.description == "N2") | (ann.description == "NREM2") | 
                         (ann.description == "N3") | (ann.description == "NREM3")) & (ann.duration >= self.min_duration)
            stage_names = "N2/N3"
        else:
            stage_mask = ((ann.description == "N2") | (ann.description == "NREM2")) & (ann.duration >= self.min_duration)
            stage_names = "N2"
            
        bad_mask = np.array([('bad' in desc.lower()) for desc in ann.description])
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
        """
        Compute raw amplitude envelope using Hilbert transform.
        Returns the amplitude envelope without any mean subtraction or recentering.
        """
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    @staticmethod
    def moving_average_zero_phase(x, win_samp):
        """
        Subtract a centered moving-average trend using uniform_filter1d.
        Uses scipy.ndimage.uniform_filter1d for O(N) performance with excellent edge handling.
        The 'reflect' mode mirrors the signal at edges to avoid artifacts.
        """
        # Compute centered moving average - O(N) using cumulative sum internally
        trend = uniform_filter1d(x, size=win_samp, mode='reflect')
        return x - trend
    
    @staticmethod
    def gaussian_smoothing_zero_phase(x, sigma_sec, sfreq):
        """
        Subtract a zero-phase Gaussian-smoothed trend to remove ultra-slow drift.
        This method uses Gaussian smoothing which provides superior frequency
        response characteristics compared to boxcar moving average. The Gaussian kernel
        has a smoother frequency response that better preserves the 0.004-0.01 Hz range
        of interest while effectively removing ultra-slow drift below ~0.003 Hz.
        """
        
        # Convert sigma from seconds to samples
        sigma_samp = sigma_sec * sfreq
        
        # Apply Gaussian smoothing - 'reflect' mode mirrors signal at edges
        trend = gaussian_filter1d(x, sigma=sigma_samp, mode='reflect')
        
        return x - trend
    
    @staticmethod
    def apply_hann_taper(signal):
        """
        Apply Hann window to signal to reduce spectral leakage in FFT.
        The Hann window smoothly tapers the signal to zero at both ends.
        """
        window = hann(len(signal))
        return signal * window

    @staticmethod
    def split_bout_into_segments(detrended_envelope, sfreq, bout_duration):
        """
        Split a bout's detrended envelope into segments based on bout duration.
        - 300 ≤ L < 600s: Return whole bout as single segment
        - 600 ≤ L < 900s: Return centered 600s segment
        - L ≥ 900s: Return multiple centered 600s segments with 50% overlap (300s step)
        """
        L = bout_duration
        
        if L < 600:
            return [detrended_envelope]
        
        if L < 900:
            start_sec = (L - 600) / 2
            start_idx = int(start_sec * sfreq)
            end_idx = start_idx + int(600 * sfreq)
            centered_segment = detrended_envelope[start_idx:end_idx]
            return [centered_segment]
        
        window_sec = 600
        step_sec = 300
        window_samples = int(window_sec * sfreq)
        step_samples = int(step_sec * sfreq)
        total_samples = len(detrended_envelope)
        
        # Calculate number of segments, total span and offset for centering
        n_segments = 1 + (total_samples - window_samples) // step_samples
        total_span = window_samples + (n_segments - 1) * step_samples
        offset = (total_samples - total_span) // 2
        segments = []
        for i in range(n_segments):
            start_idx = offset + i * step_samples
            segment = detrended_envelope[start_idx:start_idx + window_samples]
            segments.append(segment)
        
        return segments

    @staticmethod
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def _compute_basic_fft(self, amplitude_envelope, target_duration=600):
        """
        Compute basic FFT power spectrum from amplitude envelope with fixed frequency resolution.
        """
        sfreq = self.target_raw.info['sfreq']
        n_samples = amplitude_envelope.shape[0]
        target_samples = int(target_duration * sfreq)
        freqs = np.fft.rfftfreq(target_samples, d=1/sfreq)
        fft_values = np.fft.rfft(amplitude_envelope, n=target_samples)
        fft_power = np.abs(fft_values) ** 2
        fft_power *= 1e6
        fft_power = fft_power / (n_samples / sfreq)
        return freqs, fft_power
    
    def detect_low_frequency_artifacts(self, amplitude_envelope):
        """Detect low-frequency artifacts (for QC only, not used in current pipeline)."""
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
    
    def compute_fft_power_spectrum(self, amplitude_envelope, min_freq=0, max_freq=0.1):
        """
        Compute FFT power spectrum and extract frequency range.
        """
        freqs, fft_power = self._compute_basic_fft(amplitude_envelope)
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs_masked = freqs[mask]
        if self._common_freqs is None:
            self._common_freqs = freqs_masked
        elif not np.array_equal(self._common_freqs, freqs_masked):
            # Verify that current frequency grid matches the stored one
            raise ValueError(f"Frequency grid mismatch in channel {self.target_channel}")
        
        return fft_power[mask]

    def fit_gaussian_to_spectrum(self, x, y):
        """Fit Gaussian curve to spectrum for ISFS detection."""
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
        """
        Analyze single bout using 9-step pipeline with bout splitting:

        Time Domain (per bout):
        1. Crop and filter to sigma band (13-16 Hz)
        2. Compute Hilbert envelope (raw amplitude)
        3. Gaussian smoothing detrending (σ = 50s)
        4. Recenter around zero
        
        Bout Splitting:
        5. Split into segments based on duration:
        
        Frequency Domain (per segment):
        6. Apply Hann taper
        7. Compute FFT power spectrum (0-0.1 Hz)
        8. Average segments (if L ≥ 900s) → 1 spectrum per bout
        
        Normalization (per bout):
        9. Divide by mean of bout spectrum (0-0.1 Hz)
        """
        # Step 1: Crop and filter to sigma band
        bout_raw = self.target_raw.copy().crop(tmin=bout_info.start_time, tmax=bout_info.end_time)
        bout_sigma = bout_raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)
        bout_data = bout_sigma.get_data()[0]
        
        # Step 2: Compute Hilbert envelope (raw amplitude)
        amplitude_envelope = self.compute_envelope(bout_data)

        # Step 3: Apply Gaussian smoothing detrending (σ = 50s) to remove ultra-slow drift
        sfreq = bout_raw.info['sfreq']
        sigma_sec = 50  # Standard deviation of Gaussian kernel in seconds
        amplitude_envelope = self.gaussian_smoothing_zero_phase(amplitude_envelope, sigma_sec, sfreq)
        
        # Step 4: Recenter around zero
        amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
        
        # Step 5: Split into segments based on bout duration
        envelope_segments = self.split_bout_into_segments(amplitude_envelope, sfreq, bout_info.duration)
        bout_info.n_segments = len(envelope_segments)
        
        # Steps 6-8: Process each segment
        segment_spectra = []
        for segment in envelope_segments:
            # Step 6: Apply Hann taper to segment
            tapered_segment = self.apply_hann_taper(segment)
            
            # Step 7: Compute FFT power spectrum (0-0.1 Hz)
            power = self.compute_fft_power_spectrum(tapered_segment, min_freq=0, max_freq=0.1)
            segment_spectra.append(power)
        
        # Step 8: Average segment spectra if multiple segments (L ≥ 900s)
        bout_spectrum = np.mean(segment_spectra, axis=0) if len(envelope_segments) > 1 else segment_spectra[0]

        # Step 9: Normalize by mean of bout spectrum (0-0.1 Hz)
        bout_mean_power = np.mean(bout_spectrum)
        normalized_spectrum = bout_spectrum / bout_mean_power
        
        bout_result = {
            'bout_id': bout_info.id,
            'start_time': bout_info.start_time,
            'end_time': bout_info.end_time,
            'duration': bout_info.duration,
            'n_segments': bout_info.n_segments
        }
        
        self._save_bout_fft_data(self._common_freqs, normalized_spectrum, bout_info)
        self.plot_bout_fft(self._common_freqs, normalized_spectrum, bout_info)
        return bout_result, normalized_spectrum

    def _save_bout_fft_data(self, frequencies, power, bout_info):
        """Save bout FFT data to CSV (relative power)."""
        fft_data = pd.DataFrame({
            'frequency': frequencies,
            'relative_power': power
        })
        
        csv_filename = f"{self.subject_id}_{self.target_channel}_bout_{bout_info.id:02d}_fft.csv"
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
                f'Time: {bout_info.start_time:.1f} - {bout_info.end_time:.1f}s, Duration: {bout_info.duration:.1f}s, # Segments: {bout_info.n_segments}')
        filename_suffix = f"bout_{bout_info.id}_fft_power"
        self.plot_fft_spectrum(x, y, title, filename_suffix, show_gaussian=False)

    def plot_fft_spectrum(self, frequencies, power_data, title, filename_suffix, 
                         power_label=None, show_gaussian=False, show_baseline_info=False, 
                         baseline_power=None, show_popup=False, show_threshold=False):
        """Unified FFT spectrum plotting function for all scenarios."""
        if power_label is None:
            power_label = f'Relative Power ({self.target_channel})'
        
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
        
        # 4. ISFS threshold line (if available and requested)
        if show_threshold and hasattr(self, 'isfs_threshold'):
            plt.axhline(y=self.isfs_threshold, color='gray', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label=f'ISFS Threshold: {self.isfs_threshold:.3f} AU')
        
        # 5. Standard plot formatting
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative Power (AU)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 6. Save and show
        plot_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channel}_{filename_suffix}.png"
        plt.savefig(plot_path, dpi=300)
        
        if show_popup:
            plt.show()
        else:
            plt.close()
        
        return plot_path

    def _validate_spindles_and_bouts(self):
        """Helper function to validate spindle detection and N2 bouts before analysis."""
        if self.spindles_df is None or self.target_raw is None:
            self.detect_spindles()
        
            # Check if spindle detection failed
            if self.spindles_df is None:
                message = f"{self.subject_id} - No spindles found in channel {self.target_channel}"
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

    def analyze_all_bouts(self):
        """Analyze all valid N2 bouts in the recording."""
        if not self._validate_spindles_and_bouts():
            return
        
        bout_results = []
        for bout_info in self.n2_bouts:
            try:
                result, power = self.analyze_single_bout(bout_info)
                if result is not None:
                    bout_results.append(result)
                    self._all_power_spectra.append(power)
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
                self.fit_gaussian_to_mean_spectrum(frequencies, baseline_corrected)
                
                # Store baseline-corrected data for plotting
                self._baseline_corrected_power = baseline_corrected
                
                print("✓ Per-channel analysis completed")
            else:
                print("✗ Could not compute mean spectral power")
        
    def compute_mean_spectral_power(self):
        """Compute mean spectral power over all bouts."""
        if self._common_freqs is None or not self._all_power_spectra:
            print("No power spectra available. Run analyze_all_bouts() first.")
            return None, None, None
        
        # Average directly across bouts (no interpolation needed)
        power_matrix = np.array(self._all_power_spectra)
        mean_power_spectrum = np.mean(power_matrix, axis=0)
        std_power_spectrum = np.std(power_matrix, axis=0)
        
        return self._common_freqs, mean_power_spectrum, std_power_spectrum
    
    def apply_baseline_correction(self, frequencies, mean_power):
        """Apply baseline correction using 0.06-0.1 Hz range."""
        baseline_mask = (frequencies >= 0.06) & (frequencies <= 0.1)
        
        if not np.any(baseline_mask):
            print("Warning: No data in 0.06-0.1 Hz range for baseline correction")
            return mean_power
        
        baseline_power = np.mean(mean_power[baseline_mask])
        baseline_corrected = mean_power - baseline_power
        return baseline_corrected
    
    def fit_gaussian_to_mean_spectrum(self, frequencies, power):
        """Fit Gaussian to baseline-corrected mean spectrum (per-channel analysis)."""
        try:
            fitted_params = self.fit_gaussian_to_spectrum(frequencies, power)
            if fitted_params is None:
                failure_reason = "Invalid parameters (negative peak frequency)"
                print(f"Gaussian fitting failed - {failure_reason}")
                self.gaussian_fit_failure_reason = failure_reason
                return
            
            peak_amplitude, peak_frequency, bandwidth_sigma = fitted_params
            
            # Calculate actual AUC of the fitted Gaussian (±1σ area)
            x_fit = np.linspace(frequencies[0], frequencies[-1], 500)
            y_fit = self.gaussian(x_fit, *fitted_params)
            x1 = peak_frequency - bandwidth_sigma
            x2 = peak_frequency + bandwidth_sigma
            mask = (x_fit >= x1) & (x_fit <= x2)
            actual_auc = simpson(x=x_fit[mask], y=y_fit[mask])

            # ISFS detection condition
            gaussian_std = np.std(y_fit)
            isfs_threshold = 1.5 * gaussian_std
            isfs_detected = peak_amplitude > isfs_threshold

            # Store threshold for plotting (whether ISFS detected or not)
            self.isfs_threshold = isfs_threshold

            if not isfs_detected:
                failure_reason = "ISFS not detected (peak amplitude below threshold)"
                self.gaussian_fit_failure_reason = failure_reason
                return

            # Store channel-level results
            channel_result = {
                'channel': self.target_channel,
                'peak_frequency': peak_frequency,
                'bandwidth': 2 * bandwidth_sigma,  # Convert sigma to full bandwidth
                'auc': actual_auc,  # Actual AUC of ±1σ area under Gaussian curve
                'peak_amplitude': peak_amplitude,
                'bandwidth_sigma': bandwidth_sigma
            }
            
            self.channel_result, self.fitted_params = channel_result, fitted_params
            
        except Exception as e:
            failure_reason = f"Exception during fitting: {str(e)}"
            print(f"Failed to fit Gaussian to mean spectrum: {e}")
            self.gaussian_fit_failure_reason = failure_reason
            return

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
                f'Channel {self.target_channel} ({self.low_freq}-{self.high_freq} Hz envelope)')
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
                f'Channel {self.target_channel} ({self.low_freq}-{self.high_freq} Hz envelope)')
        filename_suffix = "mean_spectral_power"
        
        self.plot_fft_spectrum(frequencies, plot_power, title, filename_suffix,
                             power_label=power_label, show_gaussian=True, show_threshold=True)
        
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
            csv_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channel}_spectral_power.csv"
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
                'channel': self.target_channel,
                'n_successful_bouts': len(self.results_df),
                'gaussian_fit_failed': True
            }
        
        return None
        
    def get_summary(self):
        if self.results_df is None:
            print("No analysis results available. Run analyze_all_bouts() first.")
            return
        
        summary_lines = []
        summary_lines.append(f"Analysis Summary for Subject {self.subject_id}, Channel {self.target_channel}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Frequency Range: {self.low_freq}-{self.high_freq} Hz")
        summary_lines.append(f"Total Spindles Detected: {len(self.spindles_df) if self.spindles_df is not None else 0}")
        bouts = "N2/N3" if self.include_n3 else "N2"
        summary_lines.append(f"Total {bouts} Bouts Found: {len(self.n2_bouts)}")
        summary_lines.append(f"Failed Bouts: {len(self.bout_errors)}")
        summary_lines.append(f"Low-Frequency Detrended Bouts: {len(self.detrended_bouts)}")
        if self.detrended_bouts:
            summary_lines.append(f"  Detrended Bout IDs: {', '.join(map(str, self.detrended_bouts))}")
        summary_lines.append("")
        summary_lines.append("Per-Bout Analysis (Spectral Power Only):")
        summary_lines.append(f"{'Bout':<6}{'Start Time':<12}{'End Time':<12}{'Duration':<10}{'Segments':<10}")
        summary_lines.append("-" * 60)
        
        for _, row in self.results_df.iterrows():
            summary_lines.append(f"{int(row['bout_id']):<6}{row['start_time']:<12.1f}{row['end_time']:<12.1f}{row['duration']:<10.1f}{int(row['n_segments']):<10}")
        
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
                summary_lines.append(f"REASON: {self.gaussian_fit_failure_reason}")
            else:
                summary_lines.append("REASON: Unknown - no fitting attempted or results unavailable")
            summary_lines.append("")
            summary_lines.append("NOTE: Channel excluded from group-level analysis and computations.")
        
        if self.bout_errors:
            summary_lines.append("")
            summary_lines.append("Analysis Errors:")
            summary_lines.append("-" * 60)
            for bout_info, error in self.bout_errors.items():
                error_type = type(error).__name__
                error_msg = str(error)
                summary_lines.append(f"Bout {bout_info.id} ({bout_info.start_time:.1f}-{bout_info.end_time:.1f}s): {error_type} - {error_msg}")
        
        summary_path = Path(self.output_dir) / f"{self.subject_id}_{self.target_channel}_analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

def analyze_all_channels(sub, raw_path, include_n3=False, subject_dir=None, max_workers=None, annotations_path=None):
    raw = mne.io.read_raw(raw_path, preload=False)
    excluded_channels = set(FACE_ELECTRODES + NECK_ELECTRODES)
    valid_channels = [
        ch for ch in raw.ch_names 
        if ch not in excluded_channels 
        and raw.info['chs'][raw.ch_names.index(ch)]['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH
    ]    
    raw.close()
    del raw
    
    total_channels = len(valid_channels)
    channel_results, all_bout_results = [], []
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() // 2) - 1)
    print(f"Parallel Workers: {max_workers} / {os.cpu_count()} CPU cores ({total_channels} channels)")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_channel = {}
        for channel in valid_channels:
            output_dir = f"{subject_dir}/{sub}_{channel}_output" if subject_dir else None
            future = executor.submit(run_channel_analysis, sub, raw_path, channel, output_dir, include_n3, annotations_path)
            future_to_channel[future] = channel
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_channel):
            channel = future_to_channel[future]
            completed += 1
            try:
                analyzer = future.result()
                aggregate_channel_results(analyzer, channel, channel_results, all_bout_results)
                print(f"✓ [{completed}/{total_channels}] {sub} - Channel {channel} completed")
                del analyzer
            except Exception as e:
                print(f"✗ [{completed}/{total_channels}] {sub} - Channel {channel} failed: {e}")
    
    gc.collect()
    save_multi_channel_summary(sub, channel_results, all_bout_results, subject_dir, total_channels)


def run_channel_analysis(subject_id, raw_path, channel, output_dir=None, include_n3=False, annotations_path=None):
    """Run complete analysis pipeline for a single channel."""
    # Format annotations path with subject_id if template provided
    if isinstance(annotations_path, str) and '{subject_id}' in annotations_path:
        annotations_path = Path(annotations_path.format(subject_id=subject_id))
    else:
        annotations_path = Path(annotations_path)
    analyzer = SpindleAnalyzer(subject_id, raw_path, low_freq=13, high_freq=16, target_channel=channel,
                               output_dir=output_dir, include_n3=include_n3, annotations_path=annotations_path)
    analyzer.analyze_all_bouts()
    analyzer.get_summary()
    analyzer.plot_baseline_correction_preview()
    analyzer.plot_mean_spectral_power()
    return analyzer


def aggregate_channel_results(analyzer, channel_name, channel_results, all_bout_results):
    """Aggregate a channel's analysis results for multi-channel processing."""
    avg_metrics = analyzer.get_averaged_metrics()
    
    # Only include channels that meet ISFS detection criteria
    if avg_metrics is not None and not avg_metrics.get('gaussian_fit_failed', False):
        channel_results.append(avg_metrics)
    else:
        print(f"  ✗ Channel {channel_name} excluded from group analysis (no ISFS or fit failed)")
    
    # Still save bout-level results for all channels for reference
    if analyzer.results_df is not None and not analyzer.results_df.empty:
        bout_df = analyzer.results_df.copy()
        bout_df['channel'] = channel_name
        cols = ['channel'] + bout_df.columns.tolist()
        bout_df = bout_df[cols]
        all_bout_results.append(bout_df)


def save_multi_channel_summary(subject_id, channel_results, all_bout_results, subject_dir=None, total_channels=None):
    """Save aggregated results from multi-channel analysis."""
    # Use subject_dir if provided, otherwise fall back to subject_id folder at root
    output_base = subject_dir if subject_dir else subject_id
    
    if channel_results:
        results_df = pd.DataFrame(channel_results)
        summary_file = f"{output_base}/{subject_id}_all_channels_summary.csv"
        
        # Calculate ISFS detection statistics
        channels_with_isfs = len(channel_results)
        channels_excluded = total_channels - channels_with_isfs if total_channels else 0
        percentage_with_isfs = (channels_with_isfs / total_channels * 100) if total_channels else 0
        
        # Write header comments with summary statistics
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Subject: {subject_id}\n")
            f.write(f"# ISFS Detection Summary:\n")
            if total_channels:
                f.write(f"#   Total valid channels analyzed: {total_channels}\n")
                f.write(f"#   Channels with ISFS detected: {channels_with_isfs} ({percentage_with_isfs:.1f}%)\n")
                f.write(f"#   Channels excluded (no ISFS): {channels_excluded} ({100-percentage_with_isfs:.1f}%)\n")
            else:
                f.write(f"#   Channels with ISFS detected: {channels_with_isfs}\n")
            f.write("#\n")
        
        # Append the dataframe
        results_df.to_csv(summary_file, mode='a', index=False)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: Saved results for {channels_with_isfs} channels WITH ISFS to {summary_file}")
        if total_channels:
            print(f"         Total channels analyzed: {total_channels}")
            print(f"         ISFS detected: {channels_with_isfs}/{total_channels} ({percentage_with_isfs:.1f}%)")
            print(f"         Excluded: {channels_excluded}/{total_channels} ({100-percentage_with_isfs:.1f}%)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: No channels with detected ISFS for subject {subject_id}")
        if total_channels:
            print(f"         Total channels analyzed: {total_channels}")
            print(f"         All {total_channels} channels excluded (no ISFS detected)")
        print(f"         No summary file created.")
        print(f"{'='*60}")
    
    if all_bout_results:
        all_bouts_df = pd.concat(all_bout_results, ignore_index=True)
        all_bouts_file = f"{output_base}/{subject_id}_all_bouts_details.csv"
        all_bouts_df.to_csv(all_bouts_file, index=False)
        print(f"DETAILED: Saved {len(all_bouts_df)} individual bout results to {all_bouts_file}")
        print(f"          (Includes all channels for reference)")


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


def process_all_subjects(main_dir, include_n3=False, max_workers=None, annotations_path=None):
    """Process all subjects in the control_clean directory with configurable parallelization."""
    start_time = time.time()
    subject_dirs = get_all_subjects(main_dir)
    if not subject_dirs:
        return

    # subject_dirs = ["EL3016"]
    processed_subjects, failed_subjects = [], []
    for i, sub in enumerate(subject_dirs):
        subject_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT {i}/{len(subject_dirs)}: {sub}")
        print(f"{'='*80}")
        
        try:
            sub_dir = f"{main_dir}/{sub}/saved_raw/CleaningPipe/"
            raw_path = find_subject_fif_file(sub_dir)
            if raw_path:
                stages = "N2N3" if include_n3 else "N2"
                subject_dir = f"{sub}_{stages}_split_bouts"
                analyze_all_channels(sub, raw_path, include_n3, subject_dir, max_workers, annotations_path)
                subject_duration = time.time() - subject_start_time
                processed_subjects.append(sub)
                print(f"✓ Successfully processed subject {sub} in {subject_duration/60:.2f} minutes")
                
                # Force garbage collection after each subject
                gc.collect()
                
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


def focused_electrode_analysis(subject, subject_dir, electrode, output_dir, include_n3=False, annotations_path=None):
    """Perform focused analysis on single electrode of single subject."""
    start_time = time.time()
    print(f"\nFOCUSED ANALYSIS: Subject {subject}, Electrode {electrode}")
    try:
        raw_path = find_subject_fif_file(subject_dir)
        if raw_path:
            print(f"{'='*60}")
            analyzer = run_channel_analysis(subject, raw_path, electrode, output_dir, include_n3, annotations_path)
            save_focused_analysis_results(analyzer, subject, electrode, output_dir)

            end_time = time.time()
            duration = end_time - start_time
            print(f"\n{'='*60}")
            print(f"✓ FOCUSED ANALYSIS COMPLETE in {duration:.2f} seconds")
            print(f"✓ Results saved in {output_dir}/ directory")
            print(f"{'='*60}")
            
        else:
            print(f"✗ No base .fif file found for subject {subject}")

    except Exception as e:
        print(f"✗ Error in focused analysis: {e}")
        raise


def main():
    """Main function to run spectral analysis."""
    
    # OPTION 1: Process all subjects with parallel processing (recommended)
    main_dir = f"{BASE_DIR}/elderly_control/"
    
    # Option A: Template with {subject_id} placeholder (recommended for multiple subjects):
    annotations_path = f"{BASE_DIR}/elderly_control/{{subject_id}}/{{subject_id}}_cleaned_hypno_annotations.txt"
    
    # Option B: Specific path (same file for all subjects):
    # annotations_path = "path/to/specific/annotations.txt"
    
    # process_all_subjects(main_dir, include_n3=True, max_workers=3, annotations_path=annotations_path)

    # OPTION 2: Focused single electrode analysis (always sequential)
    subject = "RD43"
    subject_dir = f"{BASE_DIR}/control_clean/{subject}/"
    focused_electrode_analysis(subject, subject_dir, electrode="E24", output_dir="tmp/sanity_check", 
                               include_n3=True, annotations_path=annotations_path)


if __name__ == "__main__":
    main()