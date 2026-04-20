import os
import gc
import glob
import mne
import numpy as np
import psutil
import traceback
import time
from dateutil import parser

from utils.config import FACE_ELECTRODES, NECK_ELECTRODES, BASE_DIR, EAR_ELECTRODES
from utils.utils import load_google_sheet, find_subject_fif_file

mne.set_log_level('WARNING')


def load_subject_cleaning_mapping():
    """Load mapping between subject ID and cleaning ID from Google Sheet"""
    df = load_google_sheet()
    mapping = dict(zip(df['ID'], df['cleaning id'].fillna(0).astype(int).astype(str)))
    return mapping


def apply_filtering(raw):
    """Apply filtering steps to raw data with dynamic channel-wise processing"""
    excluded_channels = set(FACE_ELECTRODES + NECK_ELECTRODES)
    channel_types = raw.get_channel_types()
    channels = [ch for ch, ch_type in zip(raw.ch_names, channel_types) 
                if ch not in excluded_channels and ch_type == 'eeg']
    raw.pick(channels)
    n_channels = len(raw.ch_names)
    
    print(f"Processing {n_channels} channels, duration: {raw.times[-1]/60:.1f} minutes")
    
    available_gb = psutil.virtual_memory().available / (1024**3)
    required_gb = (n_channels * len(raw.times) * 8) / (1024**3)  # float64
    print(f"Available memory: {available_gb:.1f} GB, Required: {required_gb:.1f} GB")
    
    memory_per_channel = (len(raw.times) * 8) / (1024**3)
    print(f"Memory per channel: {memory_per_channel:.2f} GB")
    
    if required_gb > available_gb * 0.7:  # Use 70% threshold to be safe
        print("⚠ Large file detected, using dynamic channel-wise filtering...")
        
        # Process channels dynamically
        processed_channels = []
        channels_remaining = list(raw.ch_names)
        while channels_remaining:
            # Calculate optimal batch size based on CURRENT available memory
            current_available_gb = psutil.virtual_memory().available / (1024**3)
            
            # Be much more conservative due to memory fragmentation and peak usage
            # Use only 10% of available memory to account for:
            # - Memory fragmentation (non-contiguous blocks)
            # - Peak memory during MNE operations (2-3x temporary overhead)
            # - Filter operations creating additional arrays
            max_channels_this_batch = max(1, int((current_available_gb * 0.1) / memory_per_channel))
            
            # Cap at reasonable batch size to avoid fragmentation issues
            max_channels_this_batch = min(max_channels_this_batch, 25)
            
            # Don't exceed remaining channels
            batch_size = min(max_channels_this_batch, len(channels_remaining))
            
            current_batch = channels_remaining[:batch_size]
            channels_remaining = channels_remaining[batch_size:]
            
            print(f"  Processing {len(current_batch)} channels ({len(processed_channels)+1}-{len(processed_channels)+len(current_batch)}/{n_channels})")
            estimated_usage = batch_size * memory_per_channel
            print(f"    Available: {current_available_gb:.1f} GB, Estimated: {estimated_usage:.1f} GB (conservative)")
            
            # Process this batch with error handling
            try:
                temp_raw = raw.copy().pick(current_batch)
                temp_raw.load_data()
                
                # TODO: check what's the relevant notch
                temp_raw.notch_filter((50, 100, 150, 200), method='spectrum_fit', phase='zero-double')
                temp_raw.filter(l_freq=0.1, h_freq=40, phase='zero-double')
                temp_raw.resample(250)
                
                # Store individual processed channels
                for ch_name in current_batch:
                    processed_channels.append(temp_raw.copy().pick([ch_name]))
                
                # Cleanup batch
                del temp_raw
                gc.collect()
                
            except MemoryError as e:
                print(f"    MemoryError with {len(current_batch)} channels, reducing batch size...")
                # Put channels back and try with smaller batch
                channels_remaining = current_batch + channels_remaining
                
                # Force much smaller batch size (single channel if needed)
                if len(current_batch) > 1:
                    # Try with half the batch size
                    new_batch_size = max(1, len(current_batch) // 2)
                    current_batch = channels_remaining[:new_batch_size]
                    channels_remaining = channels_remaining[new_batch_size:]
                    print(f"    Retrying with {len(current_batch)} channels...")
                    
                    # Process reduced batch
                    temp_raw = raw.copy().pick(current_batch)
                    temp_raw.load_data()
                    temp_raw.notch_filter((50, 100, 150, 200), method='spectrum_fit', phase='zero-double')
                    temp_raw.filter(l_freq=0.1, h_freq=40, phase='zero-double')
                    temp_raw.resample(1000)
                    
                    for ch_name in current_batch:
                        processed_channels.append(temp_raw.copy().pick([ch_name]))
                    
                    del temp_raw
                    gc.collect()
                else:
                    raise  # If single channel fails, there's a bigger problem
        
        print("  Combining processed channels...")
        
        raw_combined = processed_channels[0]
        for i, ch_raw in enumerate(processed_channels[1:], 1):
            raw_combined.add_channels([ch_raw])
            if i % 50 == 0:  # Periodic cleanup during combination
                gc.collect()
        
        # Clean up individual channel objects
        for ch_raw in processed_channels:
            del ch_raw
        del processed_channels
        gc.collect()
        
        print("  OK: Dynamic channel-wise filtering completed")
        
        # Return the combined raw object instead of trying to modify in-place
        return raw_combined
        
    else:
        # Normal processing for smaller files
        print("Normal processing (sufficient memory available)")
        raw.load_data()
        # TODO: check what's the relevant notch
        raw.notch_filter((50, 100, 150, 200), method='spectrum_fit', phase='zero-double')
        raw.filter(l_freq=0.1, h_freq=40, phase='zero-double')
        raw.resample(1000)
    
    # Make data contiguous
    raw._data = np.ascontiguousarray(raw._data)
    
    print("  OK: Filtering completed")
    return None  # Return None for normal processing (raw object modified in-place)


def resample_and_filter(raw):
    excluded_channels = set(FACE_ELECTRODES + NECK_ELECTRODES + EAR_ELECTRODES)
    channel_types = raw.get_channel_types()
    channels = [ch for ch, ch_type in zip(raw.ch_names, channel_types) 
                if ch not in excluded_channels and ch_type == 'eeg']
    raw.pick(channels)
    
    print(f"Processing {len(raw.ch_names)} channels, duration: {raw.times[-1]/60:.1f} minutes")
    print(f"Original sampling rate: {raw.info['sfreq']} Hz")
    
    if raw.info['sfreq'] != 250:
        print("  Step 1: Resampling to 250 Hz...")
        resample_start = time.time()
        raw.resample(250)
        print(f"    Resample took: {time.time() - resample_start:.1f} seconds")
    
    print("  Loading data into memory...")
    load_start = time.time()
    raw.load_data()
    print(f"    Load took: {time.time() - load_start:.1f} seconds")
    
    print("  Step 2: Applying notch filter...")
    notch_start = time.time()
    # After resampling to 250 Hz, Nyquist is 125 Hz
    # Can only filter 50 Hz and 100 Hz (150 and 200 are above Nyquist)
    raw.notch_filter([50, 100], method='fir', phase='zero', verbose=True, n_jobs=-1)
    print(f"    Notch filter took: {time.time() - notch_start:.1f} seconds")
    
    print("  Step 3: Applying band-pass filter (0.1-40 Hz)...")
    filter_start = time.time()
    # Using n_jobs=-1 to use all CPU cores
    raw.filter(l_freq=0.1, h_freq=40, phase='zero', verbose=True, n_jobs=-1)
    print(f"    Band-pass filter took: {time.time() - filter_start:.1f} seconds")


def add_annotations(raw, annotations_path):

    annotations = mne.read_annotations(annotations_path)
    # Extract orig_time from annotations file for verification
    with open(annotations_path, 'r') as f:
        lines = f.readlines()
        orig_time_line = [line for line in lines if line.startswith('# orig_time')]
        if orig_time_line:
            annotations_orig_time = orig_time_line[0].strip().split(' : ')[1]
            raw_orig_time = str(raw.info['meas_date'])
            if parser.parse(raw_orig_time, ignoretz=True) != parser.parse(annotations_orig_time, ignoretz=True):
                raise ValueError(f"orig_time mismatch: annotations ({annotations_orig_time}) vs raw ({raw_orig_time})")
    
    raw.set_annotations(annotations)


def add_sleep_scoring(raw, sub, epoch_duration, hypnogram_path=None):
    """
    Add sleep scoring annotations to raw data.
    epoch_duration : int, Duration of each epoch in seconds (usually 30 or 1)
    hypnogram_path : str, optional. If None, defaults to {BASE_DIR}/scoring/MCI/{sub}_hypno.txt
    """
    if hypnogram_path is None:
        hypnogram_path = f"{BASE_DIR}/scoring/MCI/{sub}_hypno.txt"
    
    # Read file and convert W→0, R→4
    stage_mapping = {'W': 0, 'R': 4, '?': 0}
    with open(hypnogram_path, 'r') as f:
        hypno = np.array([stage_mapping[val] if val in stage_mapping else int(val) 
                          for val in (line.strip() for line in f)])

    expected_epochs = int(raw.times[-1]) // epoch_duration
    diff_mins = np.abs((expected_epochs - len(hypno)) * epoch_duration / 60)
    if len(hypno) > expected_epochs:
        hypno = hypno[:expected_epochs]
        note_message = f"hypno is longer by {diff_mins:.2f} minutes (epoch_duration={epoch_duration}s)"
        print(note_message)
        
        os.makedirs("notes", exist_ok=True)
        with open("notes/notes.txt", "a", encoding="utf-8") as f:
            f.write(f"{sub} - {note_message}\n")

    stage_map = {-1: 'UNKNOWN', 0: 'Wake', 1: 'NREM1', 2: 'NREM2', 3: 'NREM3', 4: 'REM'}

    # Collapse into annotation blocks (convert indices to seconds)
    onsets, durations, descriptions = [], [], []
    start, current = 0, hypno[0]
    for i in range(1, len(hypno)):
        if hypno[i] != current:
            onsets.append(start * epoch_duration)
            durations.append((i - start) * epoch_duration)
            descriptions.append(stage_map.get(current, 'UNKNOWN'))
            start, current = i, hypno[i]

    onsets.append(start * epoch_duration)
    durations.append((len(hypno) - start) * epoch_duration)
    descriptions.append(stage_map.get(current, 'UNKNOWN'))

    sleep_annot = mne.Annotations(onsets, durations, descriptions, raw.annotations.orig_time)

    # Combine with existing annotations 
    annotations = raw.annotations
    raw.set_annotations(annotations + sleep_annot)


def save_processed_raw(raw, sub, output_dir):
    """Save processed raw data and its annotations into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    n_channels = len(raw.ch_names)
    title = f"{n_channels}-head-ch_resample250_filtered_scored_bad-epochs_raw"

    output_path = f'{output_dir}/{sub}_{title}.fif'
    raw.save(output_path, overwrite=True)
    annotations_output_path = f'{output_dir}/{sub}_annotations.txt'
    raw.annotations.save(annotations_output_path)

    print(f"  Files saved to: {output_dir}")


def main():
    # Set MNE to use less memory and enable parallel processing cache
    mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M')
    mne.set_config('MNE_CACHE_DIR', 'I:/Shaked/ISO/tmp/mne_cache')
    mne.set_config('MNE_USE_NUMBA', 'false')
    os.makedirs('I:/Shaked/ISO/tmp/mne_cache', exist_ok=True)

    # ── Configuration for elderly control MCI subjects ──
    input_dir = "E:/May/May_MCI_clean/input_files"
    directory = f"{BASE_DIR}/control_raw"
    annotations_dir = "E:/May/May_MCI_clean"


    error_by_subject = {}
    files = sorted(os.listdir(directory))
    n_subjects = len(files)

    for i, file in enumerate(files):
        sub = file.split('_')[0]
        output_dir = f'{BASE_DIR}/elderly_control_clean/a_the_rest/{sub}'
        if os.path.exists(output_dir):
            print(f"Skipping {sub} - already processed")
            continue

        iteration_start_time = time.time()
        raw = None

        print(f"\n{'='*60}")
        print(f"PROCESSING SUBJECT {i+1}/{n_subjects}: {sub}")
        print(f"{'='*60}")

        try:
            step = "loading"
            step_start_time = time.time()
            print(f"{step}: {sub}")
            raw = mne.io.read_raw(os.path.join(directory, file), preload=False)
            step_duration = time.time() - step_start_time
            print(f"  OK: {step} completed in {step_duration:.1f} seconds")

            step = "filtering"
            step_start_time = time.time()
            print(f"{step}: {sub}")
            resample_and_filter(raw)
            n_channels = len(raw.ch_names)
            assert n_channels == 176, f"Expected 176 channels, got {n_channels}"
            step_duration = time.time() - step_start_time
            print(f"  OK: {step} completed in {step_duration:.1f} seconds ({n_channels} channels)")

            step = "annotations"
            step_start_time = time.time()
            print(f"{step}: {sub}")
            annot_folders = glob.glob(os.path.join(annotations_dir, f"{sub}_*"))
            if not annot_folders:
                raise FileNotFoundError(f"No annotation folder found for {sub} in {annotations_dir}")
            annot_path = os.path.join(annot_folders[0], "CleaningPipe", "annotations.txt")
            add_annotations(raw, annot_path)
            step_duration = time.time() - step_start_time
            print(f"  OK: {step} completed in {step_duration:.1f} seconds")

            step = "sleep scoring"
            step_start_time = time.time()
            print(f"{step}: {sub}")
            hypno_path = f"{BASE_DIR}/scoring/elderly_control/{sub}_hypno.txt"
            add_sleep_scoring(raw, sub, epoch_duration=30, hypnogram_path=hypno_path)
            step_duration = time.time() - step_start_time
            print(f"  OK: {step} completed in {step_duration:.1f} seconds")

            step = "saving"
            step_start_time = time.time()
            print(f"{step}: {sub} ->{output_dir}")
            save_processed_raw(raw, sub, output_dir)
            step_duration = time.time() - step_start_time
            print(f"  OK: {step} completed in {step_duration:.1f} seconds")

            iteration_duration = time.time() - iteration_start_time
            print(f"\nSubject {sub} processed successfully!")
            print(f"  Total time: {iteration_duration:.1f} seconds ({iteration_duration/60:.1f} minutes)")

        except Exception as e:
            iteration_duration = time.time() - iteration_start_time
            error_traceback = traceback.format_exc()
            error_by_subject[sub] = (step, e, error_traceback)
            print(f"FAIL: Error in {step} for {sub}: {e}")
            print(f"  Time before error: {iteration_duration:.1f} seconds ({iteration_duration/60:.1f} minutes)")
            print(f"Full traceback:\n{error_traceback}")

        finally:
            if raw is not None:
                raw.close()
                del raw
            for _ in range(3):
                gc.collect()
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            available_gb = psutil.virtual_memory().available / (1024**3)
            print(f"  Memory after cleanup: {memory_usage_mb:.1f} MB used, {available_gb:.1f} GB available")

    gc.collect()

    if error_by_subject:
        print(f"\nProcessing completed: {n_subjects - len(error_by_subject)}/{n_subjects} successful")
        print("Errors encountered:")
        for sub, error_info in error_by_subject.items():
            step, error, traceback_str = error_info
            print(f"  {sub}: Failed at {step} - {error}")
            traceback_lines = traceback_str.strip().split('\n')
            print(f"    Last error line: {traceback_lines[-1]}")
    else:
        print(f"\nAll {n_subjects} subjects processed successfully!")


if __name__ == "__main__":
    main()