import numpy as np
import mne


def _split_segment_around_bads(sleep_start, sleep_end, bad_segments):
    """
    Split a sleep segment around BAD annotations, extracting clean parts.
    
    Example:
        Sleep: [0, 600]
        BAD: [200, 250], [400, 450]
        Result: [(0, 200), (250, 400), (450, 600)]
    """
    if len(bad_segments) == 0:
        return [(sleep_start, sleep_end)]
    
    # Find BAD segments that overlap with this sleep period
    overlapping_bads = []
    for bad_start, bad_end in bad_segments:
        # Check if there's any overlap
        if not (bad_end <= sleep_start or bad_start >= sleep_end):
            overlapping_bads.append((bad_start, bad_end))
    
    if len(overlapping_bads) == 0:
        return [(sleep_start, sleep_end)]
    
    # Sort BAD segments by start time
    overlapping_bads = sorted(overlapping_bads, key=lambda x: x[0])
    
    # Extract valid segments between BAD periods
    valid_segments = []
    current_pos = sleep_start
    
    for bad_start, bad_end in overlapping_bads:
        # Add the clean segment before this BAD annotation
        if current_pos < bad_start:
            valid_segments.append((current_pos, bad_start))
        # Move past the BAD segment
        current_pos = max(current_pos, bad_end)
    
    # Add remaining segment after the last BAD annotation
    if current_pos < sleep_end:
        valid_segments.append((current_pos, sleep_end))
    
    return valid_segments


def extract_clean_sleep_bouts(raw, include_n3=False, min_bout_duration=300):
    """
    Identifies clean sleep bouts by splitting around BAD annotations.
    Returns concatenated data and bout boundaries.
    """
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    
    # Define which sleep stage annotations we're looking for
    target_labels = ['NREM2', 'N2']
    if include_n3:
        target_labels.extend(['NREM3', 'N3'])
    
    # Parse all annotations into a structured list
    sleep_segments = []
    bad_segments = []
    
    for ann in raw.annotations:
        onset_sec = ann['onset']
        duration_sec = ann['duration']
        description = ann['description']
        
        # Convert to sample indices
        onset_sample = int(onset_sec * sfreq)
        end_sample = int((onset_sec + duration_sec) * sfreq)
        end_sample = min(end_sample, n_samples)  # Clip to recording length
        
        if description in target_labels:
            sleep_segments.append((onset_sample, end_sample))
        elif 'BAD' in description.upper():
            bad_segments.append((onset_sample, end_sample))
    
    # Process each sleep segment: split around BAD annotations
    clean_segments = []
    for sleep_start, sleep_end in sleep_segments:
        # Split this sleep period around any overlapping BAD segments
        split_segments = _split_segment_around_bads(sleep_start, sleep_end, bad_segments)
        clean_segments.extend(split_segments)

    # Merge consecutive segments (in case adjacent N2 epochs create continuous clean data)
    if len(clean_segments) > 0:
        clean_segments = sorted(clean_segments, key=lambda x: x[0])  # Sort by start time
        merged_segments = []
        current_start, current_end = clean_segments[0]
        
        for start, end in clean_segments[1:]:
            if start <= current_end:  # Overlapping or adjacent
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))
        clean_segments = merged_segments
    
    # Filter by minimum duration
    min_samples = int(min_bout_duration * sfreq)
    valid_bouts = [(s, e) for s, e in clean_segments if (e - s) >= min_samples]
    
    if len(valid_bouts) == 0:
        print("WARNING: No valid clean bouts found!")
        n_channels = len(raw.ch_names)
        return np.array([]).reshape(n_channels, 0), np.array([]).reshape(2, 0)
    
    # Extract and concatenate data
    raw_data = raw.get_data()
    all_chunks = []
    boundaries = []
    curr_pos = 0
    
    for start, end in valid_bouts:
        chunk = raw_data[:, start:end]
        all_chunks.append(chunk)
        
        chunk_len = chunk.shape[1]
        boundaries.append([curr_pos, curr_pos + chunk_len - 1])
        curr_pos += chunk_len
    
    concatenated_data = np.hstack(all_chunks)
    boundaries_array = np.array(boundaries).T
    
    total_duration = curr_pos / sfreq
    print(f"Extracted {len(valid_bouts)} clean bouts (Total: {total_duration:.2f}s, {total_duration/60:.2f}min)")
    
    return concatenated_data, boundaries_array