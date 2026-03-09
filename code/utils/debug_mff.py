import os
import xml.etree.ElementTree as ET
from mne.io.egi.general import _get_blocks, _get_signalfname

MICROSEC_PER_SEC = 1000000


def _parse_epochs_xml(epochs_path):
    # Parse epochs.xml file and extract epoch times.
    try:
        tree = ET.parse(epochs_path)
        root = tree.getroot()
    except ET.ParseError:
        print("Error: Could not parse epochs.xml file")
        return None
    
    epochs = []
    for epoch_elem in root:
        if epoch_elem.tag.endswith('epoch'):
            begin_time = None
            end_time = None
            for child in epoch_elem:
                if child.tag.endswith('beginTime'):
                    begin_time = int(child.text)
                elif child.tag.endswith('endTime'):
                    end_time = int(child.text)
            
            if begin_time is not None and end_time is not None:
                epochs.append({'begin': begin_time, 'end': end_time})
    
    return epochs if epochs else None


def diagnose_mff_epochs(mff_path):
    """
    Analyzes MFF epochs.xml and provides exact fixes for validation errors.
    Shows only the XML changes needed to fix epoch validation issues.
    """
    epochs_path = os.path.join(mff_path, "epochs.xml")
    if not os.path.exists(epochs_path):
        return None
    
    epochs = _parse_epochs_xml(epochs_path)
    if not epochs:
        return None
    
    # Get signal information
    try:
        all_files = _get_signalfname(mff_path)
        eeg_file = all_files["EEG"]["signal"]
        fname = os.path.join(mff_path, eeg_file)
        signal_blocks = _get_blocks(fname)
        sfreq = signal_blocks['sfreq']
        total_block_samples = signal_blocks["samples_block"].sum()
    except Exception:
        print("Error: Could not read signal blocks from MFF file")
        return None
    
    # Convert epochs to samples (microseconds to samples using sampling frequency)
    first_samps = [int(epoch['begin'] / MICROSEC_PER_SEC * sfreq) for epoch in epochs]
    last_samps = [int(epoch['end'] / MICROSEC_PER_SEC * sfreq) for epoch in epochs]
    
    # Perform validation checks
    total_epoch_samples = sum(end - start for start, end in zip(first_samps, last_samps))
    
    # Check 1: Sample count match
    check1 = total_epoch_samples == total_block_samples
    diff = total_epoch_samples - total_block_samples
    
    # Check 2: Valid ranges
    invalid_epochs = [i for i, (start, end) in enumerate(zip(first_samps, last_samps)) if start >= end]
    check2 = len(invalid_epochs) == 0
    
    # Check 3: Sequential epochs (no overlaps)
    overlapping_epochs = [i for i in range(len(first_samps) - 1) 
                         if first_samps[i+1] < last_samps[i]]
    check3 = len(overlapping_epochs) == 0
    
    # Generate fixes
    if check1 and check2 and check3:
        print("No fixes needed - file is valid!")
        return True
    
    # Print XML changes needed
    if not check1:
        new_end_time = epochs[-1]['end'] + int((-diff if diff > 0 else abs(diff)) / sfreq * MICROSEC_PER_SEC)
        print(f"Change <endTime>{epochs[-1]['end']}</endTime> to <endTime>{new_end_time}</endTime>")
    
    if not check2:
        for idx in invalid_epochs:
            print(f"Swap times for epoch {idx}: <beginTime>{epochs[idx]['begin']}</beginTime> ↔ <endTime>{epochs[idx]['end']}</endTime>")
    
    if not check3:
        for idx in overlapping_epochs:
            new_end_time = int((first_samps[idx+1] - 1) / sfreq * MICROSEC_PER_SEC)
            print(f"Change <endTime>{epochs[idx]['end']}</endTime> to <endTime>{new_end_time}</endTime>")
    
    return True


def main():
    """Run the MFF diagnostic tool on an example subject."""
    mff_path = "I:/Shaked/ISO_data/control_raw/ON68_Sleep_20230109_234420.mff"
    
    print(f"Analyzing MFF file: {os.path.basename(mff_path)}")
    print("-" * 50)
    
    result = diagnose_mff_epochs(mff_path)
    
    if result is None:
        print("Could not analyze file - check if epochs.xml exists and is valid")


if __name__ == "__main__":
    main()
