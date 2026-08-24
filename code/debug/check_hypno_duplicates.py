import os
import pandas as pd
import requests
from io import BytesIO

# Google Sheets URL - convert to Excel export URL to access specific sheet
sheet_url = "https://docs.google.com/spreadsheets/d/1BE0Yu-wECLe0NkdIIvxGjYKNL84FRgUE/export?format=xlsx"

# Read the Google Sheet - specific tab "included in analysis"
response = requests.get(sheet_url)
df = pd.read_excel(BytesIO(response.content), sheet_name="included in analysis")

# Get the "elderly control" column
if "elderly control" not in df.columns:
    print("Column 'elderly control' not found. Available columns:")
    print(df.columns.tolist())
    exit(1)

elderly_control_ids = df["elderly control"].dropna().astype(str).tolist()

# Filter out the rows to ignore
ignore_values = ["ID", "min", "max", "average", "count"]
elderly_control_ids = [id for id in elderly_control_ids if id not in ignore_values]

# Directory
scoring_dir = r"I:\Shaked\ISO_data\scoring"
elderly_control_dir = r"I:\Shaked\ISO_data\scoring\elderly_control"

# Create elderly_control directory if it doesn't exist
os.makedirs(elderly_control_dir, exist_ok=True)

for subject_id in elderly_control_ids:
    # Search for subject in scoring directory and subdirs (case insensitive)
    found_files = []
    
    for root, dirs, files in os.walk(scoring_dir):
        for file in files:
            if subject_id.lower() in file.lower():
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    
    # Print results
    print(f"\n{subject_id}:")
    if not found_files:
        print(f"  didn't find")
        continue
    
    # Separate description files from regular files
    description_files = [f for f in found_files if "_description" in os.path.basename(f).lower()]
    regular_files = [f for f in found_files if "_description" not in os.path.basename(f).lower()]
    
    # Delete all description files immediately
    for desc_file in description_files:
        try:
            os.remove(desc_file)
            print(f"  Deleted (description): {os.path.relpath(desc_file, scoring_dir)}")
        except Exception as e:
            print(f"  Error deleting {os.path.relpath(desc_file, scoring_dir)}: {e}")
    
    # Further filter to only .txt files for comparison (exclude .mat, .csv, etc.)
    txt_files = [f for f in regular_files if f.lower().endswith('.txt')]
    non_txt_files = [f for f in regular_files if not f.lower().endswith('.txt')]
    
    # Case 1: Only one file found
    if len(found_files) == 1:
        src = found_files[0]
        dest = os.path.join(elderly_control_dir, os.path.basename(src))
        # Only move if not already in elderly_control
        if os.path.normpath(os.path.dirname(src)) != os.path.normpath(elderly_control_dir):
            os.rename(src, dest)
            print(f"  Moved: {os.path.relpath(src, scoring_dir)} -> elderly_control\\{os.path.basename(dest)}")
        else:
            print(f"  Already in elderly_control: {os.path.basename(src)}")
    
    # Case 2: 2 files found and one is description file
    elif len(found_files) == 2 and len(description_files) == 1 and len(txt_files) == 1:
        txt_file = txt_files[0]
        
        # Move regular file (description already deleted)
        dest = os.path.join(elderly_control_dir, os.path.basename(txt_file))
        if os.path.normpath(os.path.dirname(txt_file)) != os.path.normpath(elderly_control_dir):
            os.rename(txt_file, dest)
            print(f"  Moved: {os.path.relpath(txt_file, scoring_dir)} -> elderly_control\\{os.path.basename(dest)}")
        else:
            print(f"  Already in elderly_control: {os.path.basename(txt_file)}")
    
    # Case 3: Multiple files - compare non-description files
    else:
        print(f"  Found {len(found_files)} files:")
        for f in found_files:
            print(f"    - {os.path.relpath(f, scoring_dir)}")
        
        if non_txt_files:
            print(f"  Note: Ignoring {len(non_txt_files)} non-txt files for comparison")
        
        if len(txt_files) >= 2:
            # Compare txt files
            print(f"  Comparing {len(txt_files)} txt files:")
            
            # Read first file
            with open(txt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                first_content = f.read()
            
            all_identical = True
            for i in range(1, len(txt_files)):
                with open(txt_files[i], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content == first_content:
                    print(f"    {os.path.relpath(txt_files[0], scoring_dir)} == {os.path.relpath(txt_files[i], scoring_dir)}: IDENTICAL")
                else:
                    print(f"    {os.path.relpath(txt_files[0], scoring_dir)} != {os.path.relpath(txt_files[i], scoring_dir)}: DIFFERENT")
                    all_identical = False
            
            if all_identical:
                print(f"  All txt files are identical")
                
                # Delete files with longer names, keep the shortest one
                # If multiple files have the same shortest length, keep the first one
                txt_files_sorted = sorted(txt_files, key=lambda x: len(os.path.basename(x)))
                file_to_keep = txt_files_sorted[0]
                
                for file_to_delete in txt_files_sorted[1:]:
                    try:
                        os.remove(file_to_delete)
                        print(f"  Deleted (duplicate): {os.path.relpath(file_to_delete, scoring_dir)}")
                    except Exception as e:
                        print(f"  Error deleting {os.path.relpath(file_to_delete, scoring_dir)}: {e}")
                
                # Move the kept file to elderly_control
                dest = os.path.join(elderly_control_dir, os.path.basename(file_to_keep))
                if os.path.normpath(os.path.dirname(file_to_keep)) != os.path.normpath(elderly_control_dir):
                    os.rename(file_to_keep, dest)
                    print(f"  Moved: {os.path.relpath(file_to_keep, scoring_dir)} -> elderly_control\\{os.path.basename(dest)}")
                else:
                    print(f"  Already in elderly_control: {os.path.basename(file_to_keep)}")
        elif len(txt_files) == 1:
            # Only one txt file, move it (description files already deleted)
            txt_file = txt_files[0]
            
            dest = os.path.join(elderly_control_dir, os.path.basename(txt_file))
            if os.path.normpath(os.path.dirname(txt_file)) != os.path.normpath(elderly_control_dir):
                os.rename(txt_file, dest)
                print(f"  Moved: {os.path.relpath(txt_file, scoring_dir)} -> elderly_control\\{os.path.basename(dest)}")
            else:
                print(f"  Already in elderly_control: {os.path.basename(txt_file)}")
