import os
import glob

# Find all AB7 files
directory = r"I:\Shaked\ISO_data\scoring\MCI"
ab7_files = glob.glob(os.path.join(directory, "AB7*"))

print(f"Found {len(ab7_files)} AB7 files:")
for f in ab7_files:
    print(f"  - {os.path.basename(f)}")
print()

def compare_files(file1_path, file2_path):
    """Compare two hypnogram files and print differences"""
    with open(file1_path, 'r') as f:
        file1_content = [line.strip() for line in f.readlines()]
    
    with open(file2_path, 'r') as f:
        file2_content = [line.strip() for line in f.readlines()]
    
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    
    print(f"\n{'='*80}")
    print(f"Comparing: {file1_name} vs {file2_name}")
    print(f"{'='*80}")
    
    print(f"{file1_name} length: {len(file1_content)}")
    print(f"{file2_name} length: {len(file2_content)}")
    
    # Count -1 values
    file1_neg1_count = file1_content.count('-1')
    file2_neg1_count = file2_content.count('-1')
    print(f"{file1_name} has {file1_neg1_count} values of -1")
    print(f"{file2_name} has {file2_neg1_count} values of -1")
    
    # Calculate difference and skip from longer file
    len_diff = abs(len(file1_content) - len(file2_content))
    print(f"Length difference: {len_diff}")
    
    if len(file1_content) > len(file2_content):
        file1_content = file1_content[len_diff:]
        print(f"Skipping first {len_diff} lines from {file1_name}")
    elif len(file2_content) > len(file1_content):
        file2_content = file2_content[len_diff:]
        print(f"Skipping first {len_diff} lines from {file2_name}")
    
    print(f"Comparing remaining lines: {min(len(file1_content), len(file2_content))}")
    
    # Find differences
    max_len = max(len(file1_content), len(file2_content))
    differences = []
    
    for i in range(max_len):
        file1_val = file1_content[i] if i < len(file1_content) else "MISSING"
        file2_val = file2_content[i] if i < len(file2_content) else "MISSING"
        
        # Treat 0 and -1 as equal
        if file1_val in ['0', '-1'] and file2_val in ['0', '-1']:
            continue
        
        if file1_val != file2_val:
            differences.append(i)
    
    print(f"\nNumber of differences: {len(differences)}")
    
    if len(differences) > 0:
        print("\nFirst 10 differences:")
        for i in differences[:10]:
            file1_val = file1_content[i] if i < len(file1_content) else "MISSING"
            file2_val = file2_content[i] if i < len(file2_content) else "MISSING"
            print(f"  Line {i+1}: {file1_name}={file1_val}, {file2_name}={file2_val}")
        
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more differences")
    else:
        print("\nFiles are IDENTICAL (after skipping length difference)")

# Compare all pairs
for i in range(len(ab7_files)):
    for j in range(i+1, len(ab7_files)):
        compare_files(ab7_files[i], ab7_files[j])
