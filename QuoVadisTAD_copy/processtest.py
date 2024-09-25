import re
import os
# Define the directory containing the files
directory = r'C:\Users\jhaja\OneDrive\Desktop\quovadis\QuoVadisTAD\resources\processed_datasets\smd'

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Match the pattern machine-x-x_test_label.npy
    match = re.match(r"(machine)-(\d+)-(\d+)_test_label.npy", filename)
    if match:
        # Construct the new filename
        new_filename = f"{match.group(1)}-{match.group(2)}-{match.group(3)}_labels.npy"

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))