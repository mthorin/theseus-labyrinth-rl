import os
import re

def find_latest_versioned_file(folder_path):
    # Regular expression to match files ending with "vX" where X is a version number
    version_pattern = re.compile(r'v(\d+)(\.\w+)?$')
    
    latest_file = None
    latest_version = -1

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        match = version_pattern.search(filename)
        if match:
            # Extract the version number as an integer
            version = int(match.group(1))
            # Update the latest file and version if this version is higher
            if version > latest_version:
                latest_version = version
                latest_file = filename
    
    return latest_file

# Usage example
folder_path = "Theseus/model_archive"  # Replace with your folder path
latest_file = find_latest_versioned_file(folder_path)
print("Latest file:", latest_file)