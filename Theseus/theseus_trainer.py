import os
import re
import torch
from Theseus.evaluator import evaluate
from Theseus.self_play import self_play
from Theseus.optimization import optimize

DATA_FILE_PATH = 'Theseus/data.pkl'
ARCHIVE_LOCATION = 'Theseus/model_archive'

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
    
    return latest_file, latest_version

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # find latest model
    latest_file, latest_version = find_latest_versioned_file(ARCHIVE_LOCATION)

    while 1:
        latest_version += 1
        new_model = optimize(device, DATA_FILE_PATH, latest_file)
        torch.save(new_model.state_dict(), f"{ARCHIVE_LOCATION}/theseus_v{latest_version}.pt")
        evaluate(new_model)
        self_play(DATA_FILE_PATH)

if __name__ == '__main__':
    main()