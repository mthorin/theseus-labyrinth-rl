import os
import re
import torch
from evaluator import evaluate
from self_play import self_play
from optimization import optimize

# DATA_FILE_PATH = 'Theseus/data.pkl'
# ARCHIVE_LOCATION = 'Theseus/model_archive'
# CURRENT_BEST_MODEL_PATH = 'Theseus/theseus_best.pt'

DATA_FILE_PATH = 'data.pkl'
ARCHIVE_LOCATION = 'model_archive'
CURRENT_BEST_MODEL_PATH = 'theseus_best.pt'

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
        print(f"Optimization, evaluation, and self-play for version {latest_version}")
        new_model = optimize(device, DATA_FILE_PATH, f"{ARCHIVE_LOCATION}/{latest_file}", n_iterations=50)
        torch.save(new_model.state_dict(), f"{ARCHIVE_LOCATION}/theseus_v{latest_version}.pt")
        evaluate(new_model, CURRENT_BEST_MODEL_PATH)
        self_play(DATA_FILE_PATH, CURRENT_BEST_MODEL_PATH)

if __name__ == '__main__':
    main()