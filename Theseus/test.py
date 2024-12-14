import pickle

# Load from a pickle file
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print("Loaded data:", loaded_data)