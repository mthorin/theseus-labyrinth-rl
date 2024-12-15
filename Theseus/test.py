import os
import pickle


if os.path.exists('Theseus/data.pkl'):
        with open('Theseus/data.pkl', 'rb') as file:
            data = pickle.load(file)
else:
    data = []

print(len(data))