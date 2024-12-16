import os

import torch
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.theseus import Theseus
from Labyrinth.player import all_player_colours
from theseus_network import TheseusNetwork

import pickle
from tqdm import tqdm

MAX_DATA_SIZE = 500000

class TensorLog:
    entries = {}
    calls = 0

    def __init__(self):
        TensorLog.entries = {color: [] for color in all_player_colours}
        TensorLog.calls = 0

    def get_log_function(self, colour):
        if all_player_colours.index(colour) == 0:
            return self.save_entry_0
        elif all_player_colours.index(colour) == 1:
            return self.save_entry_1
        elif all_player_colours.index(colour) == 2:
            return self.save_entry_2
        else: 
            return self.save_entry_3
        
    def save_entry_1(self, entry):
        self.save_entry(entry, all_player_colours[1])

    def save_entry_2(self, entry):
        self.save_entry(entry, all_player_colours[2])

    def save_entry_3(self, entry):
        self.save_entry(entry, all_player_colours[3])

    def save_entry_0(self, entry):
        self.save_entry(entry, all_player_colours[0])

    def save_entry(self, entry, colour):
        TensorLog.entries[colour].append(entry)
        TensorLog.calls += 1

    def get_all_entries(self):
        return self.entries

def self_play(device, data_file_path, best_model_path, n=1000):
    # Load previous game list
    if os.path.exists(data_file_path):
        with open(data_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = []

    ruleset = RuleSet()
    utils.enable_colours(True)

    # initialize network
    network = TheseusNetwork()
    network.load_state_dict(torch.load(best_model_path, weights_only=True))
    network.eval()
    network = network.to(device)

    loop = tqdm(total=n, position=0, leave=False)
    for i in range(n):
        game_data = TensorLog()
        players = [Theseus(colour, network, device, exploration_weight=1, data_bank=game_data.get_log_function(colour)) for colour in all_player_colours]

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1
            loop.set_description('Turn: {}'.format(turns))

        for colour, results in game_data.get_all_entries().items():
            value = 0
            if colour == lab.who_won().colour:
                value = 1

            for frame in results:
                data.append(frame + (value,))
                if len(data) > MAX_DATA_SIZE:
                    data.pop(0)

        loop.update(1)

    # Save list
    with open(data_file_path, 'wb') as file:
        pickle.dump(data, file)