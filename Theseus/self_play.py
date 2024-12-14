import os
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.theseus import Theseus
from Labyrinth.player import all_player_colours

import pickle
from tqdm import tqdm

MAX_DATA_SIZE = 500000

def self_play(n, data_file_path):
    # Load previous game list
    if os.path.exists(data_file_path):
        with open(data_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = []

    ruleset = RuleSet()
    utils.enable_colours(True)

    # intialize network #TODO
    network = None

    loop = tqdm(total=n, position=0, leave=False)
    for i in range(n):
        game_data = {color: [] for color in all_player_colours}
        players = [Theseus(colour, network, exploration_weight=1, data_bank=game_data[colour]) for colour in all_player_colours]

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1

        for colour, results in game_data.items():
            value = 0
            if colour == lab.who_won().colour:
                value = 1

            for frame in results:
                data.append(frame + (value,))
                if len(data) > MAX_DATA_SIZE:
                    data.pop(0)

        loop.update(1)
        loop.set_description('Games Played: {}'.format(n))

    # Save list
    with open(data_file_path, 'wb') as file:
        pickle.dump(data, file)