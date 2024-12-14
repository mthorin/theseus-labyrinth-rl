import random

import torch
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.theseus import Theseus
from Labyrinth.player import all_player_colours

from tqdm import tqdm

from Theseus.theseus_network import TheseusNetwork

CURRENT_BEST_MODEL_PATH = 'theseus_best.pt'

def evaluate(n, new_model):
    ruleset = RuleSet()
    utils.enable_colours(True)

    # initialize current best network
    curr_best_network = TheseusNetwork()
    curr_best_network.load_state_dict(torch.load(CURRENT_BEST_MODEL_PATH, weights_only=True))
    curr_best_network.eval()

    # initialize 
    network = new_model

    wins = 0

    loop = tqdm(total=n, position=0, leave=False)
    for i in range(n):
        players = [Theseus(colour, curr_best_network) for colour in all_player_colours]

        colour = random.choice(all_player_colours)

        players[all_player_colours.index(colour)] = Theseus(colour, network)

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1

        if lab.who_won().colour == colour:
            wins += 1 

        loop.update(1)
        loop.set_description('Games Played: {} Win rate: {}'.format(n, wins/n))

    if wins/n > .275:
        # replace curr_best_network on disk
        torch.save(network.state_dict(), CURRENT_BEST_MODEL_PATH)
        