import os
import random

import torch
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.theseus import Theseus
from Labyrinth.player import all_player_colours

from tqdm import tqdm

from theseus_network import TheseusNetwork


def evaluate(device, new_model, best_model_path, n=100):
    if not os.path.exists(best_model_path):
        torch.save(new_model.state_dict(), best_model_path)
        return
        
    ruleset = RuleSet()
    utils.enable_colours(True)

    # initialize current best network
    curr_best_network = TheseusNetwork()
    curr_best_network.load_state_dict(torch.load(best_model_path, weights_only=True))
    curr_best_network.eval()
    curr_best_network = curr_best_network.to(device)

    # initialize 
    network = new_model.eval()
    network = network.to(device)


    wins = 0

    loop = tqdm(total=n, position=0, leave=False)
    for i in range(n):
        players = [Theseus(colour, curr_best_network, device) for colour in all_player_colours]

        colour = random.choice(all_player_colours)

        players[all_player_colours.index(colour)] = Theseus(colour, network, device)

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1
            loop.set_description('Turn: {} Win rate: {}'.format(turns, wins/(i+1)))

        if lab.who_won().colour == colour:
            wins += 1 

        loop.update(1)

    if wins/n > .275:
        # replace curr_best_network on disk
        torch.save(network.state_dict(), best_model_path)
        