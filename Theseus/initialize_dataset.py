import os
import pickle
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

module_path = os.path.abspath("/Users/matt/Documents/LabrynthGo/theseus-larbryinth-rl/Labyrinth")
if module_path not in sys.path:
    sys.path.append(module_path)

from self_play import MAX_DATA_SIZE
from theseus_trainer import DATA_FILE_PATH
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.player import all_player_colours
from Labyrinth.theseus import Theseus


class DummyNet(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, slide=True, move=True, value=True):
    p = torch.rand(48)
    m = torch.rand(49)
    y = torch.rand(1)
    return p, m, y
  
def main():
    ruleset = RuleSet()
    utils.enable_colours(True)

    data = []

    network = DummyNet()
    
    loop = tqdm(total=50, position=0, leave=False)
    for n in range(50):
        
        game_data = {color: [] for color in all_player_colours}

        players = [Theseus(colour, network, exploration_weight=1, data_bank=game_data[colour]) for colour in all_player_colours]

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1
            print(turns)

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
    with open(DATA_FILE_PATH, 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    main()