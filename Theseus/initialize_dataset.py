import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from Labyrinth.tile import TileMovement
from self_play import MAX_DATA_SIZE, TensorLog
from theseus_trainer import DATA_FILE_PATH
from Labyrinth import utils
from Labyrinth.labyrinth import Labyrinth, RuleSet
from Labyrinth.player import Player, PlayerMovement, all_player_colours
from Labyrinth.theseus import convert_gameboard_to_tensor

class SpectatedPlayer(Player):
    def __init__(self, colour, data_bank):
        super().__init__(colour)
        self.data_bank = data_bank

    def decide_move(self, gameboard):
        best_slide, best_orientation, best_path = super().decide_move(gameboard)

        x = convert_gameboard_to_tensor(gameboard, self.cards, self.colour)

        p = torch.zeros(48)
        actions = TileMovement.all_moves()
        orientations = [0, 90, 180, 270]
        p[actions.index(best_slide) + (orientations.index(best_orientation) * 12)] = 1

        m = torch.zeros(49)
        pos_x = self.x
        pos_y = self.y
        for step in best_path:
            if step == PlayerMovement.UP:
                pos_y -= 1
            elif step == PlayerMovement.DOWN:
                pos_y += 1
            elif step == PlayerMovement.LEFT:
                pos_x -= 1
            else:
                pos_x += 1
        pos_x = min(pos_x, 6)
        pos_y = min(pos_y, 6)
        m[pos_x + (7 * pos_y)] = 1

        self.data_bank(tuple([x, p, m]))

        return best_slide, best_orientation, best_path
    
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

    # network = DummyNet()
    
    loop = tqdm(total=5000, position=0, leave=False)
    for n in range(5000):
        
        game_data = TensorLog()

        # players = [Theseus(colour, network, exploration_weight=1, data_bank=game_data[colour]) for colour in all_player_colours]
        players = [SpectatedPlayer(colour, game_data.get_log_function(colour)) for colour in all_player_colours]

        lab = Labyrinth(ruleset, players)

        turns = 0
        while lab.who_won() is None:
            lab.make_turn()
            turns += 1

        for colour, results in game_data.get_all_entries().items():
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