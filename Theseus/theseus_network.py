import torch.nn as nn

class TheseusNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.main_network = None

    self.slide_policy_network = None

    self.move_policy_network = None

    self.value_network = None
    

  def forward(self, x):
    new_x = self.main_network(x)
    p = self.slide_policy_network(new_x)
    m = self.move_policy_network(new_x)
    y = self.value_network(new_x)
    return p, m, y