import os
import pickle
import torch
from tqdm import tqdm
from theseus_network import TheseusNetwork
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class TheseusLoss(nn.Module):
    def __init__(self, weight_regularization=1e-4):
        super(TheseusLoss, self).__init__()
        self.weight_regularization = weight_regularization
    
    def forward(self, slide_pred, slide_target, move_pred, move_target, value_pred, value_target, model):
        # Policy Loss (Cross-Entropy)
        slide_loss = -torch.sum(slide_target * F.log_softmax(slide_pred, dim=-1)) / slide_target.size(0)
        move_loss = -torch.sum(move_target * F.log_softmax(move_pred, dim=-1)) / move_target.size(0)

        # Value Loss (Mean Squared Error)
        value_loss = torch.mean((value_pred - value_target) ** 2)

        # Regularization Loss (L2 norm of weights)
        reg_loss = sum(torch.sum(param ** 2) for param in model.parameters())
        reg_loss /= sum(param.numel() for param in model.parameters())  # Normalize by total parameters
        reg_loss *= self.weight_regularization

        # Total Loss
        total_loss = slide_loss + move_loss + value_loss + reg_loss
        return total_loss
    

class RLDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file)

    def __getitem__(self, i):
        x, p, m, y = self.data[i]
        return x, p, m, y

    def __len__(self):
        return len(self.data)
    

def optimize(device, data_file_path, latest_model_path, n_iterations=10000, lr=1e-3, batch_size = 64):

    # Init networks, load latest version
    network = TheseusNetwork()
    if latest_model_path:
        network.load_state_dict(torch.load(latest_model_path, weights_only=True))
    network = network.to(device)

    # Init optimizer
    optim = torch.optim.Adam(network.parameters(), lr=lr)

    # Prepare dataset
    dataset = RLDataset(data_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # loss function
    loss_fn = TheseusLoss()

    # Start main loop
    loop = tqdm(total=n_iterations, position=0, leave=False)
    for _ in range(n_iterations):
        x, p, m, y = next(iter(loader))
        x, p, m, y = x.to(device), p.to(device), m.to(device), y.to(device)
        optim.zero_grad()

        p_logits, m_logits, y_logits = network(x)
        loss = loss_fn(p_logits, p, m_logits, m, y_logits, y, network)

        if torch.isnan(loss):
                print("NaN detected in loss! Stopping training.")
        
        loss.backward()
        optim.step()

        # Print results
        loop.update(1)
        loop.set_description("Loss: {}".format(loss.item()))

    return network