import torch
import torch.nn as nn

INPUT_CHANNELS = 12
HIDDEN_CHANNELS = 64
VALUE_CHANNELS = 1
MOVE_POLICY_CHANNELS = 49
SLIDE_POLICY_CHANNELS = 48
RESIDUAL_BLOCK_COUNT = 4

class ResidualBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class MainResNet(nn.Module):
    def __init__(self, layers_numbers, in_channels, out_channels):
        super(MainResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Stacked residual blocks
        layers = []
        for n in range(layers_numbers):
            layers.append(ResidualBlock(out_channels))

        self.residual_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.residual_blocks(x)
        
        return x
    

class ValueNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_channels=1, intermediate_channels=64):
        super(ValueNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=1, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(conv_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(81, intermediate_channels)
        self.fc2 = nn.Linear(intermediate_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class PolicyNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_channels=2):
        super(PolicyNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=1, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(conv_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(162, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x
    

class TheseusNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_network = MainResNet(RESIDUAL_BLOCK_COUNT, INPUT_CHANNELS, HIDDEN_CHANNELS)

        self.slide_policy_network = PolicyNet(HIDDEN_CHANNELS, SLIDE_POLICY_CHANNELS)

        self.move_policy_network = PolicyNet(HIDDEN_CHANNELS, MOVE_POLICY_CHANNELS)

        self.value_network = ValueNet(HIDDEN_CHANNELS, VALUE_CHANNELS)
        

    def forward(self, x, slide=True, move=True, value=True):
        new_x = self.main_network(x)
        p = None
        m = None
        y = None
        if slide:
            p = self.slide_policy_network(new_x)
        if move:
            m = self.move_policy_network(new_x)
        if value:
            y = self.value_network(new_x)
        return p, m, y