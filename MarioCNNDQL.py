import torch
import torch.nn as nn


class MarioCNNDQL(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU() ,# not sure why 512
            nn.Linear(512,n_actions)
        )

    def forward(self, x):
        return  self.linear(self.conv(x))

