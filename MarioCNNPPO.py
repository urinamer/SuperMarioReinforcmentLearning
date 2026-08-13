import torch
import torch.nn as nn
import torch.nn.functional as F


class MarioCNNPPO(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
           nn.Conv2d(4,32,kernel_size=8,stride=4),nn.ReLU(),
           nn.Conv2d(32,64,kernel_size=4,stride = 2),nn.ReLU(),
           nn.Conv2d(64,64,kernel_size= 3,stride=1),nn.ReLU(),
           nn.Flatten(),
       )
        self.linear = nn.Sequential(
            nn.Linear(3136,512),nn.Tanh() #not sure why 512
        )
        self.value = nn.Linear(512,1)
        self.dist = nn.Linear(512,2)



    def forward(self,x):
        x = self.linear(self.conv(x))
        return self.value(x),self.dist(x)


