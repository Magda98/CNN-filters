import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class cnnNet(nn.Module):
    def __init__(self):
        """
        CNN module
        """
        super().__init__()
        size_out = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,  kernel_size=11)
        
        size_out = math.floor((size_out +2*0 - 1*(11-1) -1)/1 +1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,  stride=2)
        size_out = math.floor((size_out +2*0 - 1*(2-1) -1)/2 +1)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        size_out = math.floor((size_out +2*0 - 1*(3-1) -1)/1 +1)

        self.pool2 = nn.MaxPool2d(kernel_size=2,  stride=2)
        size_out = math.floor((size_out +2*0 - 1*(2-1) -1)/2 +1)

        self.fc1 = nn.Linear(size_out*size_out* 16, 84)
        self.fc2 = nn.Linear(84, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
