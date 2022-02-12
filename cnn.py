from typing import List
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class CnnNet(nn.Module):
    def __init__(self, input_size, output_size,  c_kernels=[7, 5], in_channels=[3, 6], out_channels=[6, 16], p_kernel=[2, 2], p_stride=[2, 2]):
        """
        CNN class
        * Architecture: Conv2d [5, 5] -> ReLu -> Conv2d [5, 5] -> ReLu Conv2d [5, 5] -> ReLu -> maxPool2d [2, 2] -> Conv2d [5, 5] -> ReLu -> Conv2d [5, 5] -> ReLu -> maxPool2d [2, 2] -> fc1 -> fc2 -> fc3 -> fc4
        @ input_size - size of input image
        @ c_kernels - size of kernel in each conv2d layer
        @ in_channels, out_channels - define channels in each conv2d layer
        @ p_kernel - size od kernel in each maxPool2d layer
        @ p_stride - size of strine in maxPool2d layer
        """
        super().__init__()
        self.cnn = nn.ModuleList()
        size_out = input_size
        for i, (k, c_in, c_out) in enumerate(zip(c_kernels, in_channels, out_channels)):
            self.cnn.append(nn.Conv2d(in_channels=c_in, out_channels=c_out,  kernel_size=k))
            self.cnn.append(nn.BatchNorm2d(c_out))
            size_out = math.floor((size_out + 2*0 - 1*(k-1) - 1)/1 + 1)
            if i % 2 == 0 and i > 0:
                self.cnn.append(nn.MaxPool2d(kernel_size=p_kernel[0],  stride=p_stride[0]))
                size_out = math.floor((size_out + 2*0 - 1*(p_kernel[0]-1) - 1)/p_stride[0] + 1)

        self.fc1 = nn.Linear(size_out*size_out * out_channels[-1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, output_size)
        print("ilość klas: {}".format(output_size))
        print("wielkość po warstawach conv: {}".format(size_out))

    def forward(self, inp: Tensor):  # type:ignore
        out = inp
        for l in self.cnn:
            if isinstance(l, nn.Conv2d):
                out = torch.Tanh(l(out))
            else:
                out = l(out)

        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        m = torch.nn.LogSoftmax(dim=1)
        out = m(self.fc4(out))
        return out
