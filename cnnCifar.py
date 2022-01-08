from typing import List
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class CnnNetC(nn.Module):
    def __init__(self, input_size: int, output_size: int,  c_kernels: List[int] = [7, 5], in_channels: List[int] = [3, 6], out_channels: List[int] = [6, 16], p_kernel: List[int] = [2, 2], p_stride: List[int] = [2, 2]):
        """
        CNN class
        * Architecture: Conv2d [3, 3] -> ReLu -> maxPool2d [2, 2] -> Conv2d [3, 3] -> ReLu -> fc1 -> fc2 -> fc3
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
            size_out = math.floor((size_out + 2*0 - 1*(k-1) - 1)/1 + 1)
            if i % 3 == 0 and i >= 0:
                self.cnn.append(nn.MaxPool2d(kernel_size=p_kernel[0],  stride=p_stride[0]))
                size_out = math.floor((size_out + 2*0 - 1*(p_kernel[0]-1) - 1)/p_stride[0] + 1)

        self.fc1 = nn.Linear(size_out*size_out * out_channels[-1], 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
        print("ilość klas: {}".format(output_size))
        print("wielkość po warstawach conv: {}".format(size_out))

    def forward(self, inp: Tensor):  # type:ignore
        out = inp
        for i, l in enumerate(self.cnn):
            if isinstance(l, nn.Conv2d):
                out = F.relu(l(out))
                if i == 0:
                    sample1 = out
                elif i == 2:
                    sample2 = out
            elif isinstance(l, nn.MaxPool2d):
                out = l(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        m = torch.nn.LogSoftmax(dim=1)
        out = m(self.fc3(out))
        return out, (sample1, sample2)
