import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class cnnNet(nn.Module):
    def __init__(self, input_size, c_kernels = [7, 5], in_channels = [3,6], out_channels =[6, 16], p_kernel=[2,2], p_stride = [2,2]):
        """
        CNN class
        * Architecture: Conv2d -> ReLu -> maxPool2d -> Conv2d -> ReLu -> maxPool2d -> fc1 -> fc2 -> fc3
        @ input_size - size of input image
        @ c_kernels - size of kernel in each conv2d layer
        @ in_channels, out_channels - define channels in each conv2d layer
        @ p_kernel - size od kernel in each maxPool2d layer
        @ p_stride - size of strine in maxPool2d layer
        TODO: better way to get sample in forward function
        """
        super().__init__()
        self.cnn = nn.ModuleList()
        size_out = input_size

        for k,c_in,c_out,p,s in zip(c_kernels, in_channels, out_channels, p_kernel, p_stride):
            self.cnn.append(nn.Conv2d(in_channels=c_in, out_channels=c_out,  kernel_size=k))
            size_out = math.floor((size_out +2*0 - 1*(k-1) -1)/1 +1)
            self.cnn.append(nn.MaxPool2d(kernel_size=p,  stride=s))
            size_out = math.floor((size_out +2*0 - 1*(p-1) -1)/s +1)

        self.fc1 = nn.Linear(size_out*size_out* out_channels[1], 84)
        self.fc2 = nn.Linear(84, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, inp):
        out = self.cnn[1](F.relu(self.cnn[0](inp)))
        sample1 = out

        out = self.cnn[3](F.relu(self.cnn[2](out)))
        sample2 = out

        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        m = torch.nn.LogSoftmax(dim=1)
        out = m(self.fc3(out))
        return out, (sample1, sample2)
