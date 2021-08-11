from torch import nn
import torch as torch
import numpy as np


class cnnNet(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, pool, features, classes=0, padding=[0, 0], dilation=[1, 1]):
        super(cnnNet, self).__init__()
        self.layers = []
        out_size = [int((features + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)]
        for index, (i, o, k, s, p, d) in enumerate(zip(input_ch, output_ch, kernel_size, stride, padding, dilation)):
            self.layers.append(nn.Conv1d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, dilation=d))
            self.layers.append(nn.MaxPool1d(pool[index]))
            if index >= 1:
                out_size.append(int(
                    (out_size[-1] + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1) / stride[
                        index] + 1))
            # wyjście "pooling layer" # domyślną wartością kroku jest wielkość filtru
            out_size.append(int((out_size[-1] + 2 * 0 - 1 * (pool[index] - 1) - 1) / pool[index] + 1))
        self.layers = nn.ModuleList(self.layers)

        # definiowanie warstw w pełni połączonych
        self.fc1 = nn.Linear(out_size[-1] * output_ch[-1], 25)
        # self.fc2 = nn.Linear(25, 1)
        self.fc2 = nn.Linear(25, 10)
        if classes == 0:
            self.fc3 = nn.Linear(10, 1)
        elif classes > 0:
            self.fc3 = nn.Linear(10, classes)
        self.classes = classes

    def forward(self, x, batch=False):
        # reshape data
        if not batch:
            out = x.view(1, 1, x.shape[0])
        else:
            out = x.view(x.shape[0], 1, x.shape[1])
        # warstawy konwolucyjne
        for layer in self.layers:
            out = layer(out)
        # przekształcenie wyjścia do odpowiedniego kształtu
        out = out.view(out.shape[0], out.shape[1] * out.shape[2])
        # warstawy w pełni połączone
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        # out = self.fc2(out)
        if self.classes == 0:
            out = self.fc3(out)
        elif self.classes > 0:
            m = torch.nn.LogSoftmax(dim=1)
            out = m(self.fc3(out))

        return out
