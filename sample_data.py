

from typing import List
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np
import pathlib
import math


class SampleDataset():
    def __init__(self):

        self.last = False
        self.k = 0
        transform_viz_test = transforms.Compose([
            transforms.Resize((840, 840)),
            transforms.ToTensor(),
        ])

        self.test_viz_data = DataLoader(datasets.ImageFolder('./datasets/sample_test', transform=transform_viz_test))

        # load sample image
        self.sample = DataLoader(datasets.ImageFolder('./datasets/sample_test', transform=transform_viz_test))

        self.sample = self.test_viz_data

        # classes
        root = pathlib.Path('./datasets/sample_test/')
        self.classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(self.classes)
        self.trainloader = self.sample
        self.validloader = self.sample
        self.testloader = self.sample

    def get_chunks(self):
        if self.k == 10:
            self.last = True
        self.k += 1
