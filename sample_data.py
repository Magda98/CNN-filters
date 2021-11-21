

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

        transform_viz_test = transforms.Compose([
            transforms.Resize((840, 840)),
            transforms.ToTensor(),
        ])

        self.test_viz_data = DataLoader(datasets.ImageFolder('./intel/sample_test', transform=transform_viz_test))

        # load sample image
        self.sample = DataLoader(datasets.ImageFolder('./intel/sample', transform=transform_viz_test))

        self.sample = self.test_viz_data

        # classes
        root = pathlib.Path('./intel/sample_test/')
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        self.trainloader = self.sample
        self.validloader = self.sample
        self.testloader = self.sample

        def get_chunks():
            print("no cv")
