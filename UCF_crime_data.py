

from typing import List
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np
import pathlib
import math


class UCFCrimeDataset():
    def __init__(self):
        """
        Dataloader
        """

        # workers(processes) in loading data
        self.num_workers = 0

        # batch size
        self.batch_size = 100

        # k-fold validation (k=10)
        valid_size = 0.1

        # flag set if passed through all training set
        self.last = False
        self.last_temp = False

        # * Horizontal flip - for augmentation

        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

        # Load data from folders
        self.train_data = datasets.ImageFolder('./datasets/crime/train',
                                               transform=transform_train)
        self.test_data = datasets.ImageFolder('./datasets/crime/test',
                                              transform=transform_test)

        # training dataset
        self.num_train = len(self.train_data)
        self.indices = list(range(self.num_train))

        # create samplers
        train_sampler = SubsetRandomSampler(self.indices)

        # dataloaders

        trainloader = DataLoader(self.train_data, batch_size=self.batch_size,
                                 sampler=train_sampler,
                                 num_workers=self.num_workers)
        testloader = DataLoader(self.test_data, batch_size=self.batch_size,
                                num_workers=self.num_workers)

        # classes
        root = pathlib.Path('./datasets/crime/train')
        self.classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(self.classes)

        self.trainloader = trainloader
        self.testloader = testloader
