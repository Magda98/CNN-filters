from typing import List
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
from torchvision import datasets
import torch
import numpy as np
import pathlib
import math


class WeaponData:
    def __init__(self):
        """
        Dataloader
        """

        # workers(processes) in loading data
        self.num_workers = 0

        # batch size
        self.batch_size = 20

        # * Horizontal flip - for augmentation

        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                # transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205)),
            ]
        )

        # Load data from folders
        self.data = datasets.ImageFolder(
            "./datasets/GunDetection/", transform=transform_train
        )

        self.data_test = datasets.ImageFolder(
            "./datasets/GunDetection/", transform=transform_test
        )

        n_train_c1 = int(0.8 * 9062)
        n_train_c2 = int(0.8 * 795)

        num_test = len(self.data_test)
        self.indices_test = list(range(num_test))

        num_train = len(self.data)
        self.indices = list(range(num_train))

        self.c1 = self.indices[:9061]
        self.c2 = self.indices[9061:]

        self.train_idx = self.c1[:n_train_c1]
        del self.c1[0:n_train_c1]
        self.train_idx.extend(self.c2[:n_train_c2])
        del self.c2[0:n_train_c2]

        self.test_idx = self.c1
        self.test_idx.extend(self.c2)

        # create samplers
        train_sampler = SubsetRandomSampler(self.train_idx)
        test_sampler = SubsetRandomSampler(self.test_idx)

        # dataloaders

        trainloader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
        )
        testloader = DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=test_sampler,
        )

        # classes
        root = pathlib.Path("./datasets/GunDetection")
        self.classes = sorted([j.name.split("/")[-1] for j in root.iterdir()])
        print(self.classes)

        self.trainloader = trainloader
        self.testloader = testloader
