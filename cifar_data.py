

from typing import List
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np
import pathlib
import math


class CifarDataset():
    def __init__(self):
        """
        Dataloader
        TODO: refactor, add cross-validation
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
            transforms.Resize((150, 150)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

        # Load data from folders
        self.train_data = datasets.ImageFolder('./datasets/cifar10/train',
                                               transform=transform_train)
        self.test_data = datasets.ImageFolder('./datasets/cifar10/test',
                                              transform=transform_test)

        self.sample = DataLoader(datasets.ImageFolder('./datasets/cifar10/sample', transform=transform_test))

        self.k = 1
        # training dataset
        self.num_train = len(self.train_data)
        self.indices = list(range(self.num_train))

        # training dataset
        self.airplane = self.indices[: 4999]
        self.automobile = self.indices[4999: 9999]
        self.bird = self.indices[9999: 14999]
        self.cat = self.indices[14999: 19999]
        self.deer = self.indices[19999: 24999]
        self.dog = self.indices[24999: 29999]
        self.frog = self.indices[29999: 34999]
        self.horse = self.indices[34999: 39999]
        self.ship = self.indices[39999: 44999]
        self.truck = self.indices[44999:]

        temp_idx: List[int] = []

        tmp_airplane: int = math.floor(0.1*len(self.airplane))
        tmp_automobile: int = math.floor(0.1*len(self.automobile))
        tmp_bird: int = math.floor(0.1*len(self.bird))
        tmp_cat: int = math.floor(0.1*len(self.cat))
        tmp_deer: int = math.floor(0.1*len(self.deer))
        tmp_dog: int = math.floor(0.1*len(self.dog))
        tmp_frog: int = math.floor(0.1*len(self.frog))
        tmp_horse: int = math.floor(0.1*len(self.horse))
        tmp_ship: int = math.floor(0.1*len(self.ship))
        tmp_truck: int = math.floor(0.1*len(self.truck))

        for _ in range(9):
            temp_idx.extend(self.airplane[: tmp_airplane])
            temp_idx.extend(self.automobile[:tmp_automobile])
            temp_idx.extend(self.bird[:tmp_bird])
            temp_idx.extend(self.cat[:tmp_cat])
            temp_idx.extend(self.deer[:tmp_deer])
            temp_idx.extend(self.dog[:tmp_dog])
            temp_idx.extend(self.frog[:tmp_frog])
            temp_idx.extend(self.horse[:tmp_horse])
            temp_idx.extend(self.ship[:tmp_ship])
            temp_idx.extend(self.truck[:tmp_truck])

            del self.airplane[:tmp_airplane]
            del self.automobile[:tmp_automobile]
            del self.bird[:tmp_bird]
            del self.cat[:tmp_cat]
            del self.deer[:tmp_deer]
            del self.dog[:tmp_dog]
            del self.frog[:tmp_frog]
            del self.horse[:tmp_horse]
            del self.ship[:tmp_ship]
            del self.truck[:tmp_truck]

        temp_idx.extend(self.airplane[:])
        temp_idx.extend(self.automobile[:])
        temp_idx.extend(self.bird[:])
        temp_idx.extend(self.cat[:])
        temp_idx.extend(self.deer[:])
        temp_idx.extend(self.dog[:])
        temp_idx.extend(self.frog[:])
        temp_idx.extend(self.horse[:])
        temp_idx.extend(self.ship[:])
        temp_idx.extend(self.truck[:])

        self.indices = temp_idx

        self.split = int(np.floor(valid_size*self.num_train))
        self.train_idx, self.valid_idx = self.indices[self.split:], self.indices[:self.split]

        # create samplers
        train_samplerCV = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)
        train_sampler = SubsetRandomSampler(self.indices)

        # dataloaders
        trainloaderCV = DataLoader(self.train_data, batch_size=self.batch_size,  # type:ignore
                                   sampler=train_samplerCV,
                                   num_workers=self.num_workers)
        validloader = DataLoader(self.train_data, batch_size=self.batch_size,  # type:ignore
                                 sampler=valid_sampler,
                                 num_workers=self.num_workers)

        trainloader = DataLoader(self.train_data, batch_size=self.batch_size,  # type:ignore
                                 sampler=train_sampler,
                                 num_workers=self.num_workers)
        testloader = DataLoader(self.test_data, batch_size=self.batch_size,  # type:ignore
                                num_workers=self.num_workers)

        # classes
        root = pathlib.Path('./datasets/cifar10/train/')
        self.classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(self.classes)
        self.trainloaderCV = trainloaderCV
        self.validloader = validloader

        self.trainloader = trainloader
        self.testloader = testloader

    def get_chunks(self):
        self.k += 1
        if self.last_temp:
            self.k = 0
            self.last = True
        if(self.k < 10):
            self.train_idx, self.valid_idx = self.indices[:(
                (self.k-1)*self.split)] + self.indices[(self.k*self.split):], self.indices[((self.k-1)*self.split):(self.k*self.split)]
        else:
            self.train_idx, self.valid_idx = self.indices[:(
                (self.k-1)*self.split)], self.indices[((self.k-1)*self.split):]
            self.last_temp = True

        train_samplerCV = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        trainloaderCV = DataLoader(self.train_data, batch_size=self.batch_size,
                                   sampler=train_samplerCV,
                                   num_workers=self.num_workers)
        validloader = DataLoader(self.train_data, batch_size=self.batch_size,  # type:ignore
                                 sampler=valid_sampler,
                                 num_workers=self.num_workers)

        print(self.k)

        self.trainloaderCV = trainloaderCV
        self.validloader = validloader
