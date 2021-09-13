

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np


def intel_data():
    # Number of subprocesses to use for data loading
    num_workers = 0

    # Number of samples per batch to load
    batch_size = 100

    # Percentage of training set to use for validation
    valid_size = 0.2

    # Transform train and test data
    transform_train = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading train and test data
    train_data = datasets.ImageFolder('./intel/seg_train',
                                    transform = transform_train)
    test_data = datasets.ImageFolder('./intel/seg_test',
                                    transform = transform_test)

    # Create validation set
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                            sampler = train_sampler,
                                            num_workers = num_workers)
    validloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                            sampler = valid_sampler,
                                            num_workers = num_workers)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                            num_workers = num_workers)

    # Get the classes
    import pathlib
    root = pathlib.Path('./intel/seg_train/')
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    print(classes)
    return trainloader, validloader, testloader

