

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np

class intelDataset():
    def __init__(self):
        """
        Dataloader
        TODO: refactor, add cross-validation
        """
        
        # Number of subprocesses to use for data loading
        self.num_workers = 0

        # Number of samples per batch to load
        self.batch_size = 200

        # k-fold validation (k=10)
        valid_size = 0.1
        
        #flag set if passed through all training set
        self.last = False

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
        self.train_data = datasets.ImageFolder('./intel/seg_train',
                                        transform = transform_train)
        self.test_data = datasets.ImageFolder('./intel/seg_test',
                                        transform = transform_test)

        self.k = 1
        # Create validation set
        self.num_train = len(self.train_data)
        self.indices = list(range(self.num_train))
        np.random.shuffle(self.indices)
        self.split = int(np.floor(valid_size*self.num_train))
        self.train_idx, self.valid_idx = self.indices[self.split:], self.indices[:self.split]

        # Define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        # Create dataloaders
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = train_sampler,
                                                num_workers = self.num_workers)
        validloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = valid_sampler,
                                                num_workers = self.num_workers)
        testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size,
                                                num_workers = self.num_workers)

        # Get the classes
        import pathlib
        root = pathlib.Path('./intel/seg_train/')
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
    
    def getChunks(self):
        self.k +=1
        if(self.k < 10):
            # Create validation set
            self.train_idx, self.valid_idx = self.indices[:((self.k-1)*self.split)] + self.indices[(self.k*self.split):] , self.indices[((self.k-1)*self.split):(self.k*self.split)]
        else: 
            self.train_idx, self.valid_idx = self.indices[:((self.k-1)*self.split)], self.indices[((self.k-1)*self.split):]
            self.last = True

        # Define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        # Create dataloaders
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = train_sampler,
                                                num_workers = self.num_workers)
        validloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = valid_sampler,
                                                num_workers = self.num_workers)
        testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size,
                                                num_workers = self.num_workers)

        # Get the classes
        import pathlib
        root = pathlib.Path('./intel/seg_train/')
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        print(self.k)
        if self.last:
            self.k = 0
        return trainloader, validloader, testloader
    
    

