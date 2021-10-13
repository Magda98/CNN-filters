

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np
import pathlib
import math

class intelDataset():
    def __init__(self):
        """
        Dataloader
        TODO: refactor, add cross-validation
        """
        
        # workers(processes) in loading data
        self.num_workers = 0

        # brach size
        self.batch_size = 200

        # k-fold validation (k=10)
        valid_size = 0.1
        
        #flag set if passed through all training set
        self.last = False

        # * Horizontal flip - for augmentation
        # ! Add normalization
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

        # Load data from folders
        self.train_data = datasets.ImageFolder('./intel/seg_train',
                                        transform = transform_train)
        self.test_data = datasets.ImageFolder('./intel/seg_test',
                                        transform = transform_test)


        self.k = 1
        # training dataset
        self.num_train = len(self.train_data)
        self.indices = list(range(self.num_train))
        self.buildings = self.indices[: 2190]
        self.forest = self.indices[2190: 4461]
        self.glacier = self.indices[4461: 6865]
        self.mountain = self.indices[6865: 9377]
        self.sea = self.indices[9377: 11651]
        self.street = self.indices[11651: ]
        
        temp_idx = []
        
        tmp_buildings = math.floor(0.1*len(self.buildings))
        tmp_forest = math.floor(0.1*len(self.forest))
        tmp_glacier = math.floor(0.1*len(self.glacier))
        tmp_mountain = math.floor(0.1*len(self.mountain))
        tmp_sea = math.floor(0.1*len(self.sea))
        tmp_street = math.floor(0.1*len(self.street))
        
        for x in range(9):
            temp_idx.extend(self.buildings[: tmp_buildings])
            temp_idx.extend(self.forest[:tmp_forest])
            temp_idx.extend(self.glacier[:tmp_glacier])
            temp_idx.extend(self.mountain[:tmp_mountain])
            temp_idx.extend(self.sea[:tmp_sea])
            temp_idx.extend(self.street[:tmp_street])
            
            del self.buildings[:tmp_buildings]
            del self.forest[:tmp_forest]
            del self.glacier[:tmp_glacier]
            del self.mountain[:tmp_mountain]
            del self.sea[:tmp_sea]
            del self.street[:tmp_street]
            
        temp_idx.extend(self.buildings[:])
        temp_idx.extend(self.forest[:])
        temp_idx.extend(self.glacier[:])
        temp_idx.extend(self.mountain[:])
        temp_idx.extend(self.sea[:])
        temp_idx.extend(self.street[:])
            
        self.indices = temp_idx
         
        self.split = int(np.floor(valid_size*self.num_train))
        self.train_idx, self.valid_idx = self.indices[self.split:], self.indices[:self.split]

        # create samplers
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        # dataloaders
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = train_sampler,
                                                num_workers = self.num_workers)
        validloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = valid_sampler,
                                                num_workers = self.num_workers)
        testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size,
                                                num_workers = self.num_workers)

        # classes
        root = pathlib.Path('./intel/seg_train/')
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
    
    def getChunks(self):
        self.k +=1
        if(self.k < 10):
            self.train_idx, self.valid_idx = self.indices[:((self.k-1)*self.split)] + self.indices[(self.k*self.split):] , self.indices[((self.k-1)*self.split):(self.k*self.split)]
        else: 
            self.train_idx, self.valid_idx = self.indices[:((self.k-1)*self.split)], self.indices[((self.k-1)*self.split):]
            self.last = True

        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = train_sampler,
                                                num_workers = self.num_workers)
        validloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size,
                                                sampler = valid_sampler,
                                                num_workers = self.num_workers)
        testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size,
                                                num_workers = self.num_workers)

        import pathlib
        root = pathlib.Path('./intel/seg_train/')
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)
        print(self.k)
        if self.last:
            self.k = 0
        return trainloader, validloader, testloader
    
    

