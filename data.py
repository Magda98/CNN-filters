import pickle
import torch


class Data:
    """ class for read dataset from file"""

    def __init__(self):
        with open('cifar-10/data_batch_1', 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
        self.labels = data_dict['labels'.encode()]
        self.filenames = data_dict['filenames'.encode()]
        self.images = data_dict['data'.encode()]
        self.images = self.images[:200].reshape(
            20, 10, 3, 32, 32).astype("uint8")
        self.labels = data_dict['labels'.encode()]
        self.labels = self.labels[:50]
        self.images = torch.tensor(self.images,  dtype=torch.float32)
        self.labels = torch.tensor(self.labels)
