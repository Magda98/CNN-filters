import torch
from data import Data
from training import training


if __name__ == "__main__":

    torch.cuda.empty_cache()

    data = Data()

    training(dataset=data)
