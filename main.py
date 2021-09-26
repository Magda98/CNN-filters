import torch
from data import Data
from training import trainingModel
from intel_data import intelDataset
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms




if __name__ == "__main__":
    sns.set()
    torch.cuda.empty_cache()

    data =  intelDataset()
    
    # input image size in px (square image)
    input_size = 150

    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal']
    # for method in methods:
    #     sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
    #     plt.plot(epoch, sse, label=method)

    model =  trainingModel(dataset=data, epoch=10, method='xavier_uniform', input_size = input_size)
    sse, pk = model.training()
    plt.figure()
    e = list(range(len(sse)))
    plt.plot(e, sse, label='xavier_uniform')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.show()
    
    e = list(range(len(pk)))  
    plt.figure()
    plt.plot(e, pk, label='pk')
    plt.xlabel("Epoch")
    plt.ylabel("PK")
    plt.legend(loc='upper left')
    plt.show()
