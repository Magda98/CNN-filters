import torch
from data import Data
from training import trainingModel
from intel_data import intelDataset
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



import numpy as np


if __name__ == "__main__":
    sns.set()
    torch.cuda.empty_cache()

    
    # input image size in px (square image)
    input_size = 150

    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal']
    # for method in methods:
    #     sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
    #     plt.plot(epoch, sse, label=method)
    c = [3,5,7,9,11,13,15,17,19,21]
    fileName = 'pool_filter_size'
    results = []
    
    for x in c:
        for y in c: 
            print("Wielkość filtrów: {:.4f}, {:.4f}".format(x, y))   
            model =  trainingModel(dataset=intelDataset(), method='xavier_uniform', input_size = input_size, c_kernels = [x, y])
            sse, pk = model.training()
            temp = [x, y, np.average(pk)]
            results.append(temp)
            
    np.savetxt("data_plots/" + fileName + ".csv", results, delimiter=";")
    
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
