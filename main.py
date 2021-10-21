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
    c1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    c2 = [21]
    fileName = 'filter_count_1'
    results = []
    
    for x in c2:
        for y in c1: 
            print("ilość filtrów: {:.4f}, {:.4f}".format(x, y))   
            model =  trainingModel(dataset=intelDataset(), method='xavier_uniform', input_size = input_size, c_kernels = [19, 7], in_channels = [3,x], out_channels =[x, y])
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
