import torch
from data import Data
from training import trainingModel
from intel_data import intelDataset
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



import numpy as np


if __name__ == "__main__":
    sns.set()
    torch.cuda.empty_cache()

    # input image size in px (square image)
    input_size = 150

    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal', 'custom']
    # for method in methods:
    #     sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
    #     plt.plot(epoch, sse, label=method)
    # c1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    # c2 = [41]
    # fileName = 'filter_count_3'
    # results = []
    
 
    # for method in methods:
    model =  trainingModel(dataset=intelDataset(), method='xavier_uniform', input_size = input_size, c_kernels = [19, 7], in_channels = [3,8], out_channels =[8, 24])
    sse, pk, e = model.training()
    # plt.plot(range(e), sse, label=method)
        
            
    # np.savetxt("data_plots/" + fileName + ".csv", results, delimiter=";")
    

    e = list(range(len(sse)))
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
