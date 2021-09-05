import torch
from data import Data
from training import training
from intel_data import intel_data
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms




if __name__ == "__main__":
    sns.set()
    torch.cuda.empty_cache()

    (data, x, test) =  intel_data()

    # data = Data()
    # data = data
    # data.cuda()
    epoch = list(range(5))
    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal']
    # for method in methods:
    #     sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
    #     plt.plot(epoch, sse, label=method)

    sse, pk = training(dataset=data, test = test ,epoch=epoch, method='xavier_uniform')
    plt.figure()
    plt.plot(epoch, sse, label='xavier_uniform')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.show()
