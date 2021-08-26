import torch
from data import Data
from training import training
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms




if __name__ == "__main__":
    sns.set()
    torch.cuda.empty_cache()

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset=torchvision.datasets.ImageNet(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    data = Data()
    epoch = list(range(200))
    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal']
    for method in methods:
        sse = training(dataset=data, epoch=epoch, method=method)
        plt.plot(epoch, sse, label=method)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.show()
