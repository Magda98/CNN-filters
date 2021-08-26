import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(conv1, conv2):
    # n = 1

    # for i,c in enumerate(conv1):
    #     for j,npimg in enumerate(c):
    #         plt.subplot(6, 6, n)
    #         plt.imshow(npimg)
    #         plt.grid(b=None)
    #         plt.axis('off')
    #         n+=1

    # plt.draw()
    # plt.pause(1e-17)
    # plt.clf()
    # plt.figure()
    n=1
    for i,c in enumerate(conv1):
        for j,npimg in enumerate(c):
            plt.subplot(5,5 , n)
            plt.imshow(npimg)
            plt.grid(b=None)
            plt.axis('off')
            n+=1

            
    plt.draw()
    plt.pause(1e-17)
    plt.clf()


def weights_init(m, method):
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.bias)

            if method == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif method == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)


def training(dataset, epoch, method):
    cnn_model = cnnNet()
    cnn_model.apply(lambda m: weights_init(m, method))

    criterion = nn.CrossEntropyLoss()

    lr = 0.0001
    er = 1.04
    lr_inc = 1.04
    lr_desc = 0.7
    old_sse = 0

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        cnn_model = cnn_model.cuda()
        dataset.images = dataset.images.cuda()
        dataset.labels = dataset.labels.cuda()
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr,  momentum=0.9)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)

    lsse = []
    for e in epoch:

        loss_array = []
        old_param = cnn_model.parameters
        for (labels, data) in zip(dataset.labels, dataset.images):
            optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
            out = cnn_model(data)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_array.append(loss.item())


        if e %10 == 0:
            imshow(cnn_model.conv1.weight.data.detach().cpu().numpy(), cnn_model.conv2.weight.data.detach().cpu().numpy())
        # %%
        # Adaptive learning rate
        sse = sum(loss_array)
        lsse.append(sse)
        lr = optimizer.param_groups[0]['lr']
        if sse > old_sse * er:
            # get old weights and bias
            cnn_model.parameters = old_param
            if lr >= 0.00001:
                lr = lr_desc * lr
        elif sse < old_sse:
            lr = lr_inc * lr
            if lr > 0.99:
                lr = 0.99
        optimizer.param_groups[0]['lr'] = lr
        old_sse = sse
        # %%

        print("learning rate:", optimizer.param_groups[0]['lr'])
        print('Epoch: {}.............'.format(e), end=' ')
        print("Loss: {:.4f}".format(loss))

    return lsse
