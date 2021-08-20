import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


def weights_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.normal_(m.bias)


def training(dataset):

    cnn_model = cnnNet()
    cnn_model.apply(weights_init)

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
    for epoch in range(200):

        loss_array = []
        old_param = cnn_model.parameters
        for (labels, data) in zip(dataset.labels, dataset.images):
            optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
            out = cnn_model(data)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_array.append(loss.item())

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
        print('Epoch: {}.............'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(loss))
