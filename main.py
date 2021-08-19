import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn


if __name__ == "__main__":

    torch.cuda.empty_cache()

    dataset = Data()

    net = cnnNet()

    criterion = nn.CrossEntropyLoss()

    lr = 0.05

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(200):
        for (labels, x) in zip(dataset.labels, dataset.images):
            optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
            out = net(x)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print("learning rate:", optimizer.param_groups[0]['lr'])
        print('Epoch: {}.............'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(loss))
