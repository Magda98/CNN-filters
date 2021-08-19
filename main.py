import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn


def weights_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.bias)


if __name__ == "__main__":

    torch.cuda.empty_cache()

    dataset = Data()

    cnn_model = cnnNet()
    cnn_model.apply(weights_init)
    # cnn_model = cnn_model.cuda

    criterion = nn.CrossEntropyLoss()

    lr = 0.05

    # optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr,  momentum=0.9)

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
    for epoch in range(200):
        for (labels, x) in zip(dataset.labels, dataset.images):
            optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
            out = cnn_model(x)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print("learning rate:", optimizer.param_groups[0]['lr'])
        print('Epoch: {}.............'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(loss))
