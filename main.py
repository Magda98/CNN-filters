import torch
from data import Data
from cnn import cnnNet

if __name__ == "__main__":

    torch.cuda.empty_cache()

    dataset = Data()

    net = cnnNet()

    criterion = torch.nn.NLLLoss()

    lr = 0.05

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for x in dataset.images:
        optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
        out = net(x)

        loss = criterion(out, dataset.labels)
        loss.backward()
        optimizer.step()
