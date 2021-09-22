import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np



def valid_classification(out, d):
    """
    Function calculating valid classification
    @ out - netowerk output
    @ d - destination value
    return: classification correctness in %
    """
    out = out.cpu().detach().numpy()
    d = d.cpu().detach().numpy()
    x = abs(d - out)
    valid = sum(i < 0.5 for i in x)
    return valid / x.shape[0] * 100

def imshow(conv1, conv2, features_map, image):
    """
    Function plotting images
    @ conv1, conv2 - filters from each conv layer
    @ features_map - each images after passing it through layer
    @ image - image given at input 
    """
    plt.subplot(5,6, 3)
    image =   np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.grid(b=None) 

    n = 7
    for f in conv1:
        for c in f:
            plt.subplot(5,6,n)
            n+=1
            npimg =  c
            plt.imshow(npimg, cmap="gray")
            plt.axis('off')
            plt.grid(b=None)    

    n = 25
    for f in features_map[0]:
        plt.subplot(5,6,n)
        n+=1
        npimg = f
        plt.imshow(npimg, cmap="gray")
        plt.axis('off')
        plt.grid(b=None)       
    plt.draw()
    plt.pause(1e-17)
    plt.clf()


def weights_init(m, method):
    """
    Function for filters initialization
    * function uses method from PyTorch to initialize weights
    @ m - model
    @ method - method to initialize weights
    TODO: add custom method to initialize weights
    """
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


def training(dataset, epoch, method, input_size):
    """
    Main function of training a model
    @ dataset - dataset loader object
    @ epoch - iterator object
    @ method - methos for weights initialization
    @ test - data for testing
    @ input_size - size of image
    TODO: refactor code, move ALR to function
    """
    cnn_model = cnnNet(input_size, c_kernels = [7, 5], out_channels =[30, 16], in_channels = [3,30], p_kernel=[2,2], p_stride = [2,2])
    # weight initialization
    cnn_model.apply(lambda m: weights_init(m, method))

    criterion = nn.NLLLoss()
    lr = 0.0001
    er = 1.04
    lr_inc = 1.04
    lr_desc = 0.7
    old_sse = 0

    pk_test = []
    loss_test = []

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        cnn_model = cnn_model.cuda()
        cnn_model.cnn = cnn_model.cnn.cuda()
        # dataset.images = dataset.images.cuda()
        # dataset.labels = dataset.labels.cuda()
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr,  momentum=0.9)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    pk_flag = True
    sse_array = []
    for e in epoch:
        while pk_flag:
            loss_array = []
            old_param = cnn_model.parameters
            # for (labels, data) in zip(dataset.labels, dataset.images):
            for data, labels in dataset.trainloader:
                labels = labels.cuda()
                data = data.cuda()
                optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki
                out, sample = cnn_model(data)

                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                loss_array.append(loss.item())


            # %%
            # Test
            
            loss_t = 0
            pk=[]
            with torch.no_grad():
                for data, labels in dataset.validloader:
                    labels = labels.cuda()
                    data = data.cuda()
                    out, sample = cnn_model(data)
                    output = torch.argmax(out, dim=1)
                    loss = criterion(out, labels)
                    loss_t+= loss.cpu().item()
                    pk.append(valid_classification(output, labels))


            loss_test.append(loss_t)
            # image = data[0].detach().cpu().numpy()
            pk = np.average(pk)
            pk_test.append(pk)
            print("pk: {} %".format(pk))
                # imshow(cnn_model.cnn[0].weight.data.detach().cpu().numpy(), cnn_model.cnn[2].weight.data.detach().cpu().numpy(), sample.detach().cpu().numpy(), image)
            
            if pk > 80:
                pk_flag = False
            # %%

            # %%
            # Adaptive learning rate
            sse = sum(loss_array)
            sse_array.append(sse)
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
            
        dataset.getChunks()
        pk_flag = True

    print(np.average(pk_test))
    return sse_array, pk_test
