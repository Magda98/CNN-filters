import torch
from data import Data
from cnn import cnnNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class trainingModel():
    """
    Main class of training a model
    @ dataset - dataset loader object
    @ epoch - iterator object
    @ method - methos for weights initialization
    @ test - data for testing
    @ input_size - size of image
    TODO: refactor code
    """
    def __init__(self,dataset, method, input_size, c_kernels = [7, 5], out_channels = [30, 16], in_channels = [3,30], p_kernel=[2,2], p_stride = [2,2]):
        
        self.dataset = dataset
        self.cnn_model = cnnNet(input_size, c_kernels = c_kernels, out_channels = out_channels, in_channels = in_channels, p_kernel=p_kernel, p_stride = p_stride)
        # weight initialization
        self.cnn_model.apply(lambda m: self.weights_init(m, method))

        self.criterion = nn.NLLLoss()
        self.lr = 0.0001
        self.er = 1.04
        self.lr_inc = 1.04
        self.lr_desc = 0.7
        self.old_sse = 0

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.cnn_model = self.cnn_model.cuda()
            self.cnn_model.cnn = self.cnn_model.cnn.cuda()
            # dataset.images = dataset.images.cuda()
            # dataset.labels = dataset.labels.cuda()
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        # optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr,  momentum=0.9)
        self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.lr)

    
         
    def valid_classification(self, out, d):
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

    def imshow(self, conv1, conv2, features_map, image):
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


    def weights_init(self, m, method):
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
                elif method == 'custom':
                    m.weight.data = torch.zeros(m.weight.data.size())

    def adaptive_leraning_rate(self):
            self.sse = sum(self.loss_array)
            self.sse_array.append(self.sse)
            lr = self.optimizer.param_groups[0]['lr']
            if self.sse > self.old_sse * self.er:
                # get old weights and bias
                self.cnn_model.parameters = self.old_param
                if lr >= 0.00001:
                    lr = self.lr_desc * lr
            elif self.sse < self.old_sse:
                lr = self.lr_inc * lr
                lr = min([lr, 0.99])
            self.optimizer.param_groups[0]['lr'] = lr
            self.old_sse = self.sse

    def test(self):
        loss_t = 0
        pk=[]
        with torch.no_grad():
            for data, labels in self.dataset.validloader:
                labels = labels.cuda()
                data = data.cuda()
                out, sample = self.cnn_model(data)
                output = torch.argmax(out, dim=1)
                loss = self.criterion(out, labels)
                loss_t+= loss.cpu().item()
                pk.append(self.valid_classification(output, labels))


        self.loss_test.append(loss_t)
        # image = data[0].detach().cpu().numpy()
        pk = np.average(pk)
        return pk
        
    def training(self):
        e = 0
        pk_test = []
        self.loss_test = []
        run = True
        pk_flag = True
        self.sse_array = []
        
        while run:
            epoch_per_k = 0
            while pk_flag:
                self.loss_array = []
                self.old_param = self.cnn_model.parameters
                
                # pass through all data 
                for data, exp in self.dataset.trainloader:
                    exp = exp.cuda()
                    #pass data to cuda
                    data = data.cuda()
                    # Wyczyszczenie gradientów z poprzedniej epoki
                    self.optimizer.zero_grad()  
                    out, sample = self.cnn_model(data)
                    loss = self.criterion(out, exp)
                    loss.backward()
                    self.optimizer.step()
                    self.loss_array.append(loss.item())
                    
                #increment epoch
                e+=1
                epoch_per_k+=1
                
                # Test
                pk = self.test()
                pk_test.append(pk)
                
                # imshow(cnn_model.cnn[0].weight.data.detach().cpu().numpy(), cnn_model.cnn[2].weight.data.detach().cpu().numpy(), sample.detach().cpu().numpy(), image)
                
                self.adaptive_leraning_rate()    
                
                if epoch_per_k >= 10 or pk > 80:
                    pk_flag = False
                print("pk: {:.2f} %".format(pk))
                print("Learning rate: {:.5f}".format(self.optimizer.param_groups[0]['lr']))
                print('Epoch: {}.............'.format(e), end=' ')
                print("Loss: {:.4f}".format(loss))
                
            self.dataset.getChunks()
            pk_flag = True
            if self.dataset.last:
                run = False
        print(np.average(pk_test))
        return self.sse_array, pk_test
