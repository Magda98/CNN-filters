import numpy as np
import torch
from cnn import CnnNet
from cnnCifar import CnnNetC
import torch.nn as nn
from pytorch_model_summary import summary
from utils import weights_init, adaptive_leraning_rate, test


class TrainModel:
    """
    Main class for training a model for Intel Dataset and Cifar-10
    @ dataset - dataset loader object
    @ epoch - iterator object
    @ method - methos for weights initialization
    @ input_size - size of image
    """

    def __init__(
        self,
        dataset,
        method,
        input_size,
        c_kernels=[7, 5],
        out_channels=[30, 16],
        in_channels=[3, 30],
        p_kernel=[2, 2],
        p_stride=[2, 2],
        apt=0,
        dataset_name="intel",
        epoch=200,
        padding_flag=True,
        maxpool_freq=2,
        activation_relu=True,
        fc_size=4,
    ):
        print("class num: {}".format(len(dataset.classes)))
        self.dataset = dataset

        if dataset_name == "intel":
            self.cnn_model = CnnNet(
                input_size,
                len(dataset.classes),
                c_kernels=c_kernels,
                out_channels=out_channels,
                in_channels=in_channels,
                p_kernel=p_kernel,
                p_stride=p_stride,
                padding_flag=padding_flag,
                maxpool_freq=maxpool_freq,
                activation_relu=activation_relu,
                fc_size=4,
            )

        elif dataset_name == "cifar":
            self.cnn_model = CnnNetC(
                input_size,
                len(dataset.classes),
                c_kernels=c_kernels,
                out_channels=out_channels,
                in_channels=in_channels,
                p_kernel=p_kernel,
                activation_relu=activation_relu,
                p_stride=p_stride,
            )

        # print model summary
        print(
            summary(
                self.cnn_model,
                torch.zeros((1, 3, input_size, input_size)),
                show_input=False,
            )
        )

        self.epoch = epoch

        # weight initialization
        self.cnn_model.apply(lambda m: weights_init(m, method))
        self.method = method
        self.apt = apt
        self.dataset_name = dataset_name

        self.criterion = nn.NLLLoss()
        self.lr = 0.00001
        # self.lr = 0.00001
        if torch.cuda.is_available():
            self.cnn_model = self.cnn_model.cuda()
            self.cnn_model.cnn = self.cnn_model.cnn.cuda()
            print("GPU is available")
        else:
            torch.device("cpu")
            print("GPU not available, CPU used")

        self.optimizer = torch.optim.SGD(
            self.cnn_model.parameters(), lr=self.lr, momentum=0.9
        )
        # self.optimizer = torch.optim.AdamW(self.cnn_model.parameters(), lr=self.lr)

    def training(self):
        loss_train = []
        loss_test = []
        old_loss = 0
        for e in range(self.epoch):
            self.old_param = self.cnn_model.parameters
            loss_train_temp = 0
            # pass through all data
            for data, exp in self.dataset.trainloader:
                exp = exp.cuda()
                # pass data to cuda
                data = data.cuda()
                # Wyczyszczenie gradientów z poprzedniej epoki
                self.optimizer.zero_grad()
                out = self.cnn_model(data)
                loss = self.criterion(out, exp)
                loss.backward()
                self.optimizer.step()
                loss_train_temp += loss.item()

            loss_train.append(loss_train_temp)

            self.cnn_model, self.optimizer = adaptive_leraning_rate(
                loss_train[-1], self.optimizer, self.cnn_model, self.old_param, old_loss
            )
            old_loss = loss_train[-1]

            # test
            acc, loss_test_temp = test(self.dataset, self.cnn_model, self.criterion)
            loss_test.append(loss_test_temp)

            print("pk: {:.2f} %".format(acc))
            print("Learning rate: {:.10f}".format(self.optimizer.param_groups[0]["lr"]))
            print("Epoch: {}.............".format(e), end=" ")
            print("Loss: {:.4f}".format(loss_train_temp))

        return (loss_train, loss_test, acc, self.epoch)
