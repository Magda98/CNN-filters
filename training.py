import cv2
from random import randint
from random import seed
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.typing as npt
from torch.functional import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_
from torch.nn.modules.module import Module
from cnn import CnnNet
import torch.nn as nn
import matplotlib
from intel_data import IntelDataset
from torchvision.utils import make_grid
import math
# matplotlib.use('Agg')


class trainingModel():
    """
    Main class of training a model
    @ dataset - dataset loader object
    @ epoch - iterator object
    @ method - methos for weights initialization
    @ test - data for testing
    @ input_size - size of image
    """

    def __init__(self, dataset, method: str, input_size: int, c_kernels: List[int] = [7, 5], out_channels: List[int] = [30, 16], in_channels: List[int] = [3, 30], p_kernel: List[int] = [2, 2], p_stride: List[int] = [2, 2], apt=0):
        seed(1)
        print("class num: {}".format(len(dataset.classes)))
        self.dataset = dataset
        self.cnn_model = CnnNet(input_size, len(dataset.classes),  c_kernels=c_kernels, out_channels=out_channels,
                                in_channels=in_channels, p_kernel=p_kernel, p_stride=p_stride)

        # weight initialization
        self.cnn_model.apply(lambda m: self.weights_init(m, method))
        self.method = method
        self.apt = apt

        self.criterion = nn.NLLLoss()
        self.lr = 0.00001
        self.er = 1.04
        self.lr_inc = 1.04
        self.lr_desc = 0.7
        self.old_sse = 0.0

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.cnn_model = self.cnn_model.cuda()
            self.cnn_model.cnn = self.cnn_model.cnn.cuda()
            print("GPU is available")
        else:
            torch.device("cpu")
            print("GPU not available, CPU used")

        # optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr,  momentum=0.9)
        self.optimizer = torch.optim.AdamW(self.cnn_model.parameters(), lr=self.lr)

    def valid_classification(self, out: Tensor, d: Tensor) -> float:
        """
        Function calculating valid classification
        @ out - netowerk output
        @ d - destination value
        return: classification correctness in %
        """
        out = out.cpu().detach().numpy()
        d = d.cpu().detach().numpy()
        temp: List[float] = abs(d - out)
        valid = sum(i < 0.5 for i in temp)
        return valid / temp.shape[0] * 100  # type:ignore

    def imshow(self, conv1: Tensor, conv2: Tensor, features_map1: Tensor, features_map2: Tensor, image: npt.NDArray[np.float64], epoch: int):
        """
        Function plotting images
        @ conv1, conv2 - filters from each conv layer
        @ features_map - each images after passing it through layer
        @ image - image given at input
        """

        features_map1 = features_map1 - features_map1.min()
        features_map1 = features_map1 / features_map1.max()
        features_map2 = features_map2 - features_map2.min()
        features_map2 = features_map2 / features_map2.max()
        conv1 = conv1 - conv1.min()
        conv1 = conv1 / conv1.max()
        conv2 = conv2 - conv2.min()
        conv2 = conv2 / conv2.max()

        features_map1 = features_map1.numpy()
        features_map2 = features_map2.numpy()
        conv1 = conv1.numpy()
        conv2 = conv2.numpy()

        y = features_map1.shape[1]  # type:ignore
        x = conv1.shape[1] + 2  # type:ignore
        plt.rcParams['axes.grid'] = False  # type:ignore

        fig, ax = plt.subplots(x, y)  # type:ignore
        [axi.set_axis_off() for axi in ax.ravel()]
        image = np.transpose(image, (1, 2, 0))  # type:ignore
        ax[0, int(y/2)].imshow(image)

        n: int = y
        for f in conv1:
            for c in f:
                npimg = c
                ax[int(n/y), n % y].imshow(npimg, cmap="gray")
                n += 1

        n = (conv1.shape[1] + 1) * y
        for f in features_map1[0]:
            npimg = f
            ax[int(n/y), n % y].imshow(npimg, cmap="gray")
            n += 1

        fig.savefig('./output_images/sample_test_8/conv1/'+str(epoch)+'.png', dpi=300)

        y = features_map2.shape[1]
        x = conv2.shape[1] + 2
        plt.rcParams['axes.grid'] = False  # type:ignore

        fig, ax = plt.subplots(x, y)
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0, int(y/2)].imshow(image)

        n = y
        for f in conv2:
            for c in f:
                npimg = c
                ax[int(n/y), n % y].imshow(npimg, cmap="gray")
                n += 1

        n = (conv2.shape[1] + 1)*y
        for f in features_map2[0]:
            npimg = f
            ax[int(n/y), n % y].imshow(npimg, cmap="gray")
            n += 1

        fig.savefig('./output_images/sample_test_8/conv2/'+str(epoch)+'.png', dpi=300)

    def imshow2(self, conv1: Tensor, conv2: Tensor, features_map1: Tensor, features_map2: Tensor, image: npt.NDArray[np.float64], epoch: int):
        features_map1 = features_map1 - features_map1.min()
        features_map1 = features_map1 / features_map1.max()
        features_map2 = features_map2 - features_map2.min()
        features_map2 = features_map2 / features_map2.max()
        conv1 = conv1 - conv1.min()
        conv1 = conv1 / conv1.max()
        conv2 = conv2 - conv2.min()
        conv2 = conv2 / conv2.max()

        conv1 = conv1 * 255
        img = make_grid(conv1).numpy()
        img = np.abs(img)
        cv2.imwrite('./test.jpg', np.transpose(img, (1, 2, 0)))

        plt.rcParams['axes.grid'] = False  # type:ignore

        # fig, ax = plt.subplots(2, 1)  # type:ignore
        # [axi.set_axis_off() for axi in ax.ravel()]
        # image = np.transpose(image, (1, 2, 0))  # type:ignore
        # ax[0].imshow(image)
        # img = make_grid(conv1)
        # ax[1].imshow(img.permute(1, 2, 0))
        # # img = make_grid(features_map1)
        # # ax[2].imshow(img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        # fig.savefig('./output_images/sample_test_8/conv1/'+str(epoch)+'.png', dpi=300)

    def saveFile(self, filename: str = 'readme'):
        with open('./output_data/' + filename + '.txt', 'w') as f:
            for (index, pk) in enumerate(self.pk_cv):
                f.write('pk'+str(index + 1) + ': ' + str(pk) + '\n')
            f.write('pk: ' + str(self.current_pk) + '\n')
            if len(self.pk_cv):
                f.write('pk_avg: ' + str(np.average(self.pk_cv)) + '\n')

    def xavier_uniform_M(self, tensor: Tensor, gain: float = 1., fac: float = 3.) -> Tensor:
        r"""Fills the input `Tensor` with values according to the method
        described in `Understanding the difficulty of training deep feedforward
        neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
        distribution. The resulting tensor will have values sampled from
        :math:`\mathcal{U}(-a, a)` where

        .. math::
            a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

        Also known as Glorot initialization.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            gain: an optional scaling factor

        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        """
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(fac) * std  # Calculate uniform bounds from standard deviation

        return _no_grad_uniform_(tensor, -a, a)

    def weights_init(self, m: Module, method: str):
        """
        Function for filters initialization
        * function uses method from PyTorch to initialize weights
        @ m - model
        @ method - method to initialize weights
        TODO: add custom method to initialize weights
        """
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.bias)  # type:ignore

                if method == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight)  # type: ignore
                elif method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')  # type: ignore
                elif method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                elif method == 'xavier_uniform_M_2':
                    self.xavier_uniform_M(m.weight, gain=1.0, fac=1.)
                elif method == 'xavier_uniform_M_10':
                    self.xavier_uniform_M(m.weight, gain=1.0, fac=5.)
                elif method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                elif method == 'sobel':
                    temp: List[Tensor] = []
                    rnd = torch.rand((5, 5))
                    gkern = torch.tensor([[1,  2, 0, - 2, - 1],
                                          [4,  8, 0, - 8, - 4],
                                          [6, 12, 0, - 12, - 6],
                                          [4,  8, 0, - 8, - 4],
                                          [1,  2, 0, - 2, - 1]], dtype=torch.float32)
                    gkern = gkern * rnd
                    for _ in range(m.out_channels):
                        x: List[Tensor] = []
                        for _ in range(m.in_channels):  # type:ignore
                            rnd = randint(1, 1000) / 1000
                            gkernx = gkern[torch.randperm(gkern.size()[0])]
                            gkernx = gkernx[:, torch.randperm(gkernx.size()[1])]
                            gkern2 = gkernx/(m.out_channels + m.in_channels) * rnd
                            x.append(gkern2)
                        temp.append(torch.stack([k for k in x], 0))
                    filter_tensor: Tensor = torch.stack([k for k in temp], 0)
                    m.weight.data = filter_tensor

    def adaptive_leraning_rate(self):
        # self.sse = sum(self.loss_array)
        self.sse = self.loss_t
        self.sse_array.append(self.sse)
        lr: float = self.optimizer.param_groups[0]['lr']
        if self.sse > self.old_sse * self.er:
            # get old weights and bias
            self.cnn_model.parameters = self.old_param
            if lr >= 0.000001:
                lr *= self.lr_desc
        elif self.sse < self.old_sse:
            lr *= self.lr_inc
            lr = min([lr, 0.99])
        self.optimizer.param_groups[0]['lr'] = lr
        self.old_sse: float = self.sse

    def getSampleData(self) -> Tuple[Tensor, npt.NDArray[np.float64]]:
        i = 0
        with torch.no_grad():
            for data, labels in self.dataset.sample:
                if i == 0:
                    labels = labels.cuda()
                    data = data.cuda()
                    out, sample = self.cnn_model(data)
                    output = torch.argmax(out, dim=1)
                    print('wyjście sieci dla danego obrazu: {}'.format(self.dataset.classes[output.detach().cpu().numpy()[0]]))
                    image = data[0].detach().cpu().numpy()
                    break
                i += 1
        return sample, image

    def test(self) -> float:
        self.loss_t = 0
        pk: List[float] = []
        with torch.no_grad():
            for data, labels in self.dataset.validloader:
                labels = labels.cuda()
                data = data.cuda()
                out, _ = self.cnn_model(data)
                output = torch.argmax(out, dim=1)
                loss = self.criterion(out, labels)
                self.loss_t += loss.cpu().item()
                pk.append(self.valid_classification(output, labels))

        self.loss_test.append(self.loss_t)
        return np.average(pk)  # type:ignore

    def training(self) -> Tuple[List[float], List[float], int]:
        e = 0
        pk_test: List[float] = []
        run = True
        pk_flag = True
        self.sse_array: List[float] = []
        self.loss_test: List[float] = []
        loss = 0
        self.pk_cv: List[float] = []
        self.current_pk = 0
        while run:
            epoch_per_k = 0
            while pk_flag:
                self.loss_array = []
                self.old_param = self.cnn_model.parameters

                # pass through all data
                for data, exp in self.dataset.trainloader:
                    exp = exp.cuda()
                    # pass data to cuda
                    data = data.cuda()
                    # Wyczyszczenie gradientów z poprzedniej epoki
                    self.optimizer.zero_grad()
                    out, sample = self.cnn_model(data)
                    loss = self.criterion(out, exp)
                    loss.backward()
                    self.optimizer.step()
                    self.loss_array.append(loss.item())

                # Test
                pk = self.test()
                self.current_pk = pk
                pk_test.append(pk)
                self.adaptive_leraning_rate()

                # if e % 10 == 0:
                #     sample, image = self.getSampleData()
                #     self.imshow(self.cnn_model.cnn[0].weight.data.detach().cpu().clone(), self.cnn_model.cnn[1].weight.data.detach(  # type:ignore
                #     ).cpu().clone(), sample[0].detach().cpu().clone(), sample[1].detach().cpu().clone(), image, e)
                #     self.saveFile()

                # increment epoch
                e += 1
                epoch_per_k += 1

                if epoch_per_k >= 20:
                    self.pk_cv.append(pk)
                    pk_flag = False

                temp_lr: float = self.optimizer.param_groups[0]['lr']
                print("pk: {:.2f} %".format(pk))
                print("Learning rate: {:.10f}".format(temp_lr))
                print('Epoch: {}.............'.format(e), end=' ')
                print("Loss: {:.4f}".format(loss))

            self.dataset.get_chunks()
            pk_flag = True
            if self.dataset.last:
                run = False
        print(np.average(pk_test))
        self.saveFile(filename=(self.method + "cifar" + str(self.apt)))

        return (self.sse_array, pk_test, e)
