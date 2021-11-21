from random import randint
from random import seed
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.typing as npt
from torch.functional import Tensor
from torch.nn.modules.module import Module
from cnn import CnnNet
import torch.nn as nn
import matplotlib
from intel_data import IntelDataset
from torchvision.utils import make_grid
matplotlib.use('Agg')


class trainingModel():
    """
    Main class of training a model
    @ dataset - dataset loader object
    @ epoch - iterator object
    @ method - methos for weights initialization
    @ test - data for testing
    @ input_size - size of image
    """

    def __init__(self, dataset, method: str, input_size: int, c_kernels: List[int] = [7, 5], out_channels: List[int] = [30, 16], in_channels: List[int] = [3, 30], p_kernel: List[int] = [2, 2], p_stride: List[int] = [2, 2]):

        seed(1)
        self.dataset = dataset
        self.cnn_model = CnnNet(input_size, c_kernels=c_kernels, out_channels=out_channels,
                                in_channels=in_channels, p_kernel=p_kernel, p_stride=p_stride)
        # weight initialization
        self.cnn_model.apply(lambda m: self.weights_init(m, method))

        self.criterion = nn.NLLLoss()
        self.lr = 0.00001
        self.er = 1.04
        self.lr_inc = 1.04
        self.lr_desc = 0.7
        self.old_sse = 0

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

        with open('./output_images/sample_test_8/readme.txt', 'w') as f:
            for (index, pk) in enumerate(self.pk_cv):
                f.write('pk'+str(index + 1) + ': ' + str(pk) + '\n')
            f.write('pk: ' + str(self.current_pk) + '\n')

    def imshow2(self, conv1: Tensor, conv2: Tensor, features_map1: Tensor, features_map2: Tensor, image: npt.NDArray[np.float64], epoch: int):
        features_map1 = features_map1 - features_map1.min()
        features_map1 = features_map1 / features_map1.max()
        features_map2 = features_map2 - features_map2.min()
        features_map2 = features_map2 / features_map2.max()
        conv1 = conv1 - conv1.min()
        conv1 = conv1 / conv1.max()
        conv2 = conv2 - conv2.min()
        conv2 = conv2 / conv2.max()

        plt.rcParams['axes.grid'] = False  # type:ignore

        fig, ax = plt.subplots(3, 1)  # type:ignore
        [axi.set_axis_off() for axi in ax.ravel()]
        image = np.transpose(image, (1, 2, 0))  # type:ignore
        ax[0].imshow(image)
        img = make_grid(conv1)
        ax[1].imshow(img.permute(1, 2, 0))
        img = make_grid(features_map1)
        ax[2].imshow(img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        fig.savefig('./output_images/sample_test_8/conv1/'+str(epoch)+'.png', dpi=300)

    def gaussian_fn(self, M: int, std: int):
        n = torch.arange(0, M) - (M - 1.0) / 2.0
        sig2 = 2 * std * std
        return torch.exp(-n ** 2 / sig2)

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
                elif method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                elif method == 'custom':
                    if (m.kernel_size[0] != 3):
                        temp: List[Tensor] = []
                        for _ in range(m.out_channels):
                            x: List[Tensor] = []
                            for _ in range(m.in_channels):  # type:ignore
                                gkern1d = self.gaussian_fn(m.kernel_size[0], randint(0, 256))
                                gkern2d = torch.outer(gkern1d, gkern1d)
                                x.append(gkern2d)
                            temp.append(torch.stack([k for k in x], 0))
                        filter_tensor: Tensor = torch.stack([k for k in temp], 0)
                    else:
                        gkern2d = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
                        filter_temp_tensor: Tensor = torch.stack(([gkern2d for _ in range(m.in_channels)]), 0)  # type:ignore
                        filter_tensor: Tensor = torch.stack(([filter_temp_tensor for _ in range(m.out_channels)]), 0)
                    m.weight.data = filter_tensor

    def adaptive_leraning_rate(self):
        self.sse = sum(self.loss_array)
        self.sse_array.append(self.sse)
        lr = self.optimizer.param_groups[0]['lr']
        if self.sse > self.old_sse * self.er:
            # get old weights and bias
            self.cnn_model.parameters = self.old_param
            if lr >= 0.000001:
                lr = self.lr_desc * lr
        elif self.sse < self.old_sse:
            lr = self.lr_inc * lr
            lr = min([lr, 0.99])
        self.optimizer.param_groups[0]['lr'] = lr
        self.old_sse = self.sse

    def getSampleData(self) -> Tuple[Tensor, npt.NDArray[np.float64]]:
        i = 0
        with torch.no_grad():
            for data, labels in self.dataset.sample:
                if i == 0:
                    labels = labels.cuda()
                    data = data.cuda()
                    out, sample = self.cnn_model(data)
                    output = torch.argmax(out, dim=1)
                    print('wyjście: {}'.format(output.detach().cpu().numpy()[0]))
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

                if e % 50 == 0:
                    sample, image = self.getSampleData()
                    self.imshow(self.cnn_model.cnn[0].weight.data.detach().cpu().clone(), self.cnn_model.cnn[2].weight.data.detach(  # type:ignore
                    ).cpu().clone(), sample[0].detach().cpu().clone(), sample[1].detach().cpu().clone(), image, e)

                self.adaptive_leraning_rate()

                # increment epoch
                e += 1
                epoch_per_k += 1

                if epoch_per_k >= 5000:
                    self.pk_cv.append(pk)
                    pk_flag = False

                temp_lr: float = self.optimizer.param_groups[0]['lr']
                print("pk: {:.2f} %".format(pk))
                print("Learning rate: {:.5f}".format(temp_lr))
                print('Epoch: {}.............'.format(e), end=' ')
                print("Loss: {:.4f}".format(loss))

            self.dataset.get_chunks()
            pk_flag = True
            if self.dataset.last:
                run = False
        print(np.average(pk_test))
        return (self.sse_array, pk_test, e)
