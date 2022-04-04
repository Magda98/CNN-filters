from typing import List, Tuple
import torch
from torch.functional import Tensor
from torch.utils.data import dataset
from cnn import CnnNet
from IntelDataset import IntelDataset
from CifarDataset import CifarDataset
from statistics import mean, stdev
import numpy.typing as npt
import numpy as np
from torchvision.utils import make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import cv2


class Model():
    def __init__(self, model_name):
        def printgradnorm(module, grad_input, grad_output):
            self.gradients = (grad_input[0], grad_output[0])
        model: CnnNet = torch.load('./output_data/models/kaiming/'+model_name)
        print(summary(model, torch.zeros((3, 3, 150, 150)).cuda(), show_input=True))
        self.model_name = model_name
        # model.eval()
        model.cnn[0].register_full_backward_hook(printgradnorm)

        self.gradients = 0

        self.model = model
        # self.dataset = CifarDataset()
        self.dataset = IntelDataset()

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

    def testModel(self):
        self.loss_t = 0
        criterion = nn.NLLLoss()
        pk: List[float] = []
        with torch.no_grad():
            for data, labels in self.dataset.testloader:
                labels = labels.cuda()
                data = data.cuda()
                out = self.model(data)
                output = torch.argmax(out, dim=1)
                loss = criterion(out, labels)
                self.loss_t += loss.cpu().item()
                pk.append(self.valid_classification(output, labels))

        return np.average(pk)  # type:ignore

    def getSampleData(self) -> Tuple[Tensor, npt.NDArray[np.float64]]:
        model = self.model
        dataset = self.dataset
        criterion = nn.NLLLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        i = 0
        for data, labels in dataset.sample:
            if i == 0:
                labels = labels.cuda()
                data = data.cuda()

                optimizer.zero_grad()
                out, sample = model(data)
                output = torch.argmax(out, dim=1)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                print('wyjście sieci dla danego obrazu: {}'.format(dataset.classes[output.detach().cpu().numpy()[0]]))
                image = data[0].detach().cpu().numpy()
                break
            i += 1
        return sample, image

    def save_gradient_images(self, gradient):
        """
            Exports the original gradient image
        Args:
            gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
            file_name (str): File name to be exported
        """
        # Normalize
        gradient = gradient - gradient.min()
        gradient /= gradient.max()
        img = (gradient.permute(1, 0, 2, 3) * 255)
        # grayscale_im = np.sum(np.abs(gradient[0].numpy()), axis=0)
        # gradient = gradient.permute(1, 0, 2, 3)
        img = make_grid(img)
        # plt.imshow(img.permute(1, 2, 0),  cmap='gray')
        # plt.show()
        # Save image
        img = np.abs(np.int16(cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)))
        # plt.imshow(img, vmin=0, vmax=255)
        # plt.show()
        # img = np.abs(img.numpy())
        cv2.imwrite('./output_data/output_images/weight_init/' + self.model_name[:-2]+'/gradients_map.jpg',  img)

    def save_images(self, conv1: Tensor, conv2: Tensor, features_map1: Tensor, features_map2: Tensor):
        conv1 = conv1 - conv1.min()
        conv1 /= conv1.max()
        img = (conv1.permute(1, 0, 2, 3) * 255)
        img = make_grid(img, nrow=conv1.shape[1])
        img = torch.unsqueeze(img, 0)
        img = make_grid(img.permute(1, 0, 2, 3), nrow=1)
        img = np.abs(np.int16(cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)))
        cv2.imwrite('./output_data/output_images/weight_init/' + self.model_name[:-2]+'/filters_1.jpg',  img)

        conv2 = conv2 - conv2.min()
        conv2 /= conv2.max()
        img = (conv2.permute(1, 0, 2, 3) * 255)
        img = make_grid(img, nrow=conv2.shape[1])
        img = torch.unsqueeze(img, 0)
        img = make_grid(img.permute(1, 0, 2, 3), nrow=1)
        img = np.abs(np.int16(cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)))
        cv2.imwrite('./output_data/output_images/weight_init/' + self.model_name[:-2]+'/filters_2.jpg',  img)

        features_map1 = features_map1 - features_map1.min()
        features_map1 /= features_map1.max()
        img = (features_map1.permute(1, 0, 2, 3) * 255)
        img = make_grid(img)
        img = np.abs(np.int16(cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)))
        cv2.imwrite('./output_data/output_images/weight_init/' + self.model_name[:-2]+'/features_map_1.jpg',  img)

        features_map2 = features_map2 - features_map2.min()
        features_map2 /= features_map2.max()
        img = (features_map2.permute(1, 0, 2, 3) * 255)
        img = make_grid(img)
        img = np.abs(np.int16(cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)))
        cv2.imwrite('./output_data/output_images/weight_init/' + self.model_name[:-2]+'/features_map_2.jpg',  img)


if __name__ == "__main__":
    x = []
    for i in range(3, 6):
        generateImages = Model(model_name="kaiming_uniform_" + str(i))
        x.append(generateImages.testModel())
        generateImages.getSampleData()
        generateImages.save_gradient_images(generateImages.gradients[1].cpu().detach())
        sample, image = generateImages.getSampleData()
        generateImages.save_images(generateImages.model.cnn[0].weight.data.detach().cpu().clone(
        ), generateImages.model.cnn[2].weight.data.detach().cpu().clone(), sample[0].detach().cpu().clone(), sample[1].detach().cpu().clone())

        generateImages.save_gradient_images(generateImages.gradients[1].cpu())
    print(mean(x))
    print(stdev(x))
