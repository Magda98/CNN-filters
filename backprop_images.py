from typing import Tuple
import torch
from torch.functional import Tensor
from torch.utils.data import dataset
from cnn import CnnNet
from intel_data import IntelDataset
from cifar_data import CifarDataset

import numpy.typing as npt
import numpy as np
from torchvision.utils import make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2


class Images():
    def __init__(self, model_name):
        def printgradnorm(module, grad_input, grad_output):
            self.gradients = (grad_input[0], grad_output[0])
        model: CnnNet = torch.load('models/'+model_name)

        self.model_name = model_name
        # model.eval()
        model.cnn[0].register_full_backward_hook(printgradnorm)

        self.gradients = 0

        self.model = model
        self.intel = CifarDataset()

    def getSampleData(self) -> Tuple[Tensor, npt.NDArray[np.float64]]:
        model = self.model
        dataset = self.intel
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
        cv2.imwrite('./output_images/weight_init/' + self.model_name[:-1]+'/gradients_map.jpg',  img)


if __name__ == "__main__":

    generateImages = Images(model_name="cifarxavier_uniform_M_104")
    generateImages.getSampleData()

    generateImages.save_gradient_images(generateImages.gradients[1].cpu())
