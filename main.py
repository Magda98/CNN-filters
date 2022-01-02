import numpy as np
import seaborn as sns
from cifar_data import CifarDataset
from sample_data import SampleDataset
from intel_data import IntelDataset
from training import trainingModel
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":

    sns.set()  # type:ignore
    torch.cuda.empty_cache()

    # input image size in px (square image)
    input_size = 150
    dataset_name = "intel"
    methods = ['xavier_uniform_M_20']
    # methods = ['xavier_uniform', 'xavier_uniform_M_10', 'xavier_uniform_M_2', 'xavier_uniform_M_1', 'xavier_uniform_M_14', 'xavier_uniform_M_20']

    # region experiments loop
    """
    for method in methods:
        sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
        plt.plot(epoch, sse, label=method)
    c1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    c2 = [41]
    fileName = 'filter_count_3'
    results = [
    """
    # endregion

    if dataset_name == "cifar":  # type: ignore
        for method in methods:
            for apt in range(5, 8):
                model = trainingModel(dataset=CifarDataset(), method=method, input_size=input_size,
                                      c_kernels=[3, 3], in_channels=[3, 16], out_channels=[16, 32], apt=apt, dataset_name=dataset_name, epoch=200)
                sse, pk, e = model.training()
                np.savetxt("data_plots/" + dataset_name + method + str(apt) + ".csv", sse, delimiter=";")
                torch.save(model.cnn_model, "models/" + dataset_name + method + str(apt))

    elif dataset_name == "intel":
        for method in methods:
            for apt in range(10, 11, 1):
                model = trainingModel(dataset=IntelDataset(), method=method, input_size=input_size,
                                      c_kernels=[5, 5, 5, 5, 5, 5, 5], in_channels=[3, 16, 32, 64, 86, 128, 128, 128], out_channels=[16, 32, 64, 86, 128, 128, 128], apt=apt, dataset_name=dataset_name, epoch=200)
                sse, pk, e = model.training()
                np.savetxt("data_plots/" + method + str(apt) + ".csv", sse, delimiter=";")
                torch.save(model.cnn_model, "models/" + method + str(apt))
