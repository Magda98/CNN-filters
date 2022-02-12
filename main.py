import numpy as np
import seaborn as sns
from CifarDataset import CifarDataset
from IntelDataset import IntelDataset
from training import trainingModel
import torch


if __name__ == "__main__":

    sns.set()  # type:ignore
    torch.cuda.empty_cache()

    # input image size in px (square image)
    input_size = 32
    dataset_name = "cifar"

    # methods = ['xavier_uniform', 'xavier_uniform_M_10', 'xavier_uniform_M_2', 'xavier_uniform_M_1', 'xavier_uniform_M_14']
    methods = ['xavier_uniform_M_10', 'xavier_uniform_M_2', 'xavier_uniform_M_1', 'xavier_uniform_M_14', 'xavier_uniform_M_20']

    if dataset_name == "cifar":  # type: ignore
        for method in methods:
            for apt in range(3):
                model = trainingModel(dataset=CifarDataset(), method=method, input_size=input_size,
                                      c_kernels=[3, 3, 3, 3, 3, 3], in_channels=[3, 16, 32, 64, 86], out_channels=[16, 32, 64, 86, 128], apt=apt, dataset_name=dataset_name, epoch=2)
                sse, sse_t, pk, e = model.training()
                np.savetxt("data_plots/" + dataset_name + method + str(apt) + ".csv", sse, delimiter=";")
                np.savetxt("data_plots/" + dataset_name + method + str(apt) + "_t.csv", sse_t, delimiter=";")
                torch.save(model.cnn_model, "models/" + dataset_name + method + str(apt))
    elif dataset_name == "intel":
        for method in methods:
            for apt in range(13, 14, 1):
                model = trainingModel(dataset=IntelDataset(), method=method, input_size=input_size,
                                      c_kernels=[5, 5, 5, 5, 5, 5, 5], in_channels=[3, 16, 32, 64, 86, 128, 128, 128], out_channels=[16, 32, 64, 86, 128, 128, 128], apt=apt, dataset_name=dataset_name, epoch=200)
                sse, sse_t, pk, e = model.training()
                np.savetxt("data_plots/" + method + str(apt) + ".csv", sse, delimiter=";")
                np.savetxt("data_plots/" + method + str(apt) + "_t.csv", sse_t, delimiter=";")
                torch.save(model.cnn_model, "models/" + method + str(apt))
