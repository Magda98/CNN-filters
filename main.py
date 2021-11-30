import seaborn as sns
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

    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal', 'custom']
    # region experiments loop
    """
    for method in methods:
        sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
        plt.plot(epoch, sse, label=method)
    c1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    c2 = [41]
    fileName = 'filter_count_3'
    results = []
    """
    # endregion

    # for method in methods:
    model = trainingModel(dataset=IntelDataset(), method='sobel', input_size=input_size,
                          c_kernels=[5, 5, 5, 5, 5], in_channels=[3, 16, 32, 64, 86], out_channels=[16, 32, 64, 86, 128])
    sse, pk, e = model.training()

    torch.save(model.cnn_model, 'model')

    # region plots
    # plt.plot(range(e), sse, label=method)

    # np.savetxt("data_plots/" + fileName + ".csv", results, delimiter=";")

    # e = list(range(len(sse)))
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(loc='upper left')
    # plt.show()

    # e = list(range(len(pk)))
    # plt.figure()
    # plt.plot(e, pk, label='pk')
    # plt.xlabel("Epoch")
    # plt.ylabel("PK")
    # plt.legend(loc='upper left')
    # plt.show()
    # endregion
