import matplotlib.pyplot as plt
import torch
from training import trainingModel
from intel_data import IntelDataset
from sample_data import SampleDataset
import seaborn as sns


if __name__ == "__main__":

    sns.set()  # type:ignore
    torch.cuda.empty_cache()

    # input image size in px (square image)
    input_size = 150

    methods = ['orthogonal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal', 'custom']
    # for method in methods:
    #     sse, pk = training(dataset=data, test = test ,epoch=epoch, method=method)
    #     plt.plot(epoch, sse, label=method)
    # c1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    # c2 = [41]
    # fileName = 'filter_count_3'
    # results = []

    # for method in methods:
    model = trainingModel(dataset=IntelDataset(), method='xavier_normal', input_size=input_size,
                          c_kernels=[30, 20], in_channels=[3, 3], out_channels=[3, 16])
    sse, pk, e = model.training()

    torch.save(model.cnn_model.state_dict(), 'model')
    # model.load_state_dict(torch.load(PATH))
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
