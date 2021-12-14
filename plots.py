import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    methods = ["xavier_uniform", "xavier_uniform_M_1", "xavier_uniform_M_2", "xavier_uniform_M_10", "xavier_uniform_M_14", "xavier_uniform_M_20"]
    methods = ["cifarxavier_uniform", "cifarxavier_uniform_M_1", "cifarxavier_uniform_M_2", "cifarxavier_uniform_M_10", "cifarxavier_uniform_M_14", "cifarxavier_uniform_M_20"]
    for m in methods:
        temp = np.empty((3, 200), np.float64)
        for i in range(3):
            t = np.array([])
            with open('data_plots/'+m+str(i)+'.csv') as f:
                for l in f:
                    t = np.append(t, float(l))
            temp[i] = t

        mean = np.mean(temp, axis=0)
        plt.plot(range(200), mean, label=m)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
