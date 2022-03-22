import torch
from WeaponData import WeaponData
import numpy as np
import shap

dataset = WeaponData()

net = torch.load("./output_data/models/weapon_dataset")
# print(summary(net, torch.zeros((1, 3, 100, 100)).cuda(), show_input=True))

batch = next(iter(dataset.testloader))
images, labels = batch

# print(labels)

for b_index, b in enumerate(images):
    for c_index, c in enumerate(b):
        images[b_index, c_index] -= c.min()
        images[b_index, c_index] /= c.max() - c.min()
        images[b_index, c_index] = torch.nan_to_num(images[b_index, c_index])


background = images[:30]
test_images = images[24:30]

e = shap.DeepExplainer(net, background.cuda())
shap_values = e.shap_values(test_images.cuda())


shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
shap.image_plot(
    shap_numpy,
    test_numpy,
    labels=[
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
        ["banknot", "karta płatnicza", "nóż", "pistolet", "portfel", "smartphone"],
    ],
)

# ['banknot', 'karta płatnicza', 'nóż', 'pistolet', 'portfel', 'smartphone']
