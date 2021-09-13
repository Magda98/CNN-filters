# Filters vizualization in Convolutiona network during training

## Dataset

Intel Image Classification

https://www.kaggle.com/puneet6060/intel-image-classification

Dataset contain 25k images of size 150px x 150px. Dataset has 6 classes:

```
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5
}
```

## CNN

Convolutional neural network has fallowing structure: `Conv2d -> ReLu -> maxPool2d -> Conv2d -> ReLu -> maxPool2d -> fc1 -> fc2 -> fc3`

Purpose of this work is to find the most efficient way to train a neural network to get the best correctness of classification.
Experiments will be conducted to find the best parameters such as filters size, filters quanitity.
There will be also examinated influence of kernels initialization to correctness of classification.
