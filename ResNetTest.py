import numpy as np
import torch.nn as nn
import torch
from intel_data import IntelDataset


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        self.expansion = 4 if self.num_layers > 34 else 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        self.expansion = 1 if num_layers < 50 else 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers in [34, 50]:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        m = torch.nn.LogSoftmax(dim=1)
        x = m(self.fc(x))
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion  # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels))  # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


def adaptive_leraning_rate(loss_t, old_sse, optimizer, lr, net, old_param):
    er = 1.04
    lr_inc = 1.04
    lr_desc = 0.7
    # self.sse = sum(self.loss_array)
    sse = loss_t
    lr = optimizer.param_groups[0]['lr']
    if sse > old_sse * er:
        # get old weights and bias
        net.parameters = old_param
        if lr >= 0.000001:
            lr *= lr_desc
    elif sse < old_sse:
        lr *= lr_inc
        lr = min([lr, 0.99])
    optimizer.param_groups[0]['lr'] = lr
    old_sse = sse

    return lr


def valid_classification(out, d):
    """
        Function calculating valid classification
        @ out - netowerk output
        @ d - destination value
        return: classification correctness in %
        """
    out = out.cpu().detach().numpy()
    d = d.cpu().detach().numpy()
    temp = abs(d - out)
    valid = sum(i < 0.5 for i in temp)
    return valid / temp.shape[0] * 100  # type:ignore


def testData(net, criterion, dataset):
    pk = []
    loss_t = 0
    with torch.no_grad():
        for data, labels in dataset.testloader:
            labels = labels.cuda()
            data = data.cuda()
            out = net(data)
            output = torch.argmax(out, dim=1)
            loss = criterion(out, labels)
            loss_t += loss.cpu().item()
            pk.append(valid_classification(output, labels))
    return np.average(pk), loss_t


def test():

    lr = 0.001
    dataset = IntelDataset()
    net = ResNet50(img_channels=3, num_classes=6)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        net = net.cuda()
        print("GPU is available")
    else:
        torch.device("cpu")
        print("GPU not available, CPU used")
    old_sse = 0
    for e in range(100):
        old_param = net.parameters
        l = 0
        for data, exp in dataset.trainloader:
            optimizer.zero_grad()
            y = net(data.cuda())
            loss = criterion(y, exp.cuda())
            loss.backward()
            l += loss.cpu().item()
            optimizer.step()
        pk, loss_t = testData(net, criterion, dataset)
        lr = adaptive_leraning_rate(l, old_sse, optimizer, lr, net, old_param)
        old_sse = l
        print("pk: {:.2f} %".format(pk))
        print("Learning rate: {:.10f}".format(lr))
        print("Loss: {:.10f}".format(l))
        print('Epoch: {}.............\n'.format(e), end=' ')
        # if pk > 40.0:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)


if __name__ == "__main__":
    test()
