
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_,  _calculate_correct_fan, calculate_gain
import torch.nn as nn
import math


def adaptive_leraning_rate(loss, optimizer, model, old_param, old_loss):
    er = 1.04
    lr_inc = 1.04
    lr_desc = 0.7

    lr = optimizer.param_groups[0]['lr']
    if loss > loss * er:
        # get old weights and bias
        model.parameters = old_param
        if lr >= 0.000001:
            lr *= lr_desc
    elif loss < old_loss:
        lr *= lr_inc
        lr = min([lr, 0.99])
    optimizer.param_groups[0]['lr'] = lr

    return model, optimizer


def test(dataset, model, criterion):
    total = 0.
    predicted = 0.
    loss_test = 0
    with torch.no_grad():
        for data, labels in dataset.testloader:
            labels = labels.cuda()
            data = data.cuda()
            out = model(data)
            predicted_output = torch.argmax(out, dim=1)

            total += labels.shape[0]
            predicted += torch.sum(predicted_output == labels).cpu().item()

            loss = criterion(out, labels)
            loss_test += loss.cpu().item()

    return (predicted/total*100), loss_test


def kaiming_uniform_M(tensor, a=0, mode='fan_in', nonlinearity='relu', M=1.0):
    # if 0 in tensor.shape:
    #     warnings.warn("Initializing zero-element tensors is a no-op")
    #     return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    # ReLU gain = :math:`\sqrt{2}`
    std = gain / math.sqrt(fan)
    bound = math.sqrt(M) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def xavier_uniform_M(tensor, gain=1., fac=3.):
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(fac) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def weights_init(m, method):
    """
    Function for filters initialization
    * function uses method from PyTorch to initialize weights
    @ m - model
    @ method - method to initialize weights
    """
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            # „fan_in” (domyślnie) zachowuje wielkość wariancji wag w przebiegu do przodu
            if method == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif method == 'kaiming_uniform_M_1':
                kaiming_uniform_M(m.weight, mode='fan_in', nonlinearity='relu', M=0.5)
            elif method == 'kaiming_uniform_M_2':
                kaiming_uniform_M(m.weight, mode='fan_in', nonlinearity='relu', M=1.0)
            elif method == 'kaiming_uniform_M_10':
                kaiming_uniform_M(m.weight, mode='fan_in', nonlinearity='relu', M=5.0)
            elif method == 'kaiming_uniform_M_14':
                kaiming_uniform_M(m.weight, mode='fan_in', nonlinearity='relu', M=7.0)
            elif method == 'kaiming_uniform_M_20':
                kaiming_uniform_M(m.weight, mode='fan_in', nonlinearity='relu', M=10.0)
            elif method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif method == 'xavier_uniform_M_2':
                xavier_uniform_M(m.weight, gain=1.0, fac=1.)
            elif method == 'xavier_uniform_M_10':
                xavier_uniform_M(m.weight, gain=1.0, fac=5.)
            elif method == 'xavier_uniform_M_14':
                xavier_uniform_M(m.weight, gain=1.0, fac=7.)
            elif method == 'xavier_uniform_M_20':
                xavier_uniform_M(m.weight, gain=1.0, fac=10.)
            elif method == 'xavier_uniform_M_1':
                self.xavier_uniform_M(m.weight, gain=1.0, fac=0.5)
            elif method == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)
