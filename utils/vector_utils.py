import numpy as np
from torch import Tensor, from_numpy, randn, full
import torch.nn as nn
from torch.autograd.variable import Variable


def vectors_to_images(vectors,w,h):
    return vectors.view(vectors.size(0), 3, w,h)

def values_target(size: tuple, value: float, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result


def weights_init(m):
    """ initialize convolutional and batch norm layers in generator and discriminator """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)