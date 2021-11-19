from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms

# Hyperparameters for our network
input_size   = 784
hidden_sizes = [128, 64]
output_size  = 10

class Net:
    def build(input_size, hidden_sizes, output_size):

        model    = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(hidden_sizes[1], output_size))]))
        return model