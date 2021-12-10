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

class Model:
    def build(input_size, hidden_sizes, output_size):

        model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(hidden_sizes[1], output_size))]))
        return model

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = 5, padding = 'same'),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, padding = 'same'),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.fc1 = nn.Sequential(
			nn.Linear(8*8*64, 64),
			nn.ReLU(),
			nn.Dropout(p = 0.50))
		self.fc2 = nn.Sequential(
			nn.Linear(64, 10),
			nn.ReLU(),
			nn.Dropout(p = 0.50))

	def forward(self, input):
		out = self.layer1(input)
		out = self.layer2(out)
		out = out.reshape(out.shape[0], -1)
		out = self.fc1(out)
		out = self.fc2(out)
		out = F.softmax(out, dim = 1)
		return out