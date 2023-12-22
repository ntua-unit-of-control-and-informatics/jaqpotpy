import torch
from jaqpotpy.cfg import config
import torch.nn.functional as F
import math


class CNN(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList, out_channels, input_size, dropout=0):
        super(CNN, self).__init__()

        torch.manual_seed(config.global_seed)

        self.dropout = dropout
        self.layers = layers
        self.input_size = input_size

        # fully connected layer
        self.out = torch.nn.Linear(self.linear_input_size(), out_channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # flatten the output to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # return x for visualization

    # after obtaining the size in above method, we call it and multiply all elements of the returned size.
    def linear_input_size(self):
        dummy = torch.rand(1, self.layers[0].__dict__['in_channels'], self.input_size[0],
                           self.input_size[1])  # image size: 64x32
        return math.prod(self.layers(dummy).size())
