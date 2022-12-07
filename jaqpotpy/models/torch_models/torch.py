from jaqpotpy.cfg import config
import torch
import torch.nn.functional as F
from torch.nn import Linear

class Feedforward_V1(torch.nn.Module):
    def __init__(self, input_size
                 , num_layers, hidden_layers, out_size
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(Feedforward_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_layers)]) #.jittable()
        for i in range(num_layers - 1):
            self.layers.append(Linear(hidden_layers, hidden_layers)) # .jittable()
        self.out = torch.nn.Linear(hidden_layers, out_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class LSTM_V1(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, out_size,
                 activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(LSTM_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.out = torch.nn.Linear(hidden_layer_size, out_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, x):
        if self.norm is not None and self.act_first is True:
            x = self.act(x)
            x = self.norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.norm is not None and self.act_first is False:
            x = self.norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        lstm_out, self.hidden_cell = self.lstm(x.view(len(x) ,1, -1), self.hidden_cell)
        x = self.out(lstm_out.view(len(x), -1))
        return x


class RNN_V1(torch.nn.Module):
    def __init__(self, input_size
                 , num_layers, hidden_layers, out_size
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(RNN_V1, self).__init__()
        torch.manual_seed(config.global_seed)

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        # Number of hidden dimensions
        self.hidden_dim = num_layers

        # Number of hidden layers
        self.layer_dim = hidden_layers

        # RNN
        self.rnn = torch.nn.RNN(input_size, self.hidden_dim, self.layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.out = torch.nn.Linear(self.hidden_dim, out_size)

    def forward(self, x):
        if self.norm is not None and self.act_first is True:
            x = self.act(x)
            x = self.norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.norm is not None and self.act_first is False:
            x = self.norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # One time step
        x, hn = self.rnn(x, h0)
        x = self.out(x[:, -1, :])
        return x

