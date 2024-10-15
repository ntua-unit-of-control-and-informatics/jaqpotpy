import torch.nn as nn
from typing import List


class Sequence_NN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_seq_layers,
        model_type,
        activation,
        dropout_rate,
        out_size,
        bidirectional,
    ):
        super(Sequence_NN, self).__init__()

        if model_type == "rnn":
            self.layer = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_seq_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif model_type == "lstm":
            self.layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_seq_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        if bidirectional:
            self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = activation
        self.fc_out = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, (h_n, _) = self.layer(x)
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out
