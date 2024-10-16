import torch.nn as nn
from typing import List
import torch
import torch.nn.functional as F
import base64
import io


def lstm_to_onnx(torch_model, dataset):
    if torch_model.training:
        torch_model.eval()
    torch_model = torch_model.cpu()
    buffer = io.BytesIO()
    torch.onnx.export(
        torch_model,
        args=(dataset.X),
        f=buffer,
        input_names=["batch", "sequence", "features"],
        dynamic_axes={"batch": [0]},
    )
    onnx_model_bytes = buffer.getvalue()
    buffer.close()
    model_scripted_base64 = base64.b64encode(onnx_model_bytes).decode("utf-8")

    return model_scripted_base64


class Sequence_LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout,
        activation,
        bidirectional=False,
    ):
        super(Sequence_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)
        out, (_, _) = self.layer(
            x, (h0, c0)
        )  # out: (batch, seq_length, hidden_size*num_directions)
        # Take the output from the last time step
        out = out[:, -1, :]  # (batch, hidden_size*num_directions)
        out = self.dropout(out)
        out = self.activation(self.fc1(out))  # (batch, hidden_size)
        out = self.fc2(out)  # (batch, num_classes)

        return out
