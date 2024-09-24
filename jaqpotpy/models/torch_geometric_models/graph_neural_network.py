import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphNorm, global_add_pool
import torch.nn.init as init
from torch import Tensor
import base64
import io


def pyg_to_onnx(torch_model, featurizer):
    if torch_model.training:
        torch_model.eval()
        torch_model = torch_model.cpu()

    dummy_smile = "CCC"
    dummy_input = featurizer.featurize(dummy_smile)
    x = dummy_input.x
    edge_index = dummy_input.edge_index
    batch = torch.zeros(x.shape[0], dtype=torch.int64)
    buffer = io.BytesIO()
    torch.onnx.export(
        torch_model,
        args=(x, edge_index, batch),
        f=buffer,
        input_names=["x", "edge_index", "batch"],
        dynamic_axes={"x": {0: "nodes"}, "edge_index": {1: "edges"}, "batch": [0]},
    )
    onnx_model_bytes = buffer.getvalue()
    buffer.close()
    model_scripted_base64 = base64.b64encode(onnx_model_bytes).decode("utf-8")

    return model_scripted_base64


def pyg_to_torchscript(torch_model):

    if torch_model.training:
        torch_model.eval()
    torch_model = torch_model.cpu()
    script_model = torch.jit.script(torch_model)
    model_buffer = io.BytesIO()
    torch.jit.save(script_model, model_buffer)
    model_buffer.seek(0)
    script_base64 = base64.b64encode(model_buffer.getvalue()).decode("utf-8")

    return script_base64


class BaseGraphNetwork(nn.Module):
    """Base class for Graph Networks to avoid code duplication"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        graph_norm: bool = False,
        seed=42,
        edge_dim: Optional[int] = None,
        heads: Optional[int] = None,
    ):
        super(BaseGraphNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_proba = dropout_proba
        self.graph_norm = graph_norm
        self.seed = seed
        self.edge_dim = edge_dim
        self.heads = heads
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(activation)
        self._validate_inputs()

        self.graph_layers = nn.ModuleList()

    def add_layer(self, conv_layer: nn.Module):
        """Helper function to add a convolution layer."""
        self.graph_layers.append(conv_layer)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor],
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if self.edge_dim is not None:
            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index, edge_attr)
                if self.graph_norm:
                    x = self.norm_layer(x, batch)
                x = self.activation(x)
                x = self.dropout(x)
            x = global_add_pool(x, batch)
            x = self.fc(x)
            return x
        else:
            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index)
                if self.graph_norm:
                    x = self.norm_layer(x, batch)
                x = self.activation(x)
                x = self.dropout(x)
            x = global_add_pool(x, batch)
            x = self.fc(x)
            return x

    def to_onnx(self, featurizer):
        if self.model.training:
            self.model.eval()
            self.model = self.model.cpu()

        dummy_smile = "CCC"
        dummy_input = featurizer.featurize(dummy_smile)
        x = dummy_input.x
        edge_index = dummy_input.edge_index
        batch = torch.zeros(x.shape[0], dtype=torch.int64)
        buffer = io.BytesIO()
        torch.onnx.export(
            self.model,
            args=(x, edge_index, batch),
            f=buffer,
            input_names=["x", "edge_index", "batch"],
            dynamic_axes={"x": {0: "nodes"}, "edge_index": {1: "edges"}, "batch": [0]},
        )
        onnx_model_bytes = buffer.getvalue()
        buffer.close()
        model_scripted_base64 = base64.b64encode(onnx_model_bytes).decode("utf-8")

        return model_scripted_base64

    def _validate_inputs(self):
        if not isinstance(self.input_dim, int):
            raise TypeError("input_dim must be of type int")
        if not isinstance(self.hidden_layers, int):
            raise TypeError("hidden_layers must be an int")
        if not isinstance(self.hidden_dim, int):
            raise TypeError("hidden_dim must be an int")
        if not isinstance(self.output_dim, int):
            raise TypeError("output_dim must be of type int")
        if not isinstance(self.activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module like nn.Relu()")
        if not isinstance(self.dropout_proba, float):
            raise TypeError("dropout must be of type float between 0 and 1")
        if not isinstance(self.graph_norm, bool):
            raise TypeError("graph_norm must be of type bool")


class GraphSageNetwork(BaseGraphNetwork):
    """Graph Sage Model"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        graph_norm: bool = False,
        seed=42,
    ):
        super(GraphSageNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            graph_norm,
            seed,
        )

        # Add SAGEConv layers
        self.add_layer(SAGEConv(input_dim, hidden_dim))
        for _ in range(hidden_layers):
            self.add_layer(SAGEConv(hidden_dim, hidden_dim))

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = GraphNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class GraphConvolutionNetwork(BaseGraphNetwork):
    """Graph Convolution Model"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        graph_norm: bool = False,
        seed=42,
    ):
        super(GraphConvolutionNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            graph_norm,
            seed,
        )

        # Add GCNConv layers
        self.add_layer(GCNConv(input_dim, hidden_dim))
        for _ in range(hidden_layers):
            self.add_layer(GCNConv(hidden_dim, hidden_dim))

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = GraphNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class GraphAttentionNetwork(BaseGraphNetwork):
    """Graph Convolution Model"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        graph_norm: bool = False,
        seed=42,
        edge_dim: Optional[int] = None,
        heads: int = 1,
    ):
        super(GraphAttentionNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            graph_norm,
            seed,
            edge_dim,
            heads,
        )

        # Add GATConv layers
        self.add_layer(GATConv(input_dim, hidden_dim, heads, edge_dim=edge_dim))
        for _ in range(hidden_layers):
            self.add_layer(
                GATConv(hidden_dim * heads, hidden_dim, heads, edge_dim=edge_dim)
            )

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = GraphNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
