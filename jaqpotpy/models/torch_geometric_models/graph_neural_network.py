import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from torch_geometric.nn import SAGEConv, GCNConv, GraphNorm, global_add_pool
import torch.nn.init as init
from torch import Tensor


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
        jittable: bool = True,
        seed=42,
    ):
        super(BaseGraphNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_proba = dropout_proba
        self.graph_norm = graph_norm
        self.jittable = jittable
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._validate_inputs()

        self.graph_layers = nn.ModuleList()

    def add_layer(self, conv_layer: nn.Module):
        """Helper function to add a convolution layer."""
        if self.jittable:
            self.graph_layers.append(conv_layer.jittable())
        else:
            self.graph_layers.append(conv_layer)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor]) -> Tensor:
        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index)
            if self.graph_norm:
                x = self.norm_layer(x, batch)
            x = self.activation(x)
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return x

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
        if not isinstance(self.jittable, bool):
            raise TypeError("jittable must be of type bool")


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
        jittable: bool = True,
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
            jittable,
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
        jittable: bool = True,
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
            jittable,
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
