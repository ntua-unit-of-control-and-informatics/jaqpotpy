import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    TransformerConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
import torch.nn.init as init
from torch import Tensor
import base64
import io


def pyg_to_onnx(torch_model, featurizer):
    """
    Converts a PyTorch geometric model to ONNX format and returns a base64-encoded string of the model.

    Args:
        torch_model (nn.Module): The trained PyTorch model to be converted.
        featurizer (SmilesGraphFeaturizer): A featurizer for transforming SMILES strings into graph data.

    Returns:
        str: A base64-encoded ONNX model string.
    """
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


# def pyg_to_torchscript(torch_model):
#     if torch_model.training:
#         torch_model.eval()
#     torch_model = torch_model.cpu()
#     script_model = torch.jit.script(torch_model)
#     model_buffer = io.BytesIO()
#     torch.jit.save(script_model, model_buffer)
#     model_buffer.seek(0)
#     script_base64 = base64.b64encode(model_buffer.getvalue()).decode("utf-8")

#     return script_base64


class BaseGraphNetwork(nn.Module):
    """
    Base class for constructing various types of graph neural networks with flexible layer configurations.

    Attributes:
        input_dim (int): Dimension of input features for each node.
        hidden_layers (int): Number of hidden layers.
        hidden_dim (int): Dimension of each hidden layer.
        output_dim (int): Dimension of the output layer.
        activation (nn.Module): Activation function.
        dropout_proba (float): Dropout probability for regularization.
        batch_norm (bool): Whether to use batch normalization.
        seed (int): Random seed for reproducibility.
        pooling (str): Pooling method to use ('mean', 'add', or 'max').
        edge_dim (Optional[int]): Dimension of edge features.
        heads (Optional[int]): Number of attention heads (for certain layers).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        batch_norm: bool = False,
        seed=42,
        pooling: str = "mean",
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
        self.batch_norm = batch_norm
        self.seed = seed
        self.pooling = pooling
        self.edge_dim = edge_dim
        self.heads = heads
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(activation)
        self._validate_inputs()

        self.graph_layers = nn.ModuleList()

    def add_layer(self, conv_layer: nn.Module):
        """
        Helper function to add a convolution layer to the graph layers list.

        Args:
            conv_layer (nn.Module): Convolution layer to add.
        """
        self.graph_layers.append(conv_layer)

    def pooling_layer(self, x: Tensor, batch: Optional[Tensor]) -> Tensor:
        """
        Applies the specified pooling method.

        Args:
            x (Tensor): Node embeddings.
            batch (Optional[Tensor]): Batch index tensor.

        Returns:
            Tensor: Pooled graph-level embeddings.
        """
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "add":
            return global_add_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError("pooling must be either 'mean' or 'add'")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor],
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph edges.
            batch (Optional[Tensor]): Batch indices for each node.
            edge_attr (Optional[Tensor]): Edge features (if applicable).

        Returns:
            Tensor: Output node/graph embeddings.
        """
        if self.edge_dim is not None:
            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index, edge_attr)
                if self.batch_norm:
                    x = self.norm_layer(x)
                x = self.activation(x)
                x = self.dropout(x)
            x = self.pooling_layer(x, batch)
            x = self.fc(x)
            return x
        else:
            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index)
                if self.batch_norm:
                    x = self.norm_layer(x)
                x = self.activation(x)
                x = self.dropout(x)
            x = self.pooling_layer(x, batch)
            x = self.fc(x)
            return x

    def _validate_inputs(self):
        """
        Validates the input parameters for the network.
        Raises appropriate errors if the types do not match.
        """
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
        if not isinstance(self.batch_norm, bool):
            raise TypeError("batch_norm must be of type bool")


class GraphSageNetwork(BaseGraphNetwork):
    """
    GraphSAGENetwork model.

    Attributes:
        Inherits attributes from BaseGraphNetwork.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        batch_norm: bool = False,
        seed=42,
        pooling: str = "mean",
    ):
        """
        Initializes the GraphSageNetwork with SAGEConv-specific layers.
        """
        super(GraphSageNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            batch_norm,
            seed,
            pooling,
        )

        # Add SAGEConv layers
        self.add_layer(SAGEConv(input_dim, hidden_dim))
        for _ in range(hidden_layers):
            self.add_layer(SAGEConv(hidden_dim, hidden_dim))

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class GraphConvolutionNetwork(BaseGraphNetwork):
    """
    Graph Convolutional Network (GCN) model.

    Attributes:
        Inherits attributes from BaseGraphNetwork.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        batch_norm: bool = False,
        seed=42,
        pooling: str = "mean",
    ):
        """
        Initializes the GraphConvolutionNetwork with GCNConv-specific layers.
        """
        super(GraphConvolutionNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            batch_norm,
            seed,
            pooling,
        )

        # Add GCNConv layers
        self.add_layer(GCNConv(input_dim, hidden_dim))
        for _ in range(hidden_layers):
            self.add_layer(GCNConv(hidden_dim, hidden_dim))

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class GraphAttentionNetwork(BaseGraphNetwork):
    """
    Graph Attention Network (GAT) model.

    Attributes:
        Inherits attributes from BaseGraphNetwork.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        batch_norm: bool = False,
        seed=42,
        pooling: str = "mean",
        edge_dim: Optional[int] = None,
        heads: int = 1,
    ):
        """
        Initializes the GraphAttentionNetwork with GATConv-specific layers.
        """
        super(GraphAttentionNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            batch_norm,
            seed,
            pooling,
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
        self.norm_layer = nn.BatchNorm1d(hidden_dim * heads)
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


class GraphTransformerNetwork(BaseGraphNetwork):
    """
    Graph Transformer Network model.

    Attributes:
        Inherits attributes from BaseGraphNetwork.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout_proba: float = 0.0,
        batch_norm: bool = False,
        seed=42,
        pooling: str = "mean",
        edge_dim: Optional[int] = None,
        heads: int = 1,
    ):
        """
        Initializes the GraphTransformerNetwork with TransformerConv-specific layers.
        """
        super(GraphTransformerNetwork, self).__init__(
            input_dim,
            hidden_layers,
            hidden_dim,
            output_dim,
            activation,
            dropout_proba,
            batch_norm,
            seed,
            pooling,
            edge_dim,
            heads,
        )

        # Add TransformerConv layers
        self.add_layer(TransformerConv(input_dim, hidden_dim, heads, edge_dim=edge_dim))
        for _ in range(hidden_layers):
            self.add_layer(
                TransformerConv(
                    hidden_dim * heads, hidden_dim, heads, edge_dim=edge_dim
                )
            )

        # Set up additional layers
        self.dropout = nn.Dropout(dropout_proba)
        self.norm_layer = nn.BatchNorm1d(hidden_dim * heads)
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
