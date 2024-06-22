"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

import torch
import torch.nn as nn
from typing import Optional, Iterable, Union
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphNorm #, BatchNorm, GraphSizeNorm, InstanceNorm, LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init
from torch import Tensor

from .fully_connected_network import FullyConnectedNetwork


class GraphConvBlock(nn.Module):
    """
    A single Graph Convolutional Block consisting of a GCNConv layer, an activation function,
    and a dropout layer. Optionally, a graph normalization layer can be applied.

    Args:
        input_dim (int): Dimension of the input node features.
        hidden_dim (int): Dimension of the hidden features.
        activation (nn.Module): Activation function to apply after the GCNConv layer.
        dropout_probability (float): Dropout probability.
        graph_norm (bool): Whether to apply graph normalization.
        jittable (bool): Whether to make the GCNConv module jittable.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: nn.Module = nn.ReLU(),
                 dropout_probability: float = 0.5,
                 graph_norm: bool = False,
                 jittable: bool = True,
                 *args,
                 **kwargs):
        
        super(GraphConvBlock, self).__init__()

        self.jittable = jittable

        self.hidden_layer = GCNConv(input_dim, hidden_dim)
        
        if self.jittable:
            self.hidden_layer = self.hidden_layer.jittable()

        self.graph_norm = graph_norm
        if self.graph_norm:
            self.gn_layer = GraphNorm(hidden_dim)
        else:
            self.gn_layer = None

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor]) -> Tensor:
        """
        Passes the input through the layer.

        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): Batch vector with shape [num_samples,] which assigns each element to a specific example.

        Returns:
            Tensor: Output tensor of the layer.
        """

        x = self.hidden_layer(x, edge_index)
        if self.gn_layer is not None:
            x = self.gn_layer(x, batch)
        x = self.activation(x)
        x = self.dropout(x)
 
        return x



class GraphConvolutionalNetwork(nn.Module):
    """
    A Graph Convolutional Network consisting of multiple Graph Convolutional Blocks followed by a Fully Connected Layer.

    Args:
        input_dim (int): Dimension of the input node features.
        hidden_dims (Iterable[int]): Dimensions of the hidden layers.
        output_dim (int, optional): Dimension of the output. Default: 1
        activation (nn.Module, optional): Activation function to apply after each GCNConv layer. Default: nn.ReLU()
        dropout (Union[float, Iterable[float]], optional): Dropout probability(s) after each layer. Default: 0.5
        graph_norm (bool, optional): Whether to apply graph normalization. Default: False
        pooling (str, optional): Type of graph pooling operation ['mean', 'add', 'max']. Default: 'mean'
        jittable (bool, optional): Whether to make the GCNConv modules jittable. Default: True
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 output_dim: int = 1,
                 activation: nn.Module = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: bool = False,
                 pooling: str = 'mean',
                 jittable: bool = True,
                 *args,
                 **kwargs):
        
        super(GraphConvolutionalNetwork, self).__init__()
        
        # Input types check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        
        if not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be an Iterable")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if not all(isinstance(hidden_dim, int) for hidden_dim in hidden_dims):
            raise TypeError("hidden_dims must only contain integers")
        
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be of type int")
        
        if not isinstance(activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module")     
        
        if not (isinstance(dropout, float) or isinstance(dropout, Iterable)):
            raise TypeError("dropout must be either of type float or Iterable")
        if isinstance(dropout, float):
            if not 0 <= dropout <= 1:
                raise ValueError("dropout probability must be between 0 and 1")
        if isinstance(dropout, Iterable):
            for item in dropout:
                if not isinstance(item, float):
                    raise TypeError("dropout list must only contain floats")
                if not 0 <= item <= 1:
                    raise ValueError("Each element in the dropout list must be between 0 and 1")
            if len(dropout) != len(hidden_dims):
                raise ValueError("hidden_dims and dropout must be of same size")        
        
        if not isinstance(graph_norm, bool):
            raise TypeError("graph_norm must be of type bool")  
        
        if pooling is not None and pooling not in ['mean', 'add', 'max']:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")
        
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout
        self.graph_norm = graph_norm
        self.pooling = pooling
        self.jittable = jittable
        
        
        self.graph_layers = nn.ModuleList()
        graph_layer = GraphConvBlock(input_dim, hidden_dims[0],
                                     activation=activation,
                                     dropout_probability=self.dropout_probabilities[0],
                                     graph_norm=graph_norm,
                                     jittable=jittable)
        self.graph_layers.append(graph_layer)
          
        
        for i in range(len(hidden_dims) - 1):
            graph_layer = GraphConvBlock(hidden_dims[i], hidden_dims[i+1],
                                         activation=activation,
                                         dropout_probability=self.dropout_probabilities[i],
                                         graph_norm=graph_norm,
                                         jittable=jittable)
            self.graph_layers.append(graph_layer)


        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor]) -> Tensor:
        """
        Forward pass through the entire network.

        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): Batch vector with shape [num_samples,] which assigns each element to a specific example.

        Returns:
            Tensor: Output tensor after passing through the network.
        """

        x = self._forward_graph(x, edge_index, batch)
        x = self.fc(x)
        return x
    

    def _forward_graph(self,
                       x: Tensor,
                       edge_index: Tensor,
                       batch: Optional[Tensor]) -> Tensor:
        """
        Helper method for the forward pass through graph layers and pooling.
        """
    
        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index, batch)
        x = self._pooling_function(x, batch)
        return x
    
    
    def _pooling_function(self,
                          x: Tensor, 
                          batch: Optional[Tensor]) -> Tensor:
        """
        Helper method for the pooling operation.
        """

        if self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")


class GraphConvolutionalNetworkWithExternal(nn.Module):
    def __init__(self,
                 graph_input_dim: int,
                 num_external_features: int,
                 graph_hidden_dims: Iterable[int],
                 fc_hidden_dims: Iterable[int],
                 graph_output_dim: int,
                 output_dim: int = 1,
                 graph_activation: nn.Module = nn.ReLU(),
                 graph_dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: bool = False,
                 graph_pooling: str = 'mean',
                 fc_dropout: Union[float, Iterable[float]] = 0.5,
                 fc_activation: nn.Module = nn.ReLU(),
                 jittable: bool = True,
                 *args,
                 **kwargs):
        
        super().__init__()
        
        if not isinstance(num_external_features, int):
            raise TypeError("num_external_features must be of type int")
        
    
        self.graph_model = GraphConvolutionalNetwork(input_dim=graph_input_dim,
                                                     hidden_dims=graph_hidden_dims,
                                                     output_dim=graph_output_dim,
                                                     activation=graph_activation,
                                                     dropout=graph_dropout,
                                                     graph_norm=graph_norm,
                                                     pooling=graph_pooling,
                                                     jittable=jittable
                                                     )
        
        self.num_external_features = num_external_features
        self.output_dim = output_dim

        self.fc_net = FullyConnectedNetwork(input_dim=graph_output_dim + self.num_external_features,
                                            hidden_dims=fc_hidden_dims,
                                            output_dim=self.output_dim,
                                            activation=fc_activation,
                                            dropout=fc_dropout)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                external: Tensor,
                batch: Optional[Tensor]) -> Tensor:
        """
        Forward pass through the entire network.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            external (Tensor): External feature matrix with shape [num_samples, num_external_features].
            batch (Optional[Tensor]): The batch vector of shape [num_samples,] which assigns each element to a specific example.
            edge_attr (OptTensor): Edge feature matrix with shape [num_edges, num_edge_features]. Default is None.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        
        x = self.graph_model(x, edge_index, batch)
        x = torch.cat((x, external), dim=1)
        x = self.fc_net(x)
        return x
    