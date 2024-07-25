"""
Author: Ioannis Pitoskas (jpitoskas@gmail.com)
"""

import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Iterable, Union
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GraphNorm #, BatchNorm, GraphSizeNorm, InstanceNorm, LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init
from torch import Tensor
from torch_geometric.typing import OptTensor

from .fully_connected_network import FullyConnectedNetwork


class GraphTransformerBlock(nn.Module):
    """
    A single Graph Transformer Block consisting of a TransformerConv layer, an activation function,
    and a dropout layer. Optionally, a graph normalization layer can be applied.

    Attributes:
        hidden_layer (torch_geometric.nn.TransformerConv): The TransformerConv layer.
        graph_norm (bool): Whether to apply graph normalization.
        gn_layer (nn.Module or None): The graph normalisation layer.
        activation (nn.Module): Activation function to apply after the hidden layer.
        dropout (nn.Dropout): The dropout layer.
        jittable (bool): Whether to make the hidden module jittable.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 heads: int = 1,
                 edge_dim: Optional[int] = None,
                 activation: nn.Module = nn.ReLU(),
                 dropout_probability: float = 0.5,
                 graph_norm: bool = False,
                 jittable: bool = True,
                 *args,
                 **kwargs):
        """
        Args:
            input_dim (int): Dimension of the input node features.
            hidden_dim (int): Dimension of the hidden features.
            heads (int): Number of attention heads. Default is 1.
            edge_dim (Optional[int]): Dimension of the edge features if any. Default is None.
            activation (nn.Module): Activation function to apply after the hidden layer. Default is nn.ReLU().
            dropout_probability (float): Dropout probability. Default is 0.5.
            graph_norm (bool): Whether to apply graph normalization. Default is False.
            jittable (bool): Whether to make the hidden module jittable. Default is True.
        """
        super(GraphTransformerBlock, self).__init__()
        
        self.jittable = jittable

        self.hidden_layer = TransformerConv(input_dim, hidden_dim, heads, edge_dim=edge_dim)

        if jittable:
            self.hidden_layer = self.hidden_layer.jittable()

        self.graph_norm = graph_norm
        if self.graph_norm:
            self.gn_layer = GraphNorm(hidden_dim*heads)
        else:
            self.gn_layer = None
            
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor],
                edge_attr: OptTensor = None) -> Tensor:
        """
        Passes the input through the layer.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): The batch vector of shape [num_samples,] which assigns each element to a specific example.
            edge_attr (OptTensor): Edge feature matrix with shape [num_edges, num_edge_features]. Default is None.

        Returns:
            Tensor: Output tensor of the layer.
        """

        x = self.hidden_layer(x, edge_index, edge_attr=edge_attr)
        if self.gn_layer is not None:
            x = self.gn_layer(x, batch)
        x = self.activation(x)
        x = self.dropout(x)

        return x



class GraphTransformerNetwork(nn.Module):
    """
    A Graph Transformer Network consisting of multiple Graph Transformer Blocks followed by a Fully Connected Layer.

    Attributes:
        input_dim (int): Dimension of the input node features.
        hidden_dims (Iterable[int]): Dimensions of the hidden layers.
        heads (Iterable[int]): List of number of attention heads for each layer.
        edge_dim (int or None): Dimension of the edge features if any.
        output_dim (int): Dimension of the network's output.
        activation (nn.Module): Activation function to apply after each hidden layer.
        dropout_probabilities (list): List of dropout probabilities after each hidden layer.
        graph_norm (bool): Whether to apply graph normalization.
        pooling (str): Type of pooling to apply ('mean', 'add', 'max').
        jittable (bool): Whether to make the hidden modules jittable.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 heads: Union[int, Iterable[int]] = 1,
                 edge_dim: Optional[int] = None,
                 output_dim: int = 1,
                 activation: nn.Module = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: bool = False,
                 pooling: str = 'mean',
                 jittable: bool = True,
                 *args,
                 **kwargs):
        torch.manual_seed(42)
        np.random.seed(42)  
        """
        Args:
            input_dim (int): Dimension of the input node features.
            hidden_dims (Iterable[int]): Dimensions of the hidden layers.
            heads (Union[int, Iterable[int]]): Number of attention heads for each layer. Default is 1.
            edge_dim (Optional[int]): Dimension of the edge features if any. Default is None.
            output_dim (int): Dimension of the network's output. Default is 1.
            activation (nn.Module): Activation function to apply after each hidden layer. Default is nn.ReLU().
            dropout (Union[float, Iterable[float]]): Dropout probability for each layer. Default is 0.5.
            graph_norm (bool): Whether to apply graph normalization. Default is False.
            pooling (str): Type of pooling to apply ('mean', 'add', 'max'). Default is 'mean'.
            jittable (bool): Whether to make the hidden modules jittable. Default is True.
        """
        super(GraphTransformerNetwork, self).__init__()
                
        # Input types check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        
        if not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be an Iterable")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if not all(isinstance(hidden_dim, int) for hidden_dim in hidden_dims):
            raise TypeError("hidden_dims must only contain integers")
        
        if not (isinstance(heads, int) or isinstance(heads, Iterable)):
            raise TypeError("heads must be either of type int or Iterable")
        if isinstance(heads, int):
            if heads <= 0:
                raise ValueError("heads must be between greater than 0")
        if isinstance(heads, Iterable):
            for item in heads:
                if not isinstance(item, int):
                    raise TypeError("heads list must only contain integers")
                if item <= 0:
                    raise ValueError("Each element in the heads list must be between greater than 0")
            if len(heads) != len(hidden_dims):
                raise ValueError("hidden_dims and heads must be of same size") 

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
        self.heads = [heads]*len(hidden_dims) if isinstance(heads, int) else heads
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout
        self.graph_norm = graph_norm
        self.pooling = pooling
        self.jittable = jittable


        self.graph_layers = nn.ModuleList() 
        graph_layer = GraphTransformerBlock(input_dim, hidden_dims[0],
                                          heads=self.heads[0],
                                          edge_dim=edge_dim,
                                          activation=activation,
                                          dropout_probability=self.dropout_probabilities[0],
                                          graph_norm=graph_norm,
                                          jittable=jittable)
        self.graph_layers.append(graph_layer)

        
        for i in range(len(hidden_dims) - 1):
            graph_layer = GraphTransformerBlock(hidden_dims[i]*self.heads[i], hidden_dims[i+1],
                                              heads=self.heads[i+1],
                                              edge_dim=edge_dim,
                                              activation=activation,
                                              dropout_probability=self.dropout_probabilities[i],
                                              graph_norm=graph_norm,
                                              jittable=jittable)
            self.graph_layers.append(graph_layer)
    
        
        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1]*self.heads[-1], output_dim)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
    
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor],
                edge_attr: OptTensor = None) -> Tensor:
        """
        Forward pass through the entire network.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): The batch vector of shape [num_samples,] which assigns each element to a specific example.
            edge_attr (OptTensor): Edge feature matrix with shape [num_edges, num_edge_features]. Default is None.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        
        x = self._forward_graph(x, edge_index, batch, edge_attr=edge_attr)
        x = self.fc(x)
        return x
    

    def _forward_graph(self,
                       x: Tensor,
                       edge_index: Tensor,
                       batch: Optional[Tensor],
                       edge_attr: OptTensor = None) -> Tensor:
        """
        Helper method for the forward pass through graph layers and pooling.
        """

        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = self._pooling_function(x, batch)
        return x
    

    def _pooling_function(self,
                          x: Tensor, 
                          batch: Optional[Tensor]) -> Tensor:
        """
        Helper method for the pooling operation.
        
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, ?].
            batch (Optional[Tensor]): The batch vector of shape [num_samples,] which assigns each element to a specific example.

        Returns:
            Tensor: Output tensor after batch-level aggregation.
        """

        if self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")


class GraphTransformerNetworkWithExternal(nn.Module):
    """
    A Graph Transformer Network that integrates external features.
    Combines the output from a Graph Transformer Network with external features
    and processes them through a Fully Connected Network.
    
    Attributes:
        graph_model (jaqpotpy.jaqpotpy_torch.models.GraphTransformerNetwork): The graph network to treat the graph features.
        num_external_features (int): Number of external features.
        output_dim (int): Dimension of the network's output.
        fc_net (jaqpotpy.jaqpotpy_torch.models.FullyConnectedNetwork): The fc network to treat the external features.
    """

    def __init__(self,
                 graph_input_dim: int,
                 num_external_features: int,
                 graph_hidden_dims: Iterable[int],
                 fc_hidden_dims: Iterable[int],
                 graph_output_dim: int,
                 output_dim: int = 1,
                 graph_heads: Union[int, Iterable[int]] = 1,
                 graph_edge_dim: Optional[int] = None,
                 graph_activation: nn.Module = nn.ReLU(),
                 graph_dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: bool = False,
                 graph_pooling: str = 'mean',
                 fc_dropout: Union[float, Iterable[float]] = 0.5,
                 fc_activation: nn.Module = nn.ReLU(),
                 jittable: bool = True,
                 *args,
                 **kwargs):
        """
        Args:
            graph_input_dim (int): Dimension of the input node features for the graph.
            num_external_features (int): Number of external features.
            graph_hidden_dims (Iterable[int]): Dimensions of the hidden layers in the graph.
            fc_hidden_dims (Iterable[int]): Dimensions of the hidden layers in the fully connected network.
            graph_output_dim (int): Dimension of the graph output features.
            output_dim (int): Dimension of the network's output. Default is 1.
            graph_heads (Union[int, Iterable[int]]): Number of attention heads for each graph layer. Default is 1.
            graph_edge_dim (Optional[int]): Dimension of the edge features if any. Default is None.
            graph_activation (nn.Module): Activation function for graph layers. Default is nn.ReLU().
            graph_dropout (Union[float, Iterable[float]]): Dropout probability for graph layers. Default is 0.5
            graph_norm (bool): Whether to apply graph normalization in the graph layers. Default is False.
            graph_pooling (str): Type of pooling to apply in the graph layers ('mean', 'add', 'max'). Default is 'mean'.
            fc_dropout (Union[float, Iterable[float]]): Dropout probability for fully connected layers. Default is 0.5.
            fc_activation (nn.Module): Activation function for fully connected layers. Default is nn.ReLU().
            jittable (bool): Whether to make the hidden modules jittable. Default is True.
        """
        super().__init__()

        if not isinstance(num_external_features, int):
            raise TypeError("num_external_features must be of type int")

        self.graph_model = GraphTransformerNetwork(input_dim=graph_input_dim,
                                                   hidden_dims=graph_hidden_dims,
                                                   heads=graph_heads,
                                                   edge_dim=graph_edge_dim,
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
                batch: Optional[Tensor],
                edge_attr: OptTensor = None) -> Tensor:
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

        x = self.graph_model(x, edge_index, batch, edge_attr=edge_attr)
        x = torch.cat((x, external), dim=1)
        x = self.fc_net(x)
        return x