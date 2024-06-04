import torch
import torch.nn as nn
from typing import Optional, Iterable, Union
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphNorm #, BatchNorm, GraphSizeNorm, InstanceNorm, LayerNorm, Set2Set
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init
from torch import Tensor

from .fully_connected_network import FullyConnectedNetwork


class GraphSAGEBlock(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout_probability: float = 0.5,
                 graph_norm: Optional[bool] = False,
                 jittable: Optional[bool] = True,
                 *args,
                 **kwargs):
        
        super(GraphSAGEBlock, self).__init__()

        self.jittable = jittable

        self.hidden_layer = SAGEConv(input_dim, hidden_dim)
        
        if self.jittable:
            self.hidden_layer = self.hidden_layer.jittable()

        self.graph_norm = graph_norm
        if self.graph_norm:
            self.gn_layer = GraphNorm(hidden_dim)
            # self.gn_layer = BatchNorm(hidden_dim)
            # self.gn_layer = GraphSizeNorm()
            # self.gn_layer = InstanceNorm(hidden_dim)
            # self.gn_layer = LayerNorm(hidden_dim)
        else:
            self.gn_layer = None

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor]) -> Tensor:

        x = self.hidden_layer(x, edge_index)
        if self.gn_layer is not None:
            x = self.gn_layer(x, batch)
            # x = self.gn_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
 
        return x
                


class GraphSAGENetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 output_dim: Optional[int] = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: Optional[bool] = False,
                 pooling: Optional[str] = 'mean',
                 jittable: Optional[bool] = True,
                 *args,
                 **kwargs):
    
        super(GraphSAGENetwork, self).__init__()
                
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
        graph_layer = GraphSAGEBlock(input_dim, hidden_dims[0],
                                     activation=activation,
                                     dropout_probability=self.dropout_probabilities[0],
                                     graph_norm=graph_norm,
                                     jittable=jittable)
        self.graph_layers.append(graph_layer)
        

        for i in range(len(hidden_dims) - 1):
            graph_layer = GraphSAGEBlock(hidden_dims[i], hidden_dims[i+1],
                                         activation=activation,
                                         dropout_probability=self.dropout_probabilities[i],
                                         graph_norm=graph_norm,
                                         jittable=jittable)
            self.graph_layers.append(graph_layer)
        
        # self.aggr = Set2Set(hidden_dims[-1], processing_steps=4)
        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor]) -> Tensor:

        x = self._forward_graph(x, edge_index, batch)
        x = self.fc(x)
        return x
    

    def _forward_graph(self,
                       x: Tensor,
                       edge_index: Tensor,
                       batch: Optional[Tensor]) -> Tensor:
    
        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index, batch)
        # x = self.aggr(x, batch)
        x = self._pooling_function(x, batch)
        return x
    

    def _pooling_function(self,
                          x: Tensor, 
                          batch: Optional[Tensor]) -> Tensor:

        if self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")


class GraphSAGENetworkWithExternal(nn.Module):
    def __init__(self,
                 graph_input_dim: int,
                 num_external_features: int,
                 graph_hidden_dims: Iterable[int],
                 fc_hidden_dims: Iterable[int],
                 graph_output_dim: int,
                 output_dim: Optional[int] = 1,
                 graph_activation: Optional[nn.Module] = nn.ReLU(),
                 graph_dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: Optional[bool] = False,
                 graph_pooling: Optional[str] = 'mean',
                 fc_dropout: Union[float, Iterable[float]] = 0.5,
                 fc_activation: Optional[nn.Module] = nn.ReLU(),
                 jittable: Optional[bool] = True,
                 *args,
                 **kwargs):
        
        super().__init__()

        if not isinstance(num_external_features, int):
            raise TypeError("num_external_features must be of type int")
        
    
        self.graph_model = GraphSAGENetwork(input_dim=graph_input_dim,
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
        
        x = self.graph_model(x, edge_index, batch)
        x = torch.cat((x, external), dim=1)
        x = self.fc_net(x)
        return x
    