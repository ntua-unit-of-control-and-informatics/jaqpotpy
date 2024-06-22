"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

import torch.nn as nn
from typing import Optional, Iterable, Union
import torch.nn.init as init
from torch import Tensor


class FullyConnectedBlock(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.ReLU(),
                 dropout_probability: float = 0.5,
                #  norm: Optional[bool] = False,
                 *args,
                 **kwargs):
        
        super().__init__()
        
        
        self.fc = nn.Linear(input_dim, output_dim)            
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
        
    
    def forward(self,
                x: Tensor) -> Tensor:

        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class FullyConnectedNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 output_dim: int = 1,
                 activation: nn.Module = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                #  norm: Optional[bool] = False,
                 *args,
                 **kwargs):
        
        super().__init__()

        # Input types check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        
        if not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be an Iterable")
        
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
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout

        if self.hidden_dims == []:
            self.fc_layers = None
            self.fc_head = nn.Linear(input_dim, output_dim)
        else:
            self.fc_layers = nn.ModuleList()
            fc_layer = FullyConnectedBlock(input_dim, self.hidden_dims[0],
                                        activation=activation,
                                        dropout_probability=self.dropout_probabilities[0])
            self.fc_layers.append(fc_layer)
            

            
            for i in range(len(hidden_dims) - 1):
                fc_layer = FullyConnectedBlock(self.hidden_dims[i], hidden_dims[i+1],
                                                activation=activation,
                                                dropout_probability=self.dropout_probabilities[i])
                self.fc_layers.append(fc_layer)
            
            self.fc_head = nn.Linear(hidden_dims[-1], output_dim)
        
    
    def forward(self,
                x: Tensor) -> Tensor:

        if self.fc_layers is not None:
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
        
        x = self.fc_head(x)

        return x
