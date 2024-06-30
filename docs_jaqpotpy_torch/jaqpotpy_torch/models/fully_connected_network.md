Module jaqpotpy_torch.models.fully_connected_network
====================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`FullyConnectedBlock(input_dim: int, output_dim: int, activation: torch.nn.modules.module.Module = ReLU(), dropout_probability: float = 0.5, *args, **kwargs)`
:   A single fully connected block consisting of a linear layer, an activation function,
    and a dropout layer. Xavier initialization is applied to the weights.
    
    Attributes:
        fc (nn.Linear): The Fully Connected linear layer.
        activation (nn.Module): The activation function.
        dropout (nn.Dropout): The dropout layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        activation (nn.Module): Activation function to apply after the linear layer.
        dropout_probability (float): Dropout probability.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Passes the input through the layer.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying the fully connected block.

`FullyConnectedNetwork(input_dim: int, hidden_dims: Iterable[int], output_dim: int = 1, activation: torch.nn.modules.module.Module = ReLU(), dropout: Union[float, Iterable[float]] = 0.5, *args, **kwargs)`
:   A fully connected neural network consisting of multiple FullyConnectedBlock layers
    and a final linear projection. The network can be customized with various activation
    functions and dropout probabilities for each layer.
    
    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        hidden_dims (Iterable[int]): Dimensions of the hidden layers.
        activation (nn.Module): Activation function to apply after each hidden layer.
        dropout_probabilities (list): List of dropout probabilities after each FC layer
        fc_layers (nn.ModuleList): ModuleList of FullyConnectedBlock layers of the network
        fc_head (nn.Linear): The final linear projection layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (Iterable[int]): Dimensions of the hidden layers.
        output_dim (int): Dimension of the output features. Default is 1.
        activation (nn.Module): Activation function to apply after each hidden layer. Default is nn.ReLU().
        dropout (Union[float, Iterable[float]]): Dropout probabilities for each hidden layer. Default is 0.5.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Passes the input through the network layers.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after passing through the network.