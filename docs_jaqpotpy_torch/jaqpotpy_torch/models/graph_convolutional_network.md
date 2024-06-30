Module jaqpotpy_torch.models.graph_convolutional_network
========================================================
Author: Ioannis Pitoskas (jpitoskas@gmail.com)

Classes
-------

`GraphConvBlock(input_dim: int, hidden_dim: int, activation: torch.nn.modules.module.Module = ReLU(), dropout_probability: float = 0.5, graph_norm: bool = False, jittable: bool = True, *args, **kwargs)`
:   A single Graph Convolutional Block consisting of a GCNConv layer, an activation function,
    and a dropout layer. Optionally, a graph normalization layer can be applied.
    
    Attributes:
        hidden_layer (torch_geometric.nn.GCNConv): The GCNConv layer.
        graph_norm (bool): Whether to apply graph normalization.
        gn_layer (nn.Module or None): The graph normalisation layer.
        activation (nn.Module): Activation function to apply after the hidden layer.
        dropout (nn.Dropout): The dropout layer.
        jittable (bool): Whether to make the hidden module jittable.
    
    Args:
        input_dim (int): Dimension of the input node features.
        hidden_dim (int): Dimension of the hidden features.
        activation (nn.Module): Activation function to apply after the hidden layer. Default is nn.ReLU().
        dropout_probability (float): Dropout probability. Default is 0.5.
        graph_norm (bool): Whether to apply graph normalization. Default is False.
        jittable (bool): Whether to make the hidden module jittable. Default is True.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor]) ‑> torch.Tensor`
    :   Passes the input through the layer.
        
        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): Batch vector with shape [num_samples,] which assigns each element to a specific example.
        
        Returns:
            Tensor: Output tensor of the layer.

`GraphConvolutionalNetwork(input_dim: int, hidden_dims: Iterable[int], output_dim: int = 1, activation: torch.nn.modules.module.Module = ReLU(), dropout: Union[float, Iterable[float]] = 0.5, graph_norm: bool = False, pooling: str = 'mean', jittable: bool = True, *args, **kwargs)`
:   A Graph Convolutional Network consisting of multiple Graph Convolutional Blocks followed by a Fully Connected Layer.
    
    Attributes:
        input_dim (int): Dimension of the input node features.
        hidden_dims (Iterable[int]): Dimensions of the hidden layers.
        output_dim (int): Dimension of the network's output.
        activation (nn.Module): Activation function to apply after each hidden layer.
        dropout_probabilities (list): List of dropout probabilities after each hidden layer.
        graph_norm (bool): Whether to apply graph normalization.
        pooling (str): Type of pooling to apply ('mean', 'add', 'max').
        jittable (bool): Whether to make the hidden modules jittable.
    
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

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor]) ‑> torch.Tensor`
    :   Forward pass through the entire network.
        
        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (Optional[Tensor]): Batch vector with shape [num_samples,] which assigns each element to a specific example.
        
        Returns:
            Tensor: Output tensor after passing through the network.

`GraphConvolutionalNetworkWithExternal(graph_input_dim: int, num_external_features: int, graph_hidden_dims: Iterable[int], fc_hidden_dims: Iterable[int], graph_output_dim: int, output_dim: int = 1, graph_activation: torch.nn.modules.module.Module = ReLU(), graph_dropout: Union[float, Iterable[float]] = 0.5, graph_norm: bool = False, graph_pooling: str = 'mean', fc_dropout: Union[float, Iterable[float]] = 0.5, fc_activation: torch.nn.modules.module.Module = ReLU(), jittable: bool = True, *args, **kwargs)`
:   A Graph Convolutional Network that integrates external features.
    Combines the output from a Graph Convolutional Network with external features
    and processes them through a Fully Connected Network.
    
    Attributes:
        graph_model (jaqpotpy.jaqpotpy_torch.models.GraphConvolutionalNetwork): The graph network to treat the graph features.
        num_external_features (int): Number of external features.
        output_dim (int): Dimension of the network's output.
        fc_net (jaqpotpy.jaqpotpy_torch.models.FullyConnectedNetwork): The fc network to treat the external features.
    
    Args:
        graph_input_dim (int): Dimension of the input node features for the graph.
        num_external_features (int): Number of external features.
        graph_hidden_dims (Iterable[int]): Dimensions of the hidden layers in the graph.
        fc_hidden_dims (Iterable[int]): Dimensions of the hidden layers in the fully connected network.
        graph_output_dim (int): Dimension of the graph output features.
        output_dim (int): Dimension of the network's output. Default is 1.
        graph_activation (nn.Module): Activation function for graph layers. Default is nn.ReLU().
        graph_dropout (Union[float, Iterable[float]]): Dropout probability for graph layers. Default is 0.5
        graph_norm (bool): Whether to apply graph normalization in the graph layers. Default is False.
        graph_pooling (str): Type of pooling to apply in the graph layers ('mean', 'add', 'max'). Default is 'mean'.
        fc_dropout (Union[float, Iterable[float]]): Dropout probability for fully connected layers. Default is 0.5.
        fc_activation (nn.Module): Activation function for fully connected layers. Default is nn.ReLU().
        jittable (bool): Whether to make the hidden modules jittable. Default is True.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, edge_index: torch.Tensor, external: torch.Tensor, batch: Optional[torch.Tensor]) ‑> torch.Tensor`
    :   Forward pass through the entire network.
        
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            external (Tensor): External feature matrix with shape [num_samples, num_external_features].
            batch (Optional[Tensor]): The batch vector of shape [num_samples,] which assigns each element to a specific example.
            edge_attr (OptTensor): Edge feature matrix with shape [num_edges, num_edge_features]. Default is None.
        
        Returns:
            Tensor: Output tensor after passing through the network.