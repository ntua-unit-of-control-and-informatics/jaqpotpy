import unittest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool
from jaqpotpy.models.torch_geometric_models.graph_attention import GraphAttentionNetwork
import torch.nn as nn

class TestGraphAttentionNetwork(unittest.TestCase):
    "Test for the Functionality of Graph Attention Network"
    def setUp(self):
        self.input_dim = 16
        self.hidden_dims = [32, 64]
        self.heads = [2, 3]
        self.edge_dim = 8
        self.output_dim = 1
        self.activation = nn.ReLU()
        self.dropout = [0.5, 0.3]
        self.graph_norm = True
        self.pooling = 'mean'
        self.jittable = True
        self.batch_size = 5
        self.num_nodes = 20
        self.num_edges = 50
        # Initialize with all atributes
        self.network = GraphAttentionNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            heads=self.heads,
            edge_dim=self.edge_dim,
            output_dim=self.output_dim,
            activation=self.activation,
            dropout=self.dropout,
            graph_norm=self.graph_norm,
            pooling=self.pooling,
            jittable=self.jittable
        )

        # Initialize random tensors with graph specific dimensions
        self.x = torch.rand((self.num_nodes, self.input_dim))
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.edge_attr = torch.rand((self.num_edges, self.edge_dim))
        self.batch = torch.randint(0, self.batch_size, (self.num_nodes,))

    def test_forward_with_everything(self):
        "Test forward propagation of a graph attention network assuming every attribute is utilized"
        output = self.network(self.x, self.edge_index, self.batch, self.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_no_heads(self):
        "Test forward propagation correct dimensions without attention heads"
        block = GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

        output = block(self.x, self.edge_index, self.batch, self.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_no_edge_attr(self):
        "Test forward propagation correct dimensions without edge attributes"
        block = GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )
        
        output = block(self.x, self.edge_index, self.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_invalid_input_dim(self):
        "Test forward propagation using invalid input dimension"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim='invalid',
                hidden_dims=self.hidden_dims,
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

    def test_invalid_hidden_dims(self):
        "Test forward propagation with invalid hidden dimensions"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims='invalid',
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

    def test_invalid_heads(self):
        "Test forward propagation using invalid heads"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                heads='invalid',
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

    def test_invalid_activation(self):
        "Test forward propagation using invalid activation function"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation='invalid',
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

    def test_invalid_dropout(self):
        "Test forward propagation using invalid dropout probability"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout='invalid',
                graph_norm=self.graph_norm,
                pooling=self.pooling,
                jittable=self.jittable
            )

    def test_invalid_graph_norm(self):
        "Test forward propagation using invalid graph norm"
        with self.assertRaises(TypeError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm='invalid',
                pooling=self.pooling,
                jittable=self.jittable
            )
    #@unittest.skip('')
    def test_invalid_pooling(self):
        "Test forward propagation using invalid pooling aggregation"
        with self.assertRaises(ValueError):
            GraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                heads=self.heads,
                edge_dim=self.edge_dim,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout,
                graph_norm=self.graph_norm,
                pooling='invalid',
                jittable=self.jittable
            )

if __name__ == '__main__':
    unittest.main()