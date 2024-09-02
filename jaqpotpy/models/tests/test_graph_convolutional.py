import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from jaqpotpy.models.torch_geometric_models.graph_convolutional_network import GraphConvolutionalNetwork, GraphConvBlock, GraphConvolutionalNetworkWithExternal
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem
from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset
import random

class TestGraphConvolutionalNetwork(unittest.TestCase):
    def setUp(self):
        """This is a test for GraphConvolutionalNetwork class"""
        smiles = [
        'COc1cccc(S(=O)(=O)Cc2cc(C(=O)NO)no2)c1',
        'O=C(N[C@H](CO)[C@H](O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl',
        'CC(=O)NC[C@H]1CN(c2ccc3c(c2F)CCCCC3=O)C(=O)O1',
        'CC(C)(C)OC(=O)C=C',
        'CC1(CC(CC(C1)(C)CN=C=O)N=C=O)C',
        'C1=CN=CN1']
        y = [0, 1, 1, 0, 1, 1]
        featurizer = SmilesGraphFeaturizer()
        featurizer.add_atom_feature('symbol', ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'])
        featurizer.add_atom_feature('degree', [0, 1, 2, 3, 4])
        self.dataset = SmilesGraphDataset(smiles, y, featurizer=featurizer)
        one_data = self.dataset[0]

        self.batch_size = 1
        self.num_nodes = one_data.x.shape[0]
        self.num_edges = one_data.edge_index.shape[1]
        self.input_dim = one_data.x.shape[1]
        self.hidden_dims = [16, 32]
        self.output_dim = 1
        self.x = one_data.x
        self.edge_index = one_data.edge_index
        self.batch = one_data.batch
        self.y = one_data.y

    def test_initialization(self):
        "Test initializing a GraphConvolutionalNetwork object"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim)
        self.assertIsInstance(model, GraphConvolutionalNetwork)

    def test_forward_pass(self):
        "Test the forward pass of the model with a single instance as a batch"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim)
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_pooling_function(self):
        "Test pooling function functionality ensuring 2D tensor is flattened"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim, pooling='mean')
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        pooled_output = model._pooling_function(data.x, data.batch)
        self.assertEqual(pooled_output.shape[0], self.batch_size)

    def test_graph_conv_block(self):
        "Test GraphConvBlock class for the correct Message Passing dimensions"
        block = GraphConvBlock(self.input_dim, self.hidden_dims[0])
        output = block(self.x, self.edge_index, self.batch)
        self.assertEqual(output.shape, (self.num_nodes, self.hidden_dims[0]))

    def test_forward_pass_multiple_instances(self):
        "Test the forward pass of the model with multiple instances as a batch"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim)
        data_list = [self.dataset[i] for i in range(5)]
        batch = Batch.from_data_list(data_list)
        output = model(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(output.shape, (5, self.output_dim))

    def test_no_dropout(self):
        "Test the forward pass of the model with no dropout"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim, dropout=0.0)
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_with_graph_norm(self):
        "Test the forward pass of the model without graphnorm"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim, graph_norm=True)
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_different_activation(self):
        "Test the forward pass of the model without the default activation ReLU"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim, activation=nn.LeakyReLU())
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_not_default_pooling_function(self):
        "Test the forward pass of the model without the default pooling function"
        model = GraphConvolutionalNetwork(self.input_dim, self.hidden_dims, self.output_dim, pooling='add')
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        pooled_output = model._pooling_function(data.x, data.batch)
        self.assertEqual(pooled_output.shape[0], self.batch_size)

if __name__ == '__main__':
    unittest.main()
