import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from jaqpotpy.models.torch_geometric_models.graph_transformer_network import GraphTransformerNetwork, GraphTransformerBlock
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem
from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset
import random

class TestGraphTransformerNetwork(unittest.TestCase):
    def setUp(self):
        """This is a test for GraphTransformerNetwork class"""
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
        featurizer.add_bond_feature('bond_type',[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
        featurizer.add_bond_feature('is_conjugated')
        self.dataset = SmilesGraphDataset(smiles, y, featurizer=featurizer)
        one_data = self.dataset[0]

        self.batch_size = 1
        self.num_nodes = one_data.x.shape[0]
        self.num_edges = one_data.edge_index.shape[1]
        self.input_dim = one_data.x.shape[1]
        self.edge_dim = one_data.edge_attr.shape[1]
        self.hidden_dims = [16, 32]
        self.heads =  random.randint(1, 4) 
        self.output_dim = 1
        self.x = one_data.x
        self.edge_index = one_data.edge_index
        self.edge_attr = one_data.edge_attr
        self.batch = one_data.batch
        self.y = one_data.y

    def test_initialization(self):
        "Test initializing a GraphTransformerNetwork object"
        model = GraphTransformerNetwork(input_dim = self.input_dim,
                                        hidden_dims =  self.hidden_dims,
                                        heads = self.heads,
                                        edge_dim = self.edge_dim,
                                        output_dim = self.output_dim)
        self.assertIsInstance(model, GraphTransformerNetwork)

    def test_forward_pass_nodes_only(self):
        "Test the forward pass of the model with a single instance as a batch"
        model = GraphTransformerNetwork(input_dim = self.input_dim,
                                        hidden_dims =  self.hidden_dims,
                                        heads = self.heads,
                                        output_dim = self.output_dim)
        data = Data(x=self.x, edge_index=self.edge_index, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_pass_node_and_edge_features(self):
        "Test the forward pass of the model with both node and edge features"
        model = GraphTransformerNetwork(input_dim = self.input_dim,
                                        hidden_dims =  self.hidden_dims,
                                        heads = self.heads,
                                        edge_dim = self.edge_dim,
                                        output_dim = self.output_dim)
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_pass_multiple_instances(self):
        "Test the forward pass of the model with multiple instances as a batch"
        # Test with node features only
        model = GraphTransformerNetwork(input_dim = self.input_dim,
                                        hidden_dims =  self.hidden_dims,
                                        heads = self.heads,
                                        output_dim = self.output_dim)
        data_list = [self.dataset[i] for i in range(5)]
        batch = Batch.from_data_list(data_list)
        output = model(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(output.shape, (5, self.output_dim))

        # Test with node and edge features
        model_with_edges = GraphTransformerNetwork(input_dim = self.input_dim,
                                                   hidden_dims =  self.hidden_dims,
                                                   heads = self.heads,
                                                   edge_dim = self.edge_dim,
                                                   output_dim = self.output_dim)
        output_with_edges = model_with_edges(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        self.assertEqual(output_with_edges.shape, (5, self.output_dim))

    def test_no_dropout(self):
        "Test the forward pass of the model with no dropout"
        model = GraphTransformerNetwork(self.input_dim, self.hidden_dims, self.heads, self.edge_dim, self.output_dim, dropout=0.0)
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_with_graph_norm(self):
        "Test the forward pass of the model with graph normalization"
        model = GraphTransformerNetwork(self.input_dim, self.hidden_dims, self.heads, self.edge_dim, self.output_dim, graph_norm=True)
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_different_activation(self):
        "Test the forward pass of the model with a different activation function"
        model = GraphTransformerNetwork(self.input_dim, self.hidden_dims, self.heads, self.edge_dim, self.output_dim, activation=nn.LeakyReLU())
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_pooling_function_add(self):
        "Test the pooling function with 'add' pooling"
        model = GraphTransformerNetwork(self.input_dim, self.hidden_dims, self.heads, self.edge_dim, self.output_dim, pooling='add')
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        pooled_output = model._pooling_function(data.x, data.batch)
        self.assertEqual(pooled_output.shape[0], self.batch_size)

    def test_pooling_function_max(self):
        "Test the pooling function with 'max' pooling"
        model = GraphTransformerNetwork(self.input_dim, self.hidden_dims, self.heads, self.edge_dim, self.output_dim, pooling='max')
        data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)
        pooled_output = model._pooling_function(data.x, data.batch)
        self.assertEqual(pooled_output.shape[0], self.batch_size)

    def test_graph_transformer_block(self):
        "Test GraphTransformerBlock class for the correct message passing dimensions"
        block = GraphTransformerBlock(self.input_dim, self.hidden_dims[0], self.heads, self.edge_dim)
        output = block(self.x, self.edge_index, self.batch, self.edge_attr)
        self.assertEqual(output.shape, (self.num_nodes, self.hidden_dims[0] * self.heads))

if __name__ == '__main__':
    unittest.main()