import numpy as np
import unittest
from rdkit import Chem
import os
import pandas as pd
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
#from jaqpotpy.cfg import config

class TestGraphFeaturizer(unittest.TestCase):
    def setUp(self):

        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, '../../test_data'))
        csv_file_path = os.path.join(test_data_dir, 'test_data_smiles_classification.csv')
        self.single_smiles_df = pd.read_csv(csv_file_path)
        self.smiles = self.single_smiles_df['SMILES']
        self.featurizer = SmilesGraphFeaturizer()

    def test_node_features(self):
        """Test the correct generation of node features and their shape"""
        self.featurizer.add_atom_characteristic('symbol', ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'])
        self.featurizer.add_atom_characteristic('degree', [0, 1, 2, 3, 4])
        self.featurizer.add_atom_characteristic('hybridization', [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP])
        self.featurizer.add_atom_characteristic('implicit_valence', [0, 1, 2, 3, 4])
        self.featurizer.add_atom_characteristic('formal_charge', [-1, 0, 1, 'UNK'])
        self.featurizer.add_atom_characteristic('is_aromatic')
        data = self.featurizer.featurize(self.smiles[0])

        self.assertEqual(len(self.featurizer.atom_allowable_sets.keys()), 6, 'Not all node features were parsed')
        self.assertEqual(data.x.shape[0], 37, 'Number of nodes is incorrect')
        self.assertEqual(data.x.shape[1], 32, 'Dimension of node feature vector is incorrect')

    def test_bond_features(self):
        """Test the correct generation of bond features and their shape"""
        self.featurizer.add_bond_characteristic('bond_type',[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
        self.featurizer.add_bond_characteristic('is_conjugated')

        data = self.featurizer.featurize(self.smiles[0])
        self.assertEqual(len(self.featurizer.bond_allowable_sets.keys()), 2, 'Not all bond features were parsed')
        self.assertEqual(data.edge_attr.shape[0], 80, 'Number of edges is incorrect')
        self.assertEqual(data.edge_attr.shape[1], 5, 'Dimension of edge feature vector is incorrect')

    def test_adjacency_matrix(self):
        """Test the correct generation of adjacency matrix and it's shape"""
        data = self.featurizer.featurize(self.smiles[0])
        self.assertEqual(data.edge_index.shape[0], 2, 'Needs to be 2 in order to match COO format')
        self.assertEqual(data.edge_index.shape[1], 80, 'Number of edges is incorrect')