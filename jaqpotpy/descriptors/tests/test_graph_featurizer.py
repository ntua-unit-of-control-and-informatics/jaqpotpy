import numpy as np
import unittest
from rdkit import Chem
import os
import pandas as pd
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
import torch

class TestGraphFeaturizer(unittest.TestCase):
    """This is a Test Class that ensures that featurizing a molecule works correctly as expected"""
    def setUp(self):
        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, '../../test_data'))
        csv_file_path = os.path.join(test_data_dir, 'test_data_smiles_classification.csv')
        self.single_smiles_df = pd.read_csv(csv_file_path)
        self.smile = self.single_smiles_df['SMILES'][0]
        self.mol = Chem.MolFromSmiles(self.smile)
        self.featurizer = SmilesGraphFeaturizer()
    
    def _calc_node_dims(self):
        "Helper to calculate the correct node feature dimensions"
        node_dims = 0
        for value in self.featurizer.atom_allowable_sets.values():
            if value is not None:
                node_dims+= len(value)
            else:
                node_dims+=1
        return node_dims
    
    def _calc_edge_dims(self):
        "Helper to calculate the correct edge feature dimensions"
        edge_dims = 0
        for value in self.featurizer.bond_allowable_sets.values():
            if value is not None:
                edge_dims+= len(value)
            else:
                edge_dims+=1
        return edge_dims

    def test_supported_atom_features(self):
        "Test all the possible names of node features that are supported in the class"
        atom_features = self.featurizer.get_supported_atom_features()
        expected_atom_features = [
        "symbol",
        "degree",
        "total_degree",
        "formal_charge",
        "num_radical_electrons",
        "hybridization",
        "is_aromatic",
        "is_in_ring",
        "total_num_hs",
        "num_explicit_hs",
        "num_implicit_hs",
        "_ChiralityPossible",
        "isotope",
        "total_valence",
        "explicit_valence",
        "implicit_valence",
        "chiral_tag",
        "mass"
    ]
        for feature in expected_atom_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, atom_features)

    def test_supported_bond_features(self):
        "Test all the possible names of edge features that are supported"
        bond_features = self.featurizer.get_supported_bond_features()
        expected_bond_features = [
        "bond_type",
        "is_conjugated",
        "is_in_ring",
        "stereo"
    ]
        for feature in expected_bond_features:
            with self.subTest(feature=feature):
                self.assertIn(feature, bond_features)

    def test_default_featurizer(self):
        "Tests the default features of the class. Ensures that correct dtype and dimensions are created for node and edge_features"
        self.featurizer.set_default_config()
        atom_features, bond_features = self.featurizer.extract_molecular_features(self.mol)
        # Dtype assertion
        self.assertIsInstance(atom_features, torch.Tensor)
        # Number of nodes assertion
        self.assertEqual(atom_features.shape[0], self.mol.GetNumAtoms())
        # Number of node features assertion
        node_dims = self._calc_node_dims()
        self.assertEqual(atom_features.shape[1],  node_dims)
        if self.featurizer.include_edge_features:
            # Dtype assertion
            self.assertIsInstance(bond_features, torch.Tensor)
            edge_dims = self._calc_edge_dims()
            self.assertEqual(bond_features.shape[1], edge_dims)

    def test_user_creating_node_features(self):
        "Test the correct generation of node features matrix and its shape"
        self.featurizer.add_atom_feature('symbol', ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'])
        self.featurizer.add_atom_feature('degree', [0, 1, 2, 3, 4])
        self.featurizer.add_atom_feature('hybridization', [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP])
        self.featurizer.add_atom_feature('implicit_valence', [0, 1, 2, 3, 4])
        self.featurizer.add_atom_feature('formal_charge')
        self.featurizer.add_atom_feature('is_aromatic')
        node_dims = self._calc_node_dims()
        data = self.featurizer.featurize(self.smile)
        # Number of nodes
        self.assertEqual(data.x.shape[0], self.mol.GetNumAtoms(), 'Number of nodes is incorrect')
        # Number of node features
        self.assertEqual(data.x.shape[1], node_dims, 'Dimension of node feature vector is incorrect')
        # Dtype assertion
        self.assertIsInstance(data.x, torch.Tensor)

    def test_user_creating_bond_features(self):
        """Test the correct generation of bond features and their shape"""
        self.featurizer.add_bond_feature('bond_type',[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
        self.featurizer.add_bond_feature('is_conjugated')
        edge_dims = self._calc_edge_dims()
        data = self.featurizer.featurize(self.smile)
        # The number edges * 2
        self.assertAlmostEqual(data.edge_attr.shape[0], self.mol.GetNumBonds()*2)
        # Number of edge_features 
        self.assertEqual(data.edge_attr.shape[1], edge_dims, 'Dimension of edge feature vector is incorrect')
        # Dtype assertion
        self.assertIsInstance(data.edge_attr, torch.Tensor)

    def test_user_creating_node_and_bond_features(self):
        """Test the correct generation of node and bond features simultaneously"""
        self.featurizer.add_atom_feature('symbol', ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'])
        self.featurizer.add_atom_feature('degree', [0, 1, 2, 3, 4])
        self.featurizer.add_bond_feature('bond_type',[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
        self.featurizer.add_bond_feature('is_conjugated')
        node_dims = self._calc_node_dims()
        edge_dims = self._calc_edge_dims()
        data = self.featurizer.featurize(self.smile)
        # Nodes
        self.assertEqual(data.x.shape[0], self.mol.GetNumAtoms(), 'Number of nodes is incorrect')
        # Number of node features
        self.assertEqual(data.x.shape[1], node_dims, 'Dimension of node feature vector is incorrect')
        # Dtype assertion
        self.assertIsInstance(data.x, torch.Tensor)
        # Edges
        self.assertAlmostEqual(data.edge_attr.shape[0], self.mol.GetNumBonds()*2)
        # Number of edge_features 
        self.assertEqual(data.edge_attr.shape[1], edge_dims, 'Dimension of edge feature vector is incorrect')
        # Dtype assertion
        self.assertIsInstance(data.edge_attr, torch.Tensor)

    def test_correct_adjacency_matrix_dimensions(self):
        "Test the correct generation of adjacency matrix and it's shape"
        data = self.featurizer.featurize(self.smile)
        self.assertEqual(data.edge_index.shape[0], 2, 'Needs to be 2 in order to match COO format')
        self.assertEqual(data.edge_index.shape[1], self.mol.GetNumBonds()*2, 'Number of edges is incorrect')

    def test_unsupported_atom_feature(self):
        "Test that adding an unsupported atom feature raises a ValueError"
        unsupported_feature = 'unsupported_feature'
        with self.assertRaises(ValueError):
            self.featurizer.add_atom_feature(unsupported_feature)

    def test_unsupported_bond_feature(self):
        "Test that adding an unsupported bond feature raises a ValueError"
        unsupported_feature = 'unsupported_feature'
        with self.assertRaises(ValueError):
            self.featurizer.add_bond_feature(unsupported_feature)

if __name__ == '__main__':
    unittest.main()