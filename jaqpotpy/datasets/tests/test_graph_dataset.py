"""
Tests for jaqpotpy Datasets.
"""

import os
import unittest
import pandas as pd
from jaqpotpy.datasets.graph_pyg_dataset import SmilesGraphDataset
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from rdkit import Chem
import torch


class TestSmilesGraphDataset(unittest.TestCase):
    """TestSmilesGraphDataset is a unit tester for the Pytorch Geometric Graph Datasets"""

    def setUp(self):
        # Sample data
        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, "../../test_data"))
        csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_classification.csv"
        )
        self.single_smiles_df = pd.read_csv(csv_file_path)
        self.smiles = self.single_smiles_df["SMILES"][:3]
        self.y = [0, 0.5, 5]
        self.featurizer = SmilesGraphFeaturizer()
        # self.dataset = SmilesGraphDataset(smiles = self.smiles, y=self.y, featurizer=self.featurizer)

    def test_len_functionality(self):
        """Check __len__ functionality of dataset"""
        dataset = SmilesGraphDataset(smiles=self.smiles, y=self.y)
        self.assertEqual(len(dataset), 3)

    def test_getitem_functionality(self):
        "Test __getitem__ functionality of pyg"
        dataset = SmilesGraphDataset(smiles=self.smiles, y=self.y)
        data = dataset[0]
        # Correct smiles in data object
        self.assertEqual(data.smiles, self.smiles[0])
        mol = Chem.MolFromSmiles(data.smiles)
        # Correct node features matrix in data object
        self.assertEqual(data.x.shape[0], mol.GetNumAtoms())
        self.assertEqual(data.x.shape[1], 28)

    def test_invalid_smiles(self):
        "Test Giving an invalid Smiles String"
        invalid_smiles = ["C1CCCCC1C", "InvalidSMILES", "CCO"]
        dataset = SmilesGraphDataset(smiles=invalid_smiles, y=self.y)
        with self.assertRaises(Exception):
            dataset[1]

    def test_empty_inputs(self):
        "Given no inputs __getitem__ shouldn't have access to the index"
        dataset = SmilesGraphDataset(smiles=[], y=[])
        self.assertEqual(len(dataset), 0)
        with self.assertRaises(IndexError):
            dataset[0]

    def test_precompute_featurization(self):
        "Test precomputing the featurization before __getitem__"
        dataset = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        dataset.precompute_featurization()
        # This checks that all instances are featurized before accessing them woith __getitem__
        self.assertEqual(len(dataset.precomputed_features), len(self.smiles))
        self.assertEqual(dataset[0].smiles, self.smiles[0])

    def test_featurization_consistency(self):
        "Ensure there is no randomness in the creation of matrices"
        dataset1 = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        dataset2 = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        dataset1.precompute_featurization()
        dataset2.precompute_featurization()
        for i in range(len(dataset1)):
            self.assertEqual(dataset1[i].smiles, dataset2[i].smiles)
            self.assertTrue(torch.equal(dataset1[i].x, dataset2[i].x))
            self.assertTrue(torch.equal(dataset1[i].edge_index, dataset2[i].edge_index))
            self.assertTrue(torch.equal(dataset1[i].edge_attr, dataset2[i].edge_attr))

    def main_functionality_with_user_inputs(self):
        """Test the correct creation of the dataset that is going to be used most frequently
        with user inputs"""
        self.featurizer.add_atom_feature("symbol", ["C", "O", "N", "UNK"])
        self.featurizer.add_atom_feature("degree", [0, 1, 2, 3, 4])
        self.featurizer.add_bond_feature(
            "bond_type",
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        )
        self.featurizer.add_bond_feature("is_conjugated")
        dataset = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        dataset.precompute_featurization()
        for i, data in enumerate(dataset):
            smiles = self.smiles[i]
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            # Node features matrix
            self.assertEqual(
                data.x.shape,
                (num_atoms, len(self.featurizer.get_atom_feature_labels())),
            )
            # Adjacency matrix
            self.assertEqual(
                data.edge_index.shape[1], num_bonds * 2
            )  # each bond is represented twice in edge_index
            # Edge features matrix
            self.assertEqual(
                data.edge_attr.shape,
                (num_bonds * 2, len(self.featurizer.get_bond_feature_labels())),
            )
            # Check that the features are not empty
            self.assertTrue(torch.is_tensor(data.x) and data.x.numel() > 0)
            self.assertTrue(
                torch.is_tensor(data.edge_index) and data.edge_index.numel() > 0
            )
            self.assertTrue(
                torch.is_tensor(data.edge_attr) and data.edge_attr.numel() > 0
            )

    def test_get_num_node_features(self):
        "Test number of nodes features that is used as input to a graph"
        self.featurizer.add_atom_feature("symbol", ["C", "O", "N", "UNK"])
        self.featurizer.add_atom_feature("degree", [0, 1, 2, 3, 4])
        dataset = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        self.assertEqual(dataset.get_num_node_features(), 9)

    def test_get_num_edge_features(self):
        "Test number of edge features that is used as input to a graph"
        self.featurizer.add_bond_feature(
            "bond_type",
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        )
        self.featurizer.add_bond_feature("is_conjugated")
        dataset = SmilesGraphDataset(
            smiles=self.smiles, y=self.y, featurizer=self.featurizer
        )
        self.assertEqual(dataset.get_num_edge_features(), 5)


if __name__ == "__main__":
    unittest.main()
