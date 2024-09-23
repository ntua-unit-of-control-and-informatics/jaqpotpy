import unittest
from rdkit import Chem
import torch
from torch_geometric.data import Data
from jaqpotpy.descriptors.graph import SmilesGraphFeaturizer


class TestSmilesGraphFeaturizer(unittest.TestCase):
    """This is testing the main functionalities of the SmilesGraphFeaturizer class focused on node features and adjacency matrix.(ONNX supported)"""

    def setUp(self):
        """Initialize the featurizer with default configuration before each test."""
        self.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.atoms = len(Chem.MolFromSmiles(self.smiles).GetAtoms())
        self.bonds = len(Chem.MolFromSmiles(self.smiles).GetBonds())
        self.feature_symbols = ["B", "Br", "C", "F", "N"]
        self.feature_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

    def test_correct_matrix_shapes(self):
        """Test the shapes of the node features and adjacency matrix."""
        featurizer = SmilesGraphFeaturizer()
        featurizer.add_atom_feature(
            "symbol",
            self.feature_symbols,
        )
        data = featurizer.featurize(self.smiles)
        # Ensure correct node feature dimensions
        self.assertEqual(data.x.shape[1], len(self.feature_symbols))
        self.assertEqual(data.x.shape[0], self.atoms)
        # Ensure correct adjacency matrix dimensions
        self.assertEqual(data.edge_index.shape[0], 2)  # COO format
        self.assertEqual(data.edge_index.shape[1], self.bonds * 2)

    def test_add_unsupported_atom_feature(self):
        """Test adding an unsupported atom feature as key raises ValueError."""
        featurizer = SmilesGraphFeaturizer()
        # Assert that only supported features can be added
        with self.assertRaises(ValueError):
            featurizer.add_atom_feature(
                "symbolism",
                ["something", "randomn"],
            )

    def test_correct_num_node_features(self):
        """Test the correct number of node features."""
        featurizer = SmilesGraphFeaturizer()
        featurizer.add_atom_feature(
            "symbol",
            self.feature_symbols,
        )
        featurizer.add_atom_feature(
            "formal_charge",
        )
        num_node_features = featurizer.get_num_node_features()
        expected = len(self.feature_symbols) + 1  # 1 is for formal charge
        self.assertEqual(num_node_features, expected)

    def test_one_of_k_encoding(self):
        """Test the one_of_k_encoding method."""
        featurizer = SmilesGraphFeaturizer()
        encoding = featurizer._one_of_k_encoding("C", ["C", "O", "N"])
        self.assertEqual(encoding, [1, 0, 0])

    def test_one_of_k_encoding_unk(self):
        """Test the one_of_k_encoding_unk method."""
        featurizer = SmilesGraphFeaturizer()
        # Input an invalid value and see if it is encoded as 'UNK'
        encoding = featurizer._one_of_k_encoding_unk("A", ["C", "O", "N", "UNK"])
        self.assertEqual(encoding, [0, 0, 0, 1])

    def test_get_dict_and_load_dict(self):
        """Test the get_dict and load_dict methods."""
        featurizer = SmilesGraphFeaturizer()
        config = featurizer.get_dict()
        new_featurizer = SmilesGraphFeaturizer()
        new_featurizer.load_dict(config)
        self.assertEqual(
            new_featurizer.include_edge_features, featurizer.include_edge_features
        )
        self.assertEqual(
            new_featurizer.atom_allowable_sets, featurizer.atom_allowable_sets
        )
        self.assertEqual(
            new_featurizer.bond_allowable_sets, featurizer.bond_allowable_sets
        )

    def test_default_features(self):
        """Test that default features are added correctly."""
        featurizer = SmilesGraphFeaturizer()
        featurizer.set_default_config()
        data = featurizer.featurize(self.smiles)
        self.assertEqual(data.x.shape[1], 30)


if __name__ == "__main__":
    unittest.main()
