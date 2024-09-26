import unittest
from rdkit import Chem
import numpy as np
import pandas as pd
from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint


class TestMACCSKeysFingerprint(unittest.TestCase):
    """Test MACCSKeysFingerprint."""

    def setUp(self):
        """Set up tests."""
        self.smiles1 = "O=C(NO)c1cc(CS(=O)(=O)c2ccc(Cl)cc2)on1"
        self.smiles2 = "CCCC(=O)OC1=CC=CC=C1C(=O)O"
        self.mol1 = Chem.MolFromSmiles(self.smiles1)
        self.mol2 = Chem.MolFromSmiles(self.smiles2)
        self.featurizer = MACCSKeysFingerprint()

    def test_maccs_keys_fingerprint_with_smiles(self):
        """Test featurize using SMILES."""
        descriptors = self.featurizer([self.smiles1, self.smiles2])
        assert descriptors.shape == (2, 167), "Wrong shape"
        assert isinstance(descriptors[0][0], np.int8), "The value is not of type int8"

    def test_maccs_keys_fingerprint_dataframe(self):
        """Test featurize_dataframe using mols."""
        descriptors_df = self.featurizer.featurize_dataframe(
            [self.smiles1, self.smiles2]
        )
        assert isinstance(descriptors_df, pd.DataFrame), "Output is not a DataFrame"
        assert descriptors_df.shape == (2, 167), "Wrong DataFrame shape"
        assert isinstance(
            descriptors_df.iloc[0, 0], np.int8
        ), "The value is not of type int8"


if __name__ == "__main__":
    unittest.main()
