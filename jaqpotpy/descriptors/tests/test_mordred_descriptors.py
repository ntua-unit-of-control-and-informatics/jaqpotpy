import numpy as np
import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Chi, ABCIndex

from jaqpotpy.descriptors.molecular import MordredDescriptors


class TestMordredDescriptors(unittest.TestCase):
    """Test MordredDescriptors."""

    def setUp(self):
        """Set up tests."""
        self.smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.mol = Chem.MolFromSmiles(self.smiles1)
        self.smiles2 = "CCCC(=O)OC1=CC=CC=C1C(=O)O"
        self.mol2 = Chem.MolFromSmiles(self.smiles2)
        self.featurizer = MordredDescriptors()

    def test_mordred_descriptors(self):
        """Test featurize using mols"""
        descriptors = self.featurizer([self.mol, self.mol2])
        assert descriptors.shape == (2, 1613), "Wrong shape"
        assert isinstance(
            descriptors[0][1], (int, float, np.number)
        ), "The value is not numeric"

    def test_mordred_descriptors_dataframe(self):
        """Test featurize_dataframe using mols"""
        descriptors = self.featurizer.featurize_dataframe([self.mol, self.mol2])
        assert descriptors.shape == (2, 1613)
        assert isinstance(descriptors.iloc[0, 1], (int, float, np.number))

    def test_mordred_descriptors_with_smiles(self):
        """Test featurize using Smiles"""
        descriptors = self.featurizer([self.smiles1, self.smiles2])
        assert descriptors.shape == (2, 1613)
        assert isinstance(descriptors[0][1], (int, float, np.number))

    def test_mordred_descriptors_with_3D_info(self):
        """Test simple descriptors with 3D info"""
        featurizer = MordredDescriptors(ignore_3D=False)
        descriptors = featurizer([self.mol, self.mol2])
        assert descriptors.shape == (2, 1826)
        assert isinstance(
            descriptors[0][1], (int, float, np.number)
        ), "The value is not numeric"

        # calculate coordinates
        mol = self.mol
        mol_with_conf = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_conf, AllChem.ETKDG())
        descriptors = featurizer([mol_with_conf])
        assert descriptors.shape == (1, 1826)
        # not zero values
        assert not np.allclose(descriptors[0][780:784], np.array([0.0, 0.0, 0.0, 0.0]))

    def test_base_mordred(self):
        """Mordredcommunity is used instead of morded cause of numpy errors"""
        benzene = Chem.MolFromSmiles("c1ccccc1")
        # create descriptor instance
        abci = ABCIndex.ABCIndex()

        # calculate descriptor value
        result = abci(benzene)

        assert result == 4.242640687119286
        # create descriptor instance with parameter
        chi_pc4 = Chi.Chi(type="path_cluster", order=4)

        # calculate
        result = chi_pc4(benzene)

        assert result == 0.0
