import numpy as np
import unittest

from jaqpotpy.descriptors import MordredDescriptors
from jaqpotpy.cfg import config

class TestMordredDescriptors(unittest.TestCase):

    def setUp(self):
        """
        Set up tests. Initialize 2 mols and the default featurizer
        """
        from rdkit import Chem
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol = Chem.MolFromSmiles(smiles)
        smiles = 'CCCC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol2 = Chem.MolFromSmiles(smiles)
        self.featurizer = MordredDescriptors()

    def test_mordred_descriptors(self):
        """
        Test simple descriptors. Shape and first 3 feature values (with missing = -1000.00)
        """
        descriptors = self.featurizer([self.mol])
        assert descriptors.shape == (1, 1613)
        # Testing the 5th descriptors from the Mordred library (SpAbs_A = 15.284)
        assert np.allclose(
        descriptors[0, 4],
        15.28,
        atol=0.1)

    def test_mordred_descriptors_dataframe(self):
        """
        Test with dataframe
        """
        descriptors = self.featurizer.featurize_dataframe([self.mol])
        assert descriptors.shape == (1, 1613)
        # Testing the 3rd and 4th descriptors (nAcid = 1, nBase = 0)
        assert descriptors.iat[0, 2] == 1.0
        assert descriptors.iat[0, 3] == 0.0

    def test_mordred_descriptors_dataframe_two_row(self):
        """
        Test with dataframe
        """
        descriptors = self.featurizer.featurize_dataframe([self.mol, self.mol2])
        assert descriptors.shape == (2, 1613)
        # Testing the 3rd and 4th descriptors (nAcid = 1, nBase = 0)
        assert descriptors.iat[0, 2] == 1.0
        assert descriptors.iat[0, 3] == 0.0

    def test_mordred_descriptors_with_smiles(self):
        """
        Test simple descriptors with smiles input.
        """
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        descriptors = self.featurizer([smiles])
        assert descriptors.shape == (1, 1613)
        assert np.allclose(
        descriptors[0, 4],
        15.28,
        atol=0.1)

    def test_mordred_descriptors_with_3D_info(self):
        """
        Test simple descriptors with 3D info
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        featurizer = MordredDescriptors(ignore_3D=False)
        descriptors = featurizer([self.mol])
        assert descriptors.shape == (1, 1826)
        assert np.allclose(descriptors[0][780:784], np.array([-1000., -1000., -1000., -1000.]))

        # calculate coordinates
        mol = self.mol
        mol_with_conf = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_conf, AllChem.ETKDG())
        descriptors = featurizer([mol_with_conf])
        assert descriptors.shape == (1, 1826)
        # not zero values
        assert not np.allclose(descriptors[0][780:784], np.array([-1000., -1000., -1000., -1000.]))

'''
    def test_base_mordred(self):
        from rdkit import Chem

        from mordred import Chi, ABCIndex

        benzene = Chem.MolFromSmiles('c1ccccc1')

        # create descriptor instance
        abci = ABCIndex.ABCIndex()

        # calculate descriptor value
        result = abci(benzene)

        assert result == 4.242640687119286

        # create descriptor instance with parameter
        chi_pc4 = Chi.Chi(type='path_cluster', order=4)

        # calculate
        result = chi_pc4(benzene)

        assert result == 0.0
'''