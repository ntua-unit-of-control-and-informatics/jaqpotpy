import numpy as np
import unittest

from jaqpotpy.descriptors import MordredDescriptors


class TestMordredDescriptors(unittest.TestCase):
    """
    Test MordredDescriptors.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit import Chem
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol = Chem.MolFromSmiles(smiles)
        smiles = 'CCCC(=O)OC1=CC=CC=C1C(=O)O'
        self.mol2 = Chem.MolFromSmiles(smiles)

    def test_mordred_descriptors(self):
        """
        Test simple descriptors.
        """
        featurizer = MordredDescriptors()
        descriptors = featurizer([self.mol])
        assert descriptors.shape == (1, 1613)
        assert np.allclose(descriptors[0][0:3],
                           np.array([9.54906713, 9.03919229, 1.0]))

    def test_mordred_descriptors_dataframe(self):
        """
        Test simple descriptors.
        """
        featurizer = MordredDescriptors()
        descriptors = featurizer.featurize_dataframe([self.mol])
        assert descriptors.shape == (1, 1613)
        assert descriptors.iat[0, 0] == 9.54906712535007
        assert descriptors.iat[0, 1] == 9.039192285773227
        assert descriptors.iat[0, 2] == 1.0

    def test_mordred_descriptors_dataframe_two_row(self):
        """
        Test simple descriptors.
        """
        featurizer = MordredDescriptors()
        descriptors = featurizer.featurize_dataframe([self.mol, self.mol2])
        assert descriptors.shape == (2, 1613)
        assert descriptors.iat[0, 0] == 9.54906712535007
        assert descriptors.iat[0, 1] == 9.039192285773227
        assert descriptors.iat[0, 2] == 1.0


    def test_mordred_descriptors_with_smiles(self):
        """
        Test simple descriptors.
        """
        featurizer = MordredDescriptors()
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        descriptors = featurizer([smiles])
        assert descriptors.shape == (1, 1613)
        assert np.allclose(descriptors[0][0:3],
                           np.array([9.54906713, 9.03919229, 1.0]))


    def test_mordred_descriptors_with_3D_info(self):
        """
        Test simple descriptors with 3D info
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        featurizer = MordredDescriptors(ignore_3D=False)
        descriptors = featurizer([self.mol])
        assert descriptors.shape == (1, 1826)
        assert np.allclose(descriptors[0][780:784], np.array([0.0, 0.0, 0.0, 0.0]))

        # calculate coordinates
        mol = self.mol
        mol_with_conf = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_conf, AllChem.ETKDG())
        descriptors = featurizer([mol_with_conf])
        assert descriptors.shape == (1, 1826)
        # not zero values
        assert not np.allclose(descriptors[0][780:784],
                               np.array([0.0, 0.0, 0.0, 0.0]))
