import unittest
import numpy as np
from jaqpotpy.descriptors.molecular import MolGanFeaturizer
from rdkit import Chem


class TestMolGan(unittest.TestCase):

    def test_mol_gan_feat(self):
        smiles = [
            'Cc1ccccc1CO', 'CC1CCC(C)C(N)C1C1CCC(C)C(N)C1', 'CCC(N)=O', 'Fc1cccc(F)c1', 'CC(C)F',
            'C1COC2NCCC2C1', 'C1=NCc2ccccc21'
        ]

        invalid_smiles = ['axa', 'xyz', 'inv']

        featurizer = MolGanFeaturizer(max_atom_count=20)
        valid_data = featurizer.featurize(smiles)
        print(valid_data[0].__dict__)

    def test_mol_gan_defeat(self):
        smiles = [
            'Cc1ccccc1CO',  'CCC(N)=O', 'Fc1cccc(F)c1', 'CC(C)F',
            'C1COC2NCCC2C1', 'C1=NCc2ccccc21'
        ]

        invalid_smiles = ['axa', 'xyz', 'inv']

        featurizer = MolGanFeaturizer(max_atom_count=20)
        valid_data = featurizer.featurize(smiles)
        sm = featurizer.defeaturize(valid_data)
        sm_def = [Chem.MolToSmiles(m) for m in sm]
        assert len(smiles) == len(sm_def)
        assert all([a == b for a, b in zip(smiles, sm_def)])

    def test_mol_gan_df(self):
        smiles = [
            'Cc1ccccc1CO', 'CC1CCC(C)C(N)C1C1CCC(C)C(N)C1', 'CCC(N)=O', 'Fc1cccc(F)c1', 'CC(C)F',
            'C1COC2NCCC2C1', 'C1=NCc2ccccc21'
        ]

        invalid_smiles = ['axa', 'xyz', 'inv']

        featurizer = MolGanFeaturizer(max_atom_count=20)
        valid_data = featurizer.featurize_dataframe(smiles)
