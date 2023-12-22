import unittest
import rdkit
from rdkit import Chem


class TestMol(unittest.TestCase):

    mols = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
            , 'O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1']

    def test_mol(self):
        mol = Chem.MolFromSmiles(self.mols[0])
        print(mol)
