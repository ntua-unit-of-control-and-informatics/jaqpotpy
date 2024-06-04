import unittest
import rdkit
from rdkit import Chem
# pylint: disable=no-member

class TestMol(unittest.TestCase):

    mols = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
            , 'O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1']
    @unittest.skip("This test needs refactoring")
    def test_mol(self):
        mol = Chem.MolFromSmiles(self.mols[0])
        print(mol)
