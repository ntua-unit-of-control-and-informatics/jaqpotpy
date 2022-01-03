"""
Test topological fingerprints.
"""
import unittest
from jaqpotpy.descriptors import TopologicalFingerprint


class TestCircularFingerprint(unittest.TestCase):
  """
    Tests for CircularFingerprint.
    """

  def setUp(self):
    """
        Set up tests.
        """
    from rdkit import Chem
    smiles = 'O=C(NO)c1cc(CS(=O)(=O)c2ccc(Cl)cc2)on1'
    self.mol = Chem.MolFromSmiles(smiles)
    smiles = 'C/C=C/C=C/C(=O)N[C@@H](CC(=O)N[C@H](C(=O)[C@@H]1C(=O)NC(=O)[C@H]1C)C(C)C)c1ccccc1'
    self.mol2 = Chem.MolFromSmiles(smiles)

  def test_circular_fingerprints(self):
    """
        Test CircularFingerprint.
        """
    featurizer = TopologicalFingerprint()
    rval = featurizer([self.mol, self.mol2])
    assert rval.shape == (2, 2048)

  def test_circular_fingerprints_with_1024(self):
    """
        Test CircularFingerprint with 1024 size.
        """
    featurizer = TopologicalFingerprint(size=1024)
    rval = featurizer([self.mol])
    assert rval.shape == (1, 1024)

  def test_sparse_circular_fingerprints(self):
    """
        Test CircularFingerprint with sparse encoding.
        """
    featurizer = TopologicalFingerprint(sparse=True)
    rval = featurizer([self.mol, self.mol2])
    assert rval.shape == (2,)
    assert isinstance(rval[0], dict)
    assert len(rval[0])

  def test_sparse_circular_fingerprints_with_smiles(self):
    """
        Test CircularFingerprint with sparse encoding and SMILES for each
        fragment.
    """
    featurizer = TopologicalFingerprint(sparse=True, smiles=True)
    rval = featurizer([self.mol])
    assert rval.shape == (1,)
    assert isinstance(rval[0], dict)
    assert len(rval[0])

    # check for separate count and SMILES entries for each fragment
    for fragment_id, value in rval[0].items():
      assert 'count' in value
      assert 'smiles' in value
