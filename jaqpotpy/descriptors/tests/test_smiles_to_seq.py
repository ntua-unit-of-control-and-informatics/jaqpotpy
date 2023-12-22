"""
Test basic molecular features.
"""
import numpy as np
import unittest

from jaqpotpy.descriptors import SmilesToSeq, create_char_to_idx


class TestSmilesToSeqDescriptors(unittest.TestCase):
  """
  Test Smiles to seq descriptors.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
    # self.featurizer = SmilesToSeq()

  def test_sts_descriptors(self):
    """
    Test simple descriptors.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    chars = create_char_to_idx(smiles=smiles)
    featurizer = SmilesToSeq(char_to_idx=chars)
    descriptors = featurizer.featurize(smiles)
    assert len(descriptors[0]) == 270

  def test_sts_descriptors_on_smiles(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    chars = create_char_to_idx(smiles=smiles)
    featurizer = SmilesToSeq(char_to_idx=chars, max_len=120, pad_len=10)
    descriptors = featurizer.featurize('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert len(descriptors[0]) == 140

  def test_sts_descriptors_on_smiles_pad(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    chars = create_char_to_idx(smiles=smiles)
    featurizer = SmilesToSeq(char_to_idx=chars, max_len=120, pad_len=0)
    descriptors = featurizer.featurize('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert len(descriptors[0]) == 120

  def test_sts_descriptors_on_smiles_df(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    chars = create_char_to_idx(smiles=smiles)
    featurizer = SmilesToSeq(char_to_idx=chars, max_len=120, pad_len=0)
    # descriptors = featurizer.featurize_dataframe(
    #     ['CC(=O)OC1=CC=CC=C1C(=O)O'])
    descriptors = featurizer.featurize_dataframe(['CC(=O)OC1=CC=CC=C1C(=O)O'
                                                     , 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
    assert descriptors.shape == (3, 1)
    featurizer = SmilesToSeq(char_to_idx=chars)
    descriptors = featurizer.featurize_dataframe(['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
    assert descriptors.shape == (2, 1)
