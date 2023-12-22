"""
Test basic molecular features.
"""
import numpy as np
import unittest

from jaqpotpy.descriptors import OneHotSequence


class TestOneHotSeqDescriptors(unittest.TestCase):
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
    featurizer = OneHotSequence()
    descriptors = featurizer.featurize(smiles)
    assert len(descriptors[0]) == 100

  def test_sts_descriptors_on_smiles(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    featurizer = OneHotSequence()
    descriptors = featurizer.featurize('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert len(descriptors[0]) == 100

  def test_sts_descriptors_on_smiles_pad(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    featurizer = OneHotSequence(max_length=50)
    descriptors = featurizer.featurize('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert len(descriptors[0]) == 50

  def test_sts_descriptors_on_smiles_df(self):
    """
    Test invocation on raw smiles.
    """
    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    featurizer = OneHotSequence()
    # descriptors = featurizer.featurize_dataframe(
    #     ['CC(=O)OC1=CC=CC=C1C(=O)O'])
    descriptors = featurizer.featurize_dataframe(['CC(=O)OC1=CC=CC=C1C(=O)O'
                                                     , 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
    assert descriptors.shape == (3, 1)
    featurizer = OneHotSequence()
    descriptors = featurizer.featurize_dataframe(['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
    print(descriptors['OneHotSequence'].loc[[0]].to_numpy().shape)
    print(descriptors.loc[[0]]['OneHotSequence'])
    print(descriptors.loc[[0]]['OneHotSequence'].to_numpy().shape)
    print(descriptors)
    assert descriptors.shape == (2, 1)
