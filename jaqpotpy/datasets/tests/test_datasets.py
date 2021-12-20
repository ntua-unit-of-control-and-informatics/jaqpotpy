"""
Tests for Jaqpotpy Datasets.
"""
import unittest
from jaqpotpy.datasets.dataset_base import SmilesTabularDataset
from jaqpotpy.descriptors.molecular import MordredDescriptors


class TestDatasets(unittest.TestCase):

    def test_smiles_tab_data_with_x(self):
        featurizer = MordredDescriptors()
        dataset = SmilesTabularDataset('./data/ecoli_DNA_gyrase_subunit_B_reductase_ic50.csv'
                                       , x_cols='molregno', y_cols='standard_value', smiles_col='canonical_smiles'
                                       , featurizer=featurizer)
        dataset.create()

        print(dataset.featurizer_name)

    def test_smiles_tab_data(self):
        featurizer = MordredDescriptors()
        dataset = SmilesTabularDataset('./data/ecoli_DNA_gyrase_subunit_B_reductase_ic50.csv'
                                       , x_cols=None, y_cols='standard_value', smiles_col='canonical_smiles'
                                       , featurizer=featurizer)
        dataset.create()

        print(dataset.featurizer_name)
