"""
Tests for Jaqpotpy Datasets.
"""
import unittest
from jaqpotpy.datasets.dataset_base import MolecularTabularDataset
from jaqpotpy.descriptors import MordredDescriptors, TopologicalFingerprint
from mordred import descriptors


class TestDatasets(unittest.TestCase):

    def test_smiles_tab_data_with_x(self):
        # featurizer = TopologicalFingerprint()
        featurizer = MordredDescriptors(ignore_3D=False)
        path_b = '../../test_data/data_big.csv'
        path = '../../test_data/small.csv'
        dataset = MolecularTabularDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )


        dataset.create()
        print(dataset.df)
        assert dataset.featurizer_name == 'MordredDescriptors'
        assert dataset.x_cols == ['molregno', 'organism']
        assert dataset.y_cols == ['standard_value']
        assert dataset.smiles_strings[0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
        assert dataset.df.shape == (4, 1830)



        # print(dataset.df)
        # dataset.X = ['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VE2_A']
        # print(dataset.X)
        # print(dataset.y)
        #
        # print(dataset.__get_X__())
        # print(dataset.__get_Y__())

    # def test_smiles_tab_data(self):
    #     featurizer = MordredDescriptors()
    #     dataset = MolecularTabularDataset('./data/small.csv'
    #                                    , y_cols='standard_value', smiles_col='canonical_smiles'
    #                                    , featurizer=featurizer)
    #     dataset.create()
    #
    #     print(dataset.featurizer_name)
    #     print(dataset.X)
    #     print(dataset.Y)
    #     print(dataset.df)
