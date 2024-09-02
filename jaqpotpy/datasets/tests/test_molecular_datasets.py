"""
Tests for jaqpotpy Datasets.
"""

import os
import unittest
import pandas as pd
from jaqpotpy.descriptors.molecular import (
    MordredDescriptors,
    MACCSKeysFingerprint,
    TopologicalFingerprint,
    MolGraphConvFeaturizer,
    RDKitDescriptors,
    SmilesToSeq,
    create_char_to_idx,
    SmilesToImage,
    MolGanFeaturizer,
)
from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset


class TestDatasets(unittest.TestCase):
    """
    TestDatasets is a unit testing class for validating jaqpotpy datasets.
    It uses the unittest framework to run tests on different types of datasets.

    Attributes:
        smiles (list): A list of SMILES strings representing molecular structures used for testing.
        activity (list): A list of binary classification labels corresponding to the molecules in mols.
        classification_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        regression_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.

    Methods:

    """

    def setUp(self) -> None:
        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, "../../test_data"))
        single_smiles_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_classification.csv"
        )
        multiple_smiles_csv_file_path = os.path.join(
            test_data_dir, "test_data_many_smiles_columns.csv"
        )
        self.path = single_smiles_csv_file_path
        self.single_smiles_df = pd.read_csv(single_smiles_csv_file_path)
        self.multi_smiles_df = pd.read_csv(multiple_smiles_csv_file_path)
        self.multi_smiles_cols = ["SMILES", "SMILES2"]
        self.single_smiles_cols = ["SMILES"]
        self.y_cols = ["ACTIVITY"]
        self.x_cols = ["X1", "X2"]
        self.featurizer = MACCSKeysFingerprint()

    def test_dataset_with_path_single_smiles_and_external(self):
        # Assert that all JaqpotpyDataset features return the desired values
        dataset = JaqpotpyDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="classification",
            featurizer=self.featurizer,
        )

        self.assertIsNotNone(dataset.df, "DataFrame should not be None")
        self.assertEqual(
            dataset.task, "classification", "Task should be 'classification'"
        )
        self.assertEqual(
            dataset.smiles_cols, self.single_smiles_cols, "SMILES columns should match"
        )
        self.assertEqual(dataset.y_cols, self.y_cols, "Y columns should match")
        self.assertEqual(dataset.x_cols, self.x_cols, "X columns should match")

        self.assertEqual(dataset.df.shape[1], 170, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

        self.assertTrue(
            set(self.x_cols).issubset(dataset.df.columns),
            "SMILES columns should be present in the DataFrame",
        )
        self.assertTrue(
            set(self.y_cols).issubset(dataset.df.columns),
            "Activity columns should be present in the DataFrame",
        )

        # Check that featurization results are in the DataFrame
        featurized_cols = [col for col in dataset.df.columns if col.startswith("f")]
        self.assertGreater(
            len(featurized_cols),
            1,
            "There should be featurized columns in the DataFrame",
        )

        self.assertEqual(
            dataset.featurizer_name,
            "MACCSKeysFingerprint",
            "Featurizer name should match",
        )
        self.assertIsInstance(
            dataset.featurizer,
            MACCSKeysFingerprint,
            "Featurizer should be an instance of MACCSKeysFingerprint",
        )

        self.assertGreater(
            len(dataset.x_cols_all),
            len(self.x_cols),
            "x_cols_all should include both original and featurized columns",
        )

        self.assertIn(
            dataset.task,
            ["regression", "classification"],
            "Task should be either 'regression' or 'classification'",
        )
        self.assertEqual(len(dataset), 139, "The length of the dataset should be 139")

        repr_str = repr(dataset)
        self.assertIn(
            "smiles=True",
            repr_str,
            "The repr string should indicate SMILES columns are present",
        )
        self.assertIn(
            "featurizer=MACCSKeysFingerprint",
            repr_str,
            "The repr string should include the featurizer name",
        )

    def test_dataset_with_df_single_smiles_and_external(self):
        # Assert that JaqpotpyDataset built with dataframe has the correct dimensions
        dataset = JaqpotpyDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="classification",
            featurizer=self.featurizer,
        )

        self.assertEqual(dataset.df.shape[1], 170, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_with_only_external(self):
        # Assert that JaqpotpyDataset built with only two external has the correct dimensions
        dataset = JaqpotpyDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=None,
            x_cols=self.x_cols,
            task="classification",
            featurizer=self.featurizer,
        )

        self.assertEqual(dataset.df.shape[1], 3, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_no_x_no_smiles_none(self):
        # Assert that a TypeError is thrown if the user doesn't provide any smiles and external
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=None,  # ["SMILES"],
                x_cols=None,  # self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_dataset_no_x_no_smiles_empty_list(self):
        # Assert that a ValueError is thrown if the user doesn't provide any smiles and external
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=[],  # ["SMILES"],
                x_cols=[],  # self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_dataset_only_smiles(self):
        # Assert that JaqpotpyDataset built with only smiles has the correct dimensions
        dataset = JaqpotpyDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=None,
            task="classification",
            featurizer=self.featurizer,
        )
        self.assertEqual(dataset.df.shape[1], 168, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_smiles_no_featurizer(self):
        # Assert that a TypeError is thrown if the user gives smiles but no featurizer
        with self.assertRaises(TypeError):
            JaqpotpyDataset(
                path=self.path,
                y_cols=self.y_cols,
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="classification",
                featurizer=None,
            )

    def test_dataset_smiles_no_task(self):
        # Assert that a TypeError is thrown if the user doesn't provide a task
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                path=self.path,
                y_cols=self.y_cols,
                smiles_cols=None,
                x_cols=self.x_cols,
                task=None,
                featurizer=self.featurizer,
            )

    def test_dataset_smiles_wrong_task(self):
        # Assert that a TypeError is thrown if the user provides a wrong task label
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                path=self.path,
                y_cols=self.y_cols,
                smiles_cols=None,
                x_cols=self.x_cols,
                task="laricifato",
                featurizer=self.featurizer,
            )

    def test_overlap_smiles_x(self):
        # Test should fail, as there is overlap between smiles_cols and x_cols
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES", "Χ1"],  # Overlap with x_cols
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_overlap_smiles_y(self):
        # Test should fail, as there is overlap between smiles_cols and y_cols
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES", "ACTIVITY"],  # Overlap with y_cols
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_overlap_x_y(self):
        # Test should fail, as there is overlap between x_cols and y_cols
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=["ACTIVITY", "X2"],  # Overlap with x_cols
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_invalid_smiles_cols(self):
        # Test should fail as SMILES2 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES2"],  # Non-existent column
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_invalid_x_cols(self):
        # Test should fail as feat3 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=self.single_smiles_cols,
                x_cols=["feat3"],  # Non-existent column
                task="classification",
                featurizer=self.featurizer,
            )

    def test_invalid_y_cols(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_no_path_no_df(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(TypeError):
            JaqpotpyDataset(
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    def test_path_and_df(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(TypeError):
            JaqpotpyDataset(
                df=self.single_smiles_df,
                path=self.path,
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="classification",
                featurizer=self.featurizer,
            )

    # @unittest.skip("This test needs refactoring")
    # def test_streaming_dataset(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys
    #                             , featurizer=MolGraphConvFeaturizer(use_edges=True), streaming=True)
    #     item = dataset.__getitem__(10)

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_tab_data_with_x(self):
    #     # featurizer = TopologicalFingerprint()
    #     featurizer = MordredDescriptors(ignore_3D=False)
    #     path_b = '../../test_data/data_big.csv'
    #     path = '../../test_data/small.csv'
    #     dataset = MolecularTabularDataset(path=path
    #                                       , x_cols=['molregno', 'organism']
    #                                       , y_cols=['standard_value']
    #                                       , smiles_col='canonical_smiles'
    #                                       , featurizer=featurizer
    #                                       ,
    #                                       X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
    #                                          'VE1_A', 'VE2_A']
    #                                       )

    #     dataset.create()
    #     assert dataset.featurizer_name == 'MordredDescriptors'
    #     assert dataset.x_cols == ['molregno', 'organism']
    #     assert dataset.y_cols == ['standard_value']
    #     assert dataset.smiles_strings[0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
    #     assert dataset.df.shape == (4, 1829)

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_tab_data_without_x(self):
    #     featurizer = MordredDescriptors(ignore_3D=False)
    #     path = '../../test_data/small.csv'
    #     dataset = MolecularTabularDataset(path=path
    #                                       , y_cols=['standard_value']
    #                                       , smiles_col='canonical_smiles'
    #                                       , featurizer=featurizer
    #                                       )

    #     dataset.create()
    #     assert dataset.featurizer_name == 'MordredDescriptors'
    #     assert dataset.y_cols == ['standard_value']
    #     assert dataset.smiles_strings[
    #                0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
    #     assert dataset.df.shape == (4, 1827)

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_dataset_mordred(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MordredDescriptors(ignore_3D=True))
    #     dataset.create()
    #     assert dataset.featurizer_name == 'MordredDescriptors'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 1614)

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_dataset_rdkit(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=RDKitDescriptors())
    #     dataset.create()
    #     dataset.__repr__()
    #     assert dataset.featurizer_name == 'RDKitDescriptors'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 209)

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_dataset_finger(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=TopologicalFingerprint())
    #     dataset.create()

    #     assert dataset.featurizer_name == 'TopologicalFingerprint'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 2049)

    # @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    # def test_smiles_dataset_molgraph(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MolGraphConvFeaturizer(use_edges=True))
    #     dataset.create()
    #     assert dataset.featurizer_name == 'MolGraphConvFeaturizer'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 2)

    # @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    # def test_smiles_torch_dataset(self):
    #     dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification')
    #     dataset.create()

    #     dataloader = DataLoader(dataset, batch_size=4,
    #                             shuffle=True, num_workers=0)
    #     for data in dataloader:
    #         print(data)

    #     assert dataset.featurizer_name == 'MolGraphConvFeaturizer'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert len(dataset.df) == 23

    # @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    # def test_smiles_torch_tab_dataset(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MordredDescriptors(ignore_3D=True),  task='classification')
    #     dataset.create()
    #     dataloader = dl(dataset, batch_size=4,
    #                             shuffle=True, num_workers=0)
    #     for data in dataloader:
    #         from torch import Tensor
    #         assert type(data[0]) == Tensor
    #         # print(data)
    #     assert dataset.featurizer_name == 'MordredDescriptors'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert len(dataset.df) == 23

    # @unittest.skip("This test needs refactoring")
    # def test_smiles_tab_data_save(self):
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=TopologicalFingerprint())
    #     dataset.create()

    #     assert dataset.featurizer_name == 'TopologicalFingerprint'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 2049)

    # @unittest.skip("This test needs refactoring")
    # def test_load_dataset(self):
    #     dataset = SmilesDataset()
    #     dataset = dataset.load("./Smiles_fingerprints.jdb")
    #     assert dataset.featurizer_name == 'TopologicalFingerprint'
    #     assert dataset.smiles_strings[
    #                0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
    #     assert dataset.df.shape == (23, 2049)

    # @unittest.skip("This test needs refactoring")
    # def test_seq_dataset(self):
    #     cid = create_char_to_idx(self.mols)
    #     feat = SmilesToSeq(char_to_idx=cid)
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys_regr, featurizer=feat)
    #     dataset.create()
    #     dataloader = dl(dataset, batch_size=4,
    #                             shuffle=True, num_workers=0)
    #     for data in dataloader:
    #         from torch import Tensor
    #         assert type(data[0]) == Tensor

    # @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    # def test_image_dataset(self):
    #     feat = SmilesToImage(img_size=80)
    #     dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=feat)
    #     dataset.create()
    #     dataloader = dl(dataset, batch_size=4,
    #                             shuffle=True, num_workers=0)
    #     for data in dataloader:
    #         from torch import Tensor
    #         assert type(data[0]) == Tensor

    # @unittest.skip("This test needs refactoring")
    # def test_generative_datasets(self):
    #     feat = MolGanFeaturizer(max_atom_count=60)
    #     dataset = SmilesDataset(smiles=self.mols, task="generation", featurizer=feat)
    #     dataset.create()
    #     df = dataset.__getitem__(10)


if __name__ == "__main__":
    unittest.main()
