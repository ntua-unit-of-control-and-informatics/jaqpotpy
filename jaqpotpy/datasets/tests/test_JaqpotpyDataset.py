"""Tests for jaqpotpy Datasets."""

import os
import unittest
import pandas as pd
import sklearn.feature_selection as skfs
from jaqpotpy.descriptors.molecular import (
    MACCSKeysFingerprint,
    MordredDescriptors,
    RDKitDescriptors,
    TopologicalFingerprint,
)
from jaqpotpy.datasets.jaqpot_tabular_dataset import JaqpotTabularDataset


class TestDatasets(unittest.TestCase):
    """TestDatasets is a unit testing class for validating jaqpotpy datasets.
    It uses the unittest framework to run tests on different types of datasets.

    Attributes
    ----------
        smiles (list): A list of SMILES strings representing molecular structures used for testing.
        activity (list): A list of binary binary_classification labels corresponding to the molecules in mols.
        binary_classification_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        regression_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.

    Methods
    -------

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
        self.featurizers = [
            MACCSKeysFingerprint(),
            MordredDescriptors(),
            RDKitDescriptors(),
            TopologicalFingerprint(),
        ]

    def test_dataset_with_path_single_smiles_and_external(self):
        # Assert that all JaqpotpyDataset features return the desired values
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        self.assertIsNotNone(dataset.df, "DataFrame should not be None")
        self.assertEqual(
            dataset.task,
            "BINARY_CLASSIFICATION",
            "Task should be 'binary_classification'",
        )
        self.assertEqual(
            dataset.smiles_cols,
            self.single_smiles_cols,
            "SMILES columns should match",
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
            dataset.featurizer_name[0],
            "MACCSKeysFingerprint",
            "Featurizer name should match",
        )
        self.assertIsInstance(
            dataset.featurizer[0],
            MACCSKeysFingerprint,
            "Featurizer should be an instance of MACCSKeysFingerprint",
        )

        self.assertGreater(
            len(dataset.x_colnames),
            len(self.x_cols),
            "x_colnames should include both original and featurized columns",
        )

        self.assertIn(
            dataset.task,
            ["REGRESSION", "BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"],
            "Task should be either 'regression', 'binary_classification' or 'multiclass_classification'",
        )
        self.assertEqual(len(dataset), 139, "The length of the dataset should be 139")

        repr_str = repr(dataset)
        self.assertIn(
            "smiles=True",
            repr_str,
            "The repr string should indicate SMILES columns are present",
        )
        self.assertIn(
            " featurizer=['MACCSKeysFingerprint']",
            repr_str,
            "The repr string should include the featurizer name",
        )

    def test_dataset_with_df_single_smiles_and_external(self):
        # Assert that JaqpotpyDataset built with dataframe has the correct dimensions
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )

        self.assertEqual(dataset.df.shape[1], 170, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_with_only_external(self):
        # Assert that JaqpotpyDataset built with only two external has the correct dimensions
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=None,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )

        self.assertEqual(dataset.df.shape[1], 3, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_no_x_no_smiles_none(self):
        # Assert that a TypeError is thrown if the user doesn't provide any smiles and external
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=None,  # ["SMILES"],
                x_cols=None,  # self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_dataset_no_x_no_smiles_empty_list(self):
        # Assert that a ValueError is thrown if the user doesn't provide any smiles and external
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=[],  # ["SMILES"],
                x_cols=[],  # self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_dataset_only_smiles(self):
        # Assert that JaqpotpyDataset built with only smiles has the correct dimensions
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=None,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        self.assertEqual(dataset.df.shape[1], 168, "DataFrame should have 170 columns")
        self.assertEqual(dataset.df.shape[0], 139, "DataFrame should have 139 rows")

    def test_dataset_smiles_no_featurizer(self):
        # Assert that a TypeError is thrown if the user gives smiles but no featurizer
        with self.assertRaises(TypeError):
            JaqpotTabularDataset(
                path=self.path,
                y_cols=self.y_cols,
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=None,
            )

    def test_dataset_smiles_no_task(self):
        # Assert that a TypeError is thrown if the user doesn't provide a task
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
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
            JaqpotTabularDataset(
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
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES", "Χ1"],  # Overlap with x_cols
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_overlap_smiles_y(self):
        # Test should fail, as there is overlap between smiles_cols and y_cols
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES", "ACTIVITY"],  # Overlap with y_cols
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_overlap_x_y(self):
        # Test should fail, as there is overlap between x_cols and y_cols
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=["ACTIVITY", "X2"],  # Overlap with x_cols
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_invalid_smiles_cols(self):
        # Test should fail as SMILES2 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=["SMILES2"],  # Non-existent column
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_invalid_x_cols(self):
        # Test should fail as feat3 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=self.y_cols,
                smiles_cols=self.single_smiles_cols,
                x_cols=["feat3"],  # Non-existent column
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_invalid_y_cols(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(ValueError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_no_path_no_df(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(TypeError):
            JaqpotTabularDataset(
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_path_and_df(self):
        # Test should fail as ACTIVITY2 does not exist in the DataFrame
        with self.assertRaises(TypeError):
            JaqpotTabularDataset(
                df=self.single_smiles_df,
                path=self.path,
                y_cols=["ACTIVITY2"],  # Non-existent column
                smiles_cols=self.single_smiles_cols,
                x_cols=self.x_cols,
                task="binary_classification",
                featurizer=self.featurizer,
            )

    def test_dataset_with_multiple_descriptors(self):
        # Assert that JaqpotpyDataset built with multiple descriptors has the correct dimensions
        dataset = JaqpotTabularDataset(
            df=self.single_smiles_df.iloc[0:3, :],
            y_cols=self.y_cols,
            smiles_cols=["SMILES"],
            x_cols=None,
            task="binary_classification",
            featurizer=self.featurizers,
        )

        self.assertEqual(dataset.X.shape[1], 4038, "DataFrame should have 4038 columns")
        self.assertEqual(dataset.X.shape[0], 3, "DataFrame should have 3 rows")

    def test_select_features_with_feature_selector(self):
        # Test with a FeatureSelector (VarianceThreshold)
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        feature_selector = skfs.VarianceThreshold(threshold=0.1)
        dataset.select_features(feature_selector)
        self.assertEqual(dataset.X.shape[1], 69, "DataFrame should have 4038 columns")
        self.assertIn("X1", dataset.selected_features, "X1 should be selected")
        self.assertNotIn(
            "f1",
            dataset.selected_features,
            "f1 should not be selected due to low variance",
        )

    def test_select_features_with_selection_list(self):
        # Test with a predefined selection list
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        selection_list = ["X1", "X2"]
        dataset.select_features(SelectColumns=selection_list)
        self.assertEqual(dataset.X.shape[1], 2, "DataFrame should have 2 columns")
        self.assertEqual(
            list(dataset.X.columns),
            selection_list,
            "Only X1 and X2 should be selected",
        )
        selection_list = ["X1", "X3"]

        with self.assertRaises(ValueError):
            dataset.select_features(SelectColumns=selection_list)

    def test_select_features_with_both_arguments(self):
        # Test that passing both FeatureSelector and SelectColumns raises an error
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        with self.assertRaises(ValueError):
            dataset.select_features(
                FeatureSelector=skfs.VarianceThreshold(threshold=0.01),
                SelectColumns=["X1", "X2"],
            )

    def test_select_features_with_neither_argument(self):
        # Test that passing neither FeatureSelector nor SelectColumns raises an error
        dataset = JaqpotTabularDataset(
            path=self.path,
            y_cols=self.y_cols,
            smiles_cols=self.single_smiles_cols,
            x_cols=self.x_cols,
            task="binary_classification",
            featurizer=self.featurizer,
        )
        with self.assertRaises(ValueError):
            dataset.select_features()


if __name__ == "__main__":
    unittest.main()
