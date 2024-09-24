"""Tests for sklearn models through the jaqpotpy module."""

import unittest
import pandas as pd
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.models.preprocessing import Preprocess


# Add the following tests:
# 1.regression model
#  1.a. x and y preprocessed
#  1.b. only x preprocessed
#  1.c. only y preprocessed
#  1.d. no preprocessesing
# 1.Classification model
#  1.a. x and y preprocessed
#  1.b. only x preprocessed
#  1.c. only y preprocessed
#  1.d. no preprocessesing


class TestModels(unittest.TestCase):
    """TestModels is a unit testing class for validating various machine learning models applied to molecular datasets.
    It uses the unittest framework to run tests on classification and regression tasks, evaluating model performance
    and ensuring correct implementation.

    Attributes
    ----------
        smiles (list): A list of SMILES strings representing molecular structures used for testing.
        activity (list): A list of binary classification labels corresponding to the molecules in mols.
        classification_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        regression_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        X_train, X_test, y_train, y_test  (list): A list of continuous regression targets corresponding to the molecules in mols.
        y_test (list): A list of continuous regression targets corresponding to the molecules in mols.

    Methods
    -------
        test_rf_classification(): Tests a RandomForestClassifier on a molecular dataset with
                                          Mordred fingerprints for classification.
        test_rf_classification(): Tests a RandomForestRegressor on a molecular dataset with
                                          Mordred fingerprints for regression.
        test_predict_proba(): Tests the probability prediction of an SVC model using RDKit descriptors for classification.
        test_eval(): Tests the evaluation of an SVC model using various scoring functions for regression.
        test_ALL_regression_ONNX(): Tests all available regression models in scikit-learn for ONNX compatibility.
        test_ONE_regression_ONNX(): Tests a specific regression model (ARDRegression) for ONNX compatibility.
        test_ALL_classification_ONNX(): Tests all available classification models in scikit-learn for ONNX compatibility.
        test_ONE_classification_ONNX(): Tests a specific classification model (QuadraticDiscriminantAnalysis) for ONNX compatibility.

    """

    def setUp(self) -> None:
        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, "../../test_data"))
        clasification_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_classification.csv"
        )
        multi_classification_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_multi_classification.csv"
        )
        multi_classification_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_multi_classification.csv"
        )
        regression_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_regression.csv"
        )
        prediction_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_prediction_dataset.csv"
        )
        multiclass_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_prediction_dataset_multiclass.csv"
        )
        prediction_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_prediction_dataset.csv"
        )
        multiclass_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_prediction_dataset_multiclass.csv"
        )
        multioutput_regression_csv_file_path = os.path.join(
            test_data_dir, "test_data_smiles_regression_multioutput.csv"
        )

        self.classification_df = pd.read_csv(clasification_csv_file_path)
        self.multi_classification_df = pd.read_csv(multi_classification_csv_file_path)
        self.regression_df = pd.read_csv(regression_csv_file_path)
        self.regression_multioutput_df = pd.read_csv(
            multioutput_regression_csv_file_path
        )
        self.prediction_df = pd.read_csv(prediction_csv_file_path)
        self.prediction_multiclass_df = pd.read_csv(multiclass_csv_file_path)

    def test_SklearnModel_classification_no_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint fingerprints for classification.
        No preprocessing is applied on the data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
        )
        jaqpot_model.fit()
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        skl_probabilities = jaqpot_model.predict_proba(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        onnx_probabilities = jaqpot_model.predict_proba_onnx(validation_dataset)

        assert np.array_equal(
            skl_predictions, [1, 0, 0, 1, 1]
        ), f"Expected skl_predictions == [1, 0, 0, 1, 1], got {skl_predictions}"
        assert np.allclose(
            skl_probabilities,
            [0.7937499999999997, 0.69, 0.7427857142857142, 0.6573333333333332, 0.88],
            atol=1e-2,
        ), (
            f"Expected skl_probabilities == [0.7937499999999997, 0.69, 0.7427857142857142, 0.6573333333333332, 0.88]"
            f"got {skl_probabilities}"
        )

        assert np.array_equal(
            onnx_predictions, [1, 0, 0, 1, 1]
        ), f"Expected onnx_predictions == [1, 0, 0, 1, 1], got {onnx_predictions}"
        assert np.allclose(
            onnx_probabilities,
            [
                0.7937495708465576,
                0.690000057220459,
                0.7427858114242554,
                0.6573330163955688,
                0.8799994587898254,
            ],
            atol=1e-2,
        ), f"Expected onnx_probabilities == [0.7937495708465576, 0.690000057220459, 0.7427858114242554, 0.6573330163955688, 0.8799994587898254], got {onnx_probabilities}"

    def test_SklearnModel_classification_x_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for classification.
        Preprocessing is applied only on the input features.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        skl_probabilities = jaqpot_model.predict_proba(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        onnx_probabilities = jaqpot_model.predict_proba_onnx(validation_dataset)

        assert np.array_equal(
            skl_predictions, [1, 0, 0, 1, 1]
        ), f"Expected skl_predictions == [1, 0, 0, 1, 1], got {skl_predictions}"
        assert np.allclose(
            skl_probabilities,
            [0.7837499999999997, 0.69, 0.7327857142857142, 0.6873333333333332, 0.88],
            atol=1e-2,
        ), f"Expected skl_probabilities == [0.7837499999999997, 0.69, 0.7327857142857142, 0.6873333333333332, 0.88], got {skl_probabilities}"

        assert np.array_equal(
            onnx_predictions, [1, 0, 0, 1, 1]
        ), f"Expected onnx_predictions == [1, 0, 0, 1, 1], got {onnx_predictions}"
        assert np.allclose(
            onnx_probabilities,
            [
                0.7837496399879456,
                0.6600000858306885,
                0.7427858114242554,
                0.6873330473899841,
                0.8799994587898254,
            ],
            atol=1e-2,
        ), f"Expected onnx_probabilities == [0.7837496399879456, 0.6600000858306885, 0.7427858114242554, 0.6873330473899841, 0.8799994587898254], got {onnx_probabilities}"

    def test_SklearnModel_classification_y_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised.
        Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class_y("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_classification_xy_preprocessing1(self):
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class_y("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_classification_xy_preprocessing2(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised. This test is different from the previous one, as it checks the case
        where both x and y columns are preprocessed.
        Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised. This test is different from the previous one, as it checks the case
        where both x and y columns are preprocessed.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )

        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_multi_classification_no_preprocessing(self):
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )

        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_multi_classification_x_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint fingerprints for multi-classification.
        Preprocessing is applied only on x features.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.multi_classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_multiclass_df,
            y_cols=None,
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        skl_probabilities = jaqpot_model.predict_proba(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        onnx_probabilities = jaqpot_model.predict_proba_onnx(validation_dataset)

        assert np.array_equal(
            skl_predictions, [0, 2, 0, 1, 0]
        ), f"Expected skl_predictions == [0, 2, 0, 1, 0], got {skl_predictions}"
        assert np.allclose(
            skl_probabilities, [0.54, 0.72, 0.52, 0.72, 0.51], atol=1e-2
        ), (
            f"Expected skl_probabilities == [0.54, 0.72, 0.52, 0.72, 0.51]"
            f"got {skl_probabilities}"
        )

        assert np.array_equal(
            onnx_predictions, [0, 2, 0, 1, 0]
        ), f"Expected onnx_predictions == [0, 2, 0, 1, 0], got {onnx_predictions}"
        assert np.allclose(
            onnx_probabilities,
            [
                0.539999783039093,
                0.7199996113777161,
                0.5199998021125793,
                0.7199996113777161,
                0.5099998116493225,
            ],
            atol=1e-2,
        ), f"Expected onnx_probabilities == [0.539999783039093, 0.7199996113777161, 0.5199998021125793, 0.7199996113777161, 0.5099998116493225], got {onnx_probabilities}"

    def test_SklearnModel_multi_classification_y_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for multi-classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.multi_classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )

        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_multi_classification_xy_preprocessing(self):
        """Test RandomForestClassifier on a molecular dataset with TopologicalFingerprint for multi-classification.

        This test is for checking the error handling regarding the transformations of y labels in
        classification tasks. Specifically, if any scaling method is selected for y labels, an
        error must be raised. This test is different from the previous one, as it checks the case
        where both x and y columns are preprocessed.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.multi_classification_df,
            y_cols=["ACTIVITY"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )
        model = RandomForestClassifier(random_state=42)
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )

        with self.assertRaises(ValueError) as context:
            jaqpot_model.fit()
        self.assertTrue(
            "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
            in str(context.exception)
        )

    def test_SklearnModel_regression_no_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for regression,
        without any preprocessing on data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_df,
            y_cols=["ACTIVITY"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )
        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
        )
        jaqpot_model.fit()
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)

        assert np.allclose(
            skl_predictions, [2145.41, 84.52, 2406.08, 5928.3, 2484.34], atol=1e-03
        ), f"Expected skl_predictions == [2145.41, 84.52, 2406.08, 5928.3, 2484.34], got {skl_predictions}"
        assert np.allclose(
            onnx_predictions,
            [2145.411, 84.51998, 2406.0806, 5928.299, 2484.3413],
            atol=1e-03,
        ), f"Expected onnx_predictions == [2145.411, 84.51998, 2406.0806, 5928.299, 2484.3413], got {onnx_predictions}"

    def test_SklearnModel_regression_x_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for regression, with preprocessing
        applied only on the input features.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_df,
            y_cols=["ACTIVITY"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="binary_classification",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)

        assert np.allclose(
            skl_predictions, [2146.81, 85.24, 2541.61, 5928.3, 2484.34], atol=1e-02
        ), f"Expected skl_predictions == [2146.81, 85.24, 2541.61, 5928.3, 2484.34], got {skl_predictions}"
        assert np.allclose(
            onnx_predictions,
            [2146.811, 85.239975, 2541.6106, 5928.299, 2484.3413],
            atol=1e-02,
        ), f"Expected onnx_predictions == [2146.811, 85.239975, 2541.6106, 5928.299, 2484.3413], got {onnx_predictions}"

    def test_SklearnModel_regression_y_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for regression, with preprocessing
        applied only on the output variables.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )
        pre = Preprocess()
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())
        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit()
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)

        assert np.allclose(
            skl_predictions,
            [59951.485, 5312.43, 2899.398, 15733.59633333, 7775.086],
            atol=1e-02,
        ), f"Expected skl_predictions == [59951.485, 5312.43, 2899.398, 15733.59633333, 7775.086], got {skl_predictions}"
        assert np.allclose(
            onnx_predictions,
            [59951.508, 5312.4307, 2899.3987, 15733.592, 7775.0854],
            atol=1e-02,
        ), f"Expected onnx_predictions == [59951.508, 5312.4307, 2899.3987, 15733.592, 7775.0854], got {onnx_predictions}"

    def test_SklearnModel_regression_xy_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for regression, with preprocessing
        applied on the input and output variables.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_df,
            y_cols=["ACTIVITY"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )
        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())
        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit(onnx_options={StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)

        assert np.allclose(
            skl_predictions,
            [66860.46, 3289.8, 3410.488, 15118.86633333, 7775.086],
            atol=1e-02,
        ), f"Expected skl_predictions == [66860.46, 3289.8, 3410.488, 15118.86633333, 7775.086], got {skl_predictions}"
        assert np.allclose(
            onnx_predictions,
            [66860.48, 3289.8005, 3410.4897, 15118.862, 7775.0854],
            atol=1e-02,
        ), f"Expected onnx_predictions == [66860.48, 3289.8005, 3410.4897, 15118.862, 7775.0854], got {onnx_predictions}"

    def test_SklearnModel_multiple_regression_no_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for multiple output regression,
        without any preprocessing on data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_multioutput_df,
            y_cols=["ACTIVITY", "ACTIVITY_2"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
        )
        jaqpot_model.fit()
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        skl_expected = np.array(
            [
                [1978.8, 54.75],
                [79.76, 67.75],
                [2901.02, 56.84],
                [6655.53, 56.79],
                [3021.47, 51.88],
            ]
        )

        onnx_expected = np.array(
            [
                [1978.8008, 54.74999],
                [79.76001, 67.75001],
                [2901.02, 56.83998],
                [6655.531, 56.78998],
                [3021.4714, 51.879993],
            ]
        )

        assert np.allclose(
            skl_predictions, skl_expected, atol=1e-02
        ), f"Expected skl_predictions == {skl_expected}, got {skl_predictions}"
        assert np.allclose(
            onnx_predictions, onnx_expected, atol=1e-02
        ), f"Expected onnx_predictions == {onnx_expected}, got {onnx_predictions}"

    def test_SklearnModel_multiple_regression_x_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for multiple output regression,
        with preprocessing only on X data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_multioutput_df,
            y_cols=["ACTIVITY", "ACTIVITY_2"],
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())

        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit({StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=["SMILES"],
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        skl_expected = np.array(
            [
                [1979.73, 55.31],
                [81.48, 67.92],
                [3036.58, 56.12],
                [6655.3, 56.53],
                [3021.47, 51.88],
            ]
        )

        onnx_expected = np.array(
            [
                [1979.7308, 55.30999],
                [81.48, 67.920006],
                [3036.58, 56.119984],
                [6655.301, 56.52998],
                [3021.4714, 51.879993],
            ]
        )

        assert np.allclose(
            skl_predictions, skl_expected, atol=1e-02
        ), f"Expected skl_predictions == {skl_expected}, got {skl_predictions}"
        assert np.allclose(
            onnx_predictions, onnx_expected, atol=1e-02
        ), f"Expected onnx_predictions == {onnx_expected}, got {onnx_predictions}"

    def test_SklearnModel_multiple_regression_y_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for multiple output regression,
        with preprocessing only on Y data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_multioutput_df,
            y_cols=["ACTIVITY", "ACTIVITY_2"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        pre = Preprocess()
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit({StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        skl_expected = np.array(
            [
                [1.74146900e04, 8.20650000e01],
                [1.50693300e04, 6.03300000e01],
                [1.46322067e03, 4.09221667e01],
                [1.56947025e05, 1.46026667e01],
                [5.50365600e03, 5.72110000e01],
            ]
        )

        onnx_expected = np.array(
            [
                [1.7414691e04, 8.2064972e01],
                [1.5069329e04, 6.0330032e01],
                [1.4632200e03, 4.0922199e01],
                [1.5694709e05, 1.4602660e01],
                [5.5036543e03, 5.7210995e01],
            ]
        )

        assert np.allclose(
            skl_predictions, skl_expected, atol=1e-02
        ), f"Expected skl_predictions == {skl_expected}, got {skl_predictions}"
        assert np.allclose(
            onnx_predictions, onnx_expected, atol=1e-02
        ), f"Expected onnx_predictions == {onnx_expected}, got {onnx_predictions}"

    def test_SklearnModel_multiple_regression_xy_preprocessing(self):
        """Test RandomForestRegressor on a molecular dataset with TopologicalFingerprint fingerprints for multiple output regression,
        with preprocessing X and Y data.
        """
        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(
            df=self.regression_multioutput_df,
            y_cols=["ACTIVITY", "ACTIVITY_2"],
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        pre = Preprocess()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("minmax_y", MinMaxScaler())

        model = RandomForestRegressor(random_state=42)
        jaqpot_model = SklearnModel(
            dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=pre
        )
        jaqpot_model.fit({StandardScaler: {"div": "div_cast"}})
        validation_dataset = dataset = JaqpotpyDataset(
            df=self.prediction_df,
            y_cols=None,
            smiles_cols=None,
            x_cols=["X1", "X2"],
            task="regression",
            featurizer=featurizer,
        )

        skl_predictions = jaqpot_model.predict(validation_dataset)
        onnx_predictions = jaqpot_model.predict_onnx(validation_dataset)
        skl_expected = np.array(
            [
                [1.77626300e04, 8.14450000e01],
                [1.03573300e04, 6.71550000e01],
                [1.46322067e03, 4.09221667e01],
                [1.55443981e05, 1.60260000e01],
                [5.50365600e03, 5.72110000e01],
            ]
        )

        onnx_expected = np.array(
            [
                [1.7762631e04, 8.1444969e01],
                [1.0357326e04, 6.7155022e01],
                [1.4632200e03, 4.0922199e01],
                [1.5544403e05, 1.6025997e01],
                [5.5036543e03, 5.7210995e01],
            ]
        )

        assert np.allclose(
            skl_predictions, skl_expected, atol=1e-02
        ), f"Expected skl_predictions == {skl_expected}, got {skl_predictions}"
        assert np.allclose(
            onnx_predictions, onnx_expected, atol=1e-02
        ), f"Expected onnx_predictions == {onnx_expected}, got {onnx_predictions}"


if __name__ == "__main__":
    unittest.main()
