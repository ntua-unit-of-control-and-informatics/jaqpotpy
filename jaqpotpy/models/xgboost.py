from typing import Any, Dict, Optional, List, Tuple, Union
import pandas as pd
from sklearn import pipeline
from sklearn.base import BaseEstimator
import numpy as np
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from sklearn.calibration import LabelEncoder
import jaqpotpy
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.models.doa import Doa
from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.api.openapi.models import ModelScores, ModelType
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.doa import DOA
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from xgboost import XGBClassifier, XGBRegressor
from onnxmltools.convert.xgboost import (
    convert,
)
from jaqpotpy.models.sklearn import SklearnModel


class XGBoostModel(SklearnModel):
    """
    XGBoostModel class for handling XGBoost models within the Jaqpotpy framework.

    Attributes:
        dataset (JaqpotpyDataset): The dataset used for training the model.
        model (Any): The XGBoost model instance.
        doa (Optional[DOA or list]): Domain of Applicability (DOA) methods.
        preprocess_x (Optional[Union[BaseEstimator, List[BaseEstimator]]]): Preprocessing steps for input features.
        preprocess_y (Optional[Union[BaseEstimator, List[BaseEstimator]]]): Preprocessing steps for target features.
    """

    def __init__(
        self,
        dataset: JaqpotpyDataset,
        model: Any,
        doa: Optional[DOA or list] = None,
        preprocess_x: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None,
        preprocess_y: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None,
        random_seed: Optional[int] = 1311,
    ):
        self.dataset = dataset
        self.featurizer = dataset.featurizer
        self.random_seed = random_seed
        self.model = model
        self.preprocess_pipeline = None
        self.pipeline = None
        self.trained_model = None
        self.doa = doa if isinstance(doa, list) else [doa] if doa else None
        self.doa_data = None
        self.preprocess_x = (
            (preprocess_x if isinstance(preprocess_x, list) else [preprocess_x])
            if preprocess_x
            else None
        )
        if self.preprocess_x is not None:
            SklearnModel.check_preprocessor(self.preprocess_x, feat_type="X")
        self.preprocess_y = (
            preprocess_y if isinstance(preprocess_y, list) else [preprocess_y]
        )
        SklearnModel.check_preprocessor(self.preprocess_y, feat_type="y")
        self.transformers_y = {}
        self.libraries = None
        self.jaqpotpy_version = jaqpotpy.__version__
        self.task = self.dataset.task
        self.initial_types_preprocessor = None
        self.initial_types = None
        self.onnx_preprocessor = None
        self.onnx_model = None
        self.type = ModelType("SKLEARN")
        self.independentFeatures = None
        self.dependentFeatures = None
        self.featurizers = []
        self.preprocessors = []
        self.test_scores = {}
        self.train_scores = {}
        self.average_cross_val_scores = {}
        self.cross_val_scores = {}
        self.randomization_test_results = {}
        # In the case the attribute does not exist initialize to None
        # This is to be compatible for older models without feat selection
        try:
            self.selected_features = self.dataset.selected_features
        except AttributeError:
            self.selected_features = None
        self.scores = ModelScores()

    def _create_onnx_model(self, onnx_options: Optional[Dict] = None):
        name = self.model.__class__.__name__ + "_ONNX"
        self.initial_types = [
            (
                "input",
                self._map_onnx_dtype("float32", self.trained_model.n_features_in_),
            )
        ]
        self.onnx_model = convert(
            self.trained_model,
            name,
            self.initial_types,
            target_opset=15,
        )

    def fit(
        self,
        eval_set: Optional[List[JaqpotpyDataset]] = None,
        onnx_options: Optional[Dict] = None,
    ):
        self.libraries = get_installed_libraries()
        if isinstance(self.featurizer, (MolecularFeaturizer, list)):
            if not isinstance(self.featurizer, list):
                self.featurizer = [self.featurizer]
            self.featurizers = []
            for featurizer_i in self.featurizer:
                self._add_transformer(featurizer_i, "featurizer")

        if self.dataset.y is None:
            raise TypeError(
                "dataset.y is None. Please provide a target variable for the model."
            )
        # Get X and y from dataset
        X = self.dataset.__get_X__()
        y = self.dataset.__get_Y__()
        y = y.to_numpy()

        if self.preprocess_x is not None:
            self.preprocess_pipeline = pipeline.Pipeline(steps=[])
            for preprocessor in self.preprocess_x:
                self.preprocess_pipeline.steps.append(
                    (preprocessor.__class__.__name__, preprocessor)
                )
            self.preprocess_pipeline.fit(X)
            self._create_onnx_preprocessor(onnx_options=onnx_options)

        if self.doa:
            self.doa_data = []
            if self.preprocess_x:
                x_doa = self.preprocess_pipeline.transform(X)
            else:
                x_doa = X
            for i, doa_method in enumerate(self.doa):
                doa_method.fit(X=x_doa)
                self.doa[i] = doa_method
                doa_instance = Doa(
                    method=doa_method.__name__,
                    data=doa_method.doa_attributes,
                )
                self.doa_data.append(doa_instance)

        #  Build preprocessing pipeline that ends up with the model
        self.pipeline = self.model

        # Apply preprocessing of response vector y
        if self.preprocess_y[0] is not None:
            if (
                self.task == "BINARY_CLASSIFICATION"
                or self.task == "MULTICLASS_CLASSIFICATION"
            ) and not isinstance(self.preprocess_y[0], LabelEncoder):
                raise ValueError(
                    "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
                )
            else:
                self.preprocessors = []
                if len(self.dataset.y_cols) == 1 and self._labels_are_strings(y):
                    y = y.ravel()  # this transformation is exclusively for LabelEncoder which is the only allowed preprocessor for y in classification tasks
                for preprocessor in self.preprocess_y:
                    y = preprocessor.fit_transform(y)
                    self._add_transformer(preprocessor, "preprocessor")
        if len(self.dataset.y_cols) == 1 and y.ndim == 2:
            y = y.ravel()
        if self.preprocess_x:
            X_transformed = self.preprocess_pipeline.transform(X)
            X_transformed = pd.DataFrame(X_transformed)
        else:
            X_transformed = X

        X_transformed = X_transformed.to_numpy()
        if eval_set is not None:
            X_eval = []
            y_eval = []
            for dataset in eval_set:
                X_eval.append(dataset.__get_X__().to_numpy())
                y_eval.append(dataset.__get_Y__())
            self.trained_model = self.pipeline.fit(
                X_transformed,
                y,
                eval_set=[(X_eval[i], y_eval[i]) for i in range(len(X_eval))],
            )
        else:
            self.trained_model = self.pipeline.fit(X_transformed, y)
        y_pred = self.predict(self.dataset)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            for output in range(y_pred.shape[1]):
                self.train_scores[self.dataset.y_cols[output]] = self._get_metrics(
                    y[:, output], y_pred[:, output]
                )
                print(f"Goodness-of-fit metrics of output {output} on training set:")
                print(self.train_scores[self.dataset.y_cols[output]])
        else:
            self.train_scores = self._get_metrics(y, y_pred)
            print("Goodness-of-fit metrics on training set:")
            print(self.train_scores)
        self._create_jaqpot_scores(
            self.train_scores, score_type="train", n_output=self.dataset.y.shape[1]
        )

        if self.dataset.smiles_cols:
            self.independentFeatures = list(
                {
                    "key": self.dataset.smiles_cols[smiles_col_i],
                    "name": self.dataset.smiles_cols[smiles_col_i],
                    "featureType": "SMILES",
                }
                for smiles_col_i in range(len(self.dataset.smiles_cols))
            )
        else:
            self.independentFeatures = list()
        if self.dataset.x_cols:
            if self.selected_features is not None:
                intesection_of_features = [
                    feature
                    for feature in self.dataset.x_cols
                    if feature in self.selected_features
                ]
            else:
                intesection_of_features = intesection_of_features = [
                    feature for feature in self.dataset.x_cols
                ]
            self.independentFeatures += list(
                {"key": feature, "name": feature, "featureType": X[feature].dtype}
                for feature in intesection_of_features
            )
        self.dependentFeatures = list(
            {
                "key": feature,
                "name": feature,
                "featureType": self.dataset.__get_Y__()[feature].dtype,
            }
            for feature in self.dataset.y_cols
        )
        self._dtypes_to_jaqpotypes()
        self._create_onnx_model(onnx_options=onnx_options)
