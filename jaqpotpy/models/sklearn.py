from typing import Any, Dict, Optional, List, Union
from sklearn import preprocessing, pipeline, compose
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator

import numpy as np
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    DoubleTensorType,
    Int64TensorType,
    Int32TensorType,
    Int8TensorType,
    UInt8TensorType,
    BooleanTensorType,
    StringTensorType,
)
import jaqpotpy
from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.models import (
    FeatureType,
    FeaturePossibleValue,
    ModelType,
    ModelExtraConfig,
    Transformer,
    ModelTask,
)
from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA


class SklearnModel(Model):
    def __init__(
        self,
        dataset: JaqpotpyDataset,
        model: Any,
        doa: Optional[DOA or list] = None,
        preprocess_x: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None,
        preprocess_y: Optional[Union[BaseEstimator, List[BaseEstimator]]] = None,
    ):
        self.dataset = dataset
        self.featurizer = dataset.featurizer
        self.model = model
        self.pipeline = None
        self.trained_model = None
        self.doa = doa if isinstance(doa, list) else [doa] if doa else []
        self.preprocess_x = (
            preprocess_x if isinstance(preprocess_x, list) else [preprocess_x]
        )
        SklearnModel.check_preprocessor(self.preprocess_x, feat_type="X")
        self.preprocess_y = (
            preprocess_y if isinstance(preprocess_y, list) else [preprocess_y]
        )
        SklearnModel.check_preprocessor(self.preprocess_y, feat_type="y")
        self.transformers_y = {}
        self.libraries = None
        self.jaqpotpy_version = jaqpotpy.__version__
        self.task = self.dataset.task
        self.initial_types = None
        self.onnx_model = None
        self.onnx_opset = None
        self.type = ModelType("SKLEARN")
        self.independentFeatures = None
        self.dependentFeatures = None
        self.extra_config = ModelExtraConfig()

    def _dtypes_to_jaqpotypes(self):
        for feature in self.independentFeatures + self.dependentFeatures:
            if feature["featureType"] in ["SMILES"]:
                feature["featureType"] = FeatureType.SMILES
            elif feature["featureType"] in ["int", "int64"]:
                feature["featureType"] = FeatureType.INTEGER
            elif feature["featureType"] in ["float", "float64"]:
                feature["featureType"] = FeatureType.FLOAT
            elif feature["featureType"] in ["string, object", "O"]:
                feature["featureType"] = FeatureType.CATEGORICAL
                categories = self.dataset.df[feature["key"]].unique()
                feature["possible_values"] = list(
                    FeaturePossibleValue(key=category, value=category)
                    for category in categories
                )

    def _extract_attributes(self, trained_class, trained_class_type):
        if trained_class_type == "doa":
            attributes = trained_class._doa_attributes
        elif trained_class_type == "featurizer":
            attributes = self.dataset.featurizers_attributes.get(
                trained_class.__class__.__name__
            )
        else:
            attributes = trained_class.__dict__
        return {
            k: (
                v.tolist()
                if isinstance(v, np.ndarray)
                else v.item()
                if isinstance(v, (np.int64, np.float64))
                else v
            )
            for k, v in attributes.items()
        }

    def _add_class_to_extraconfig(self, added_class, added_class_type):
        configurations = {}

        for attr_name, attr_value in self._extract_attributes(
            added_class, added_class_type
        ).items():
            configurations[attr_name] = attr_value

        if added_class_type == "preprocessor":
            self.extra_config.preprocessors.append(
                Transformer(name=added_class.__class__.__name__, config=configurations)
            )
        elif added_class_type == "featurizer":
            self.extra_config.featurizers.append(
                Transformer(name=added_class.__class__.__name__, config=configurations)
            )
        elif added_class_type == "doa":
            self.extra_config.doa.append(
                Transformer(name=added_class.__class__.__name__, config=configurations)
            )

    def _map_onnx_dtype(self, dtype, shape=1):
        if dtype == "int64":
            return Int64TensorType(shape=[None, shape])
        elif dtype == "int32":
            return Int32TensorType(shape=[None, shape])
        elif dtype == "int8":
            return Int8TensorType(shape=[None, shape])
        elif dtype == "uint8":
            return UInt8TensorType(shape=[None, shape])
        elif dtype == "bool":
            return BooleanTensorType(shape=[None, shape])
        elif dtype == "float32":
            return FloatTensorType(shape=[None, shape])
        elif dtype == "float64":
            return DoubleTensorType(shape=[None, shape])
        elif dtype in ["string", "object", "category"]:
            return StringTensorType(shape=[None, shape])
        else:
            return None

    def _create_onnx(self, onnx_options: Optional[Dict] = None):
        name = self.model.__class__.__name__ + "_ONNX"
        self.initial_types = []
        dtype_array = self.dataset.X.dtypes.values
        dtype_str_array = np.array([str(dtype) for dtype in dtype_array])
        all_numerical = all(
            dtype
            in [
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float16",
                "float32",
                "float64",
                "bool",
            ]
            for dtype in dtype_str_array
        )
        if all_numerical:
            self.initial_types = [
                (
                    "input",
                    self._map_onnx_dtype("float32", len(self.dataset.X.columns)),
                )
            ]
        else:
            for i, feature in enumerate(self.dataset.X.columns):
                self.initial_types.append(
                    (
                        self.dataset.X.columns[i],
                        self._map_onnx_dtype(self.dataset.X[feature].dtype.name),
                    )
                )

        self.onnx_model = convert_sklearn(
            self.trained_model,
            initial_types=self.initial_types,
            name=name,
            options=onnx_options,
        )
        self.onnx_opset = self.onnx_model.opset_import[0].version

    def fit(self, onnx_options: Optional[Dict] = None):
        self.libraries = get_installed_libraries()
        if isinstance(self.featurizer, (MolecularFeaturizer, list)):
            if not isinstance(self.featurizer, list):
                self.featurizer = [self.featurizer]
            self.extra_config.featurizers = []
            for featurizer_i in self.featurizer:
                self._add_class_to_extraconfig(featurizer_i, "featurizer")

        if self.dataset.y is None:
            raise TypeError(
                "dataset.y is None. Please provide a target variable for the model."
            )
        # Get X and y from dataset
        X = self.dataset.__get_X__()
        y = self.dataset.__get_Y__()
        if len(self.dataset.y_cols) == 1:
            y = y.to_numpy().ravel()

        if self.doa:
            self.extra_config.doa = []
            # if not isinstance(self.doa, list):
            #     self.doa = [self.doa]
            for doa_method in self.doa:
                doa_method.fit(X=X)
                self._add_class_to_extraconfig(doa_method, "doa")

        #  Build preprocessing pipeline that ends up with the model
        self.pipeline = pipeline.Pipeline(steps=[])
        if self.preprocess_x[0] is not None:
            for preprocessor in self.preprocess_x:
                self.pipeline.steps.append((str(preprocessor), preprocessor))
        self.pipeline.steps.append(("model", self.model))

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
                self.extra_config.preprocessors = []
                y = self.dataset.__get_Y__()
                for preprocessor in self.preprocess_y:
                    y = preprocessor.fit_transform(y)
                    self._add_class_to_extraconfig(preprocessor, "preprocessor")
                if len(self.dataset.y_cols) == 1:
                    y = y.ravel()
        self.trained_model = self.pipeline.fit(X, y)

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
            self.independentFeatures += list(
                {"key": feature, "name": feature, "featureType": X[feature].dtype}
                for feature in self.dataset.active_features
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

        self._create_onnx(onnx_options=onnx_options)

    def predict(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sklearn_prediction = self.trained_model.predict(
            dataset.X[self.dataset.active_features]
        )
        if self.preprocess_y[0] is not None:
            for func in self.preprocess_y[::-1]:
                if len(self.dataset.y_cols) == 1:
                    sklearn_prediction = func.inverse_transform(
                        sklearn_prediction.reshape(-1, 1)
                    ).flatten()
                else:
                    sklearn_prediction = func.inverse_transform(sklearn_prediction)
        return sklearn_prediction

    def predict_proba(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        if self.task == "regression":
            raise ValueError("predict_proba is available only for classification tasks")

        sklearn_probs = self.trained_model.predict_proba(
            dataset.X[self.dataset.active_features]
        )

        sklearn_probs_list = [
            max(sklearn_probs[instance]) for instance in range(len(sklearn_probs))
        ]
        return sklearn_probs_list

    def predict_onnx(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sess = InferenceSession(self.onnx_model.SerializeToString())
        if len(self.initial_types) == 1:
            input_dtype = (
                "float32"
                if isinstance(self.initial_types[0][1], FloatTensorType)
                else "string"
            )
            input_data = {
                sess.get_inputs()[0].name: dataset.X.values.astype(input_dtype)
            }
        else:
            input_data = {
                sess.get_inputs()[i].name: dataset.X[
                    self.initial_types[i][0]
                ].values.reshape(-1, 1)
                for i in range(len(self.initial_types))
            }
        onnx_prediction = sess.run(None, input_data)
        if len(self.dataset.y_cols) == 1:
            onnx_prediction[0] = onnx_prediction[0].reshape(-1, 1)
        if self.preprocess_y[0] is not None:
            for func in self.preprocess_y[::-1]:
                onnx_prediction[0] = func.inverse_transform(onnx_prediction[0])
        if len(self.dataset.y_cols) == 1:
            return onnx_prediction[0].flatten()
        return onnx_prediction[0]

    def predict_proba_onnx(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        if self.task == "regression":
            raise ValueError(
                "predict_onnx_proba is available only for classification tasks"
            )
        sess = InferenceSession(self.onnx_model.SerializeToString())
        if len(self.initial_types) == 1:
            input_dtype = (
                "float32"
                if isinstance(self.initial_types[0][1], FloatTensorType)
                else "string"
            )
            input_data = {
                sess.get_inputs()[0].name: dataset.X.values.astype(input_dtype)
            }
        else:
            input_data = {
                sess.get_inputs()[i].name: dataset.X[
                    self.initial_types[i][0]
                ].values.reshape(-1, 1)
                for i in range(len(self.initial_types))
            }
        onnx_probs = sess.run(None, input_data)
        onnx_probs_list = [
            max(onnx_probs[1][instance].values())
            for instance in range(len(onnx_probs[1]))
        ]
        return onnx_probs_list

    def deploy_on_jaqpot(self, jaqpot, name, description, visibility):
        jaqpot.deploy_sklearn_model(
            model=self, name=name, description=description, visibility=visibility
        )

    @staticmethod
    def check_preprocessor(preprocessor_list: List, feat_type: str):
        # Get all valid preprocessing classes from sklearn.preprocessing
        valid_preprocessing_classes_1 = [
            getattr(preprocessing, name)
            for name in dir(preprocessing)
            if isinstance(getattr(preprocessing, name), type)
        ]
        valid_preprocessing_classes_2 = [
            getattr(compose, name)
            for name in dir(compose)
            if isinstance(getattr(compose, name), type)
        ]
        valid_preprocessing_classes = (
            valid_preprocessing_classes_1 + valid_preprocessing_classes_2
        )

        for preprocessor in preprocessor_list:
            # Check if preprocessor is an instance of one of these classes
            if (
                not isinstance(preprocessor, tuple(valid_preprocessing_classes))
                and preprocessor is not None
            ):
                if feat_type == "X":
                    raise ValueError(
                        f"Feature preprocessing must be an instance of a valid class from sklearn.preprocessing, but got {type(preprocessor)}."
                    )
                elif feat_type == "y":
                    raise ValueError(
                        f"Response preprocessing must be an instance of a valid class from sklearn.preprocessing, but got {type(preprocessor)}."
                    )
