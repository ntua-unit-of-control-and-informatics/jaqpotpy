import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.models import Evaluator, Preprocess
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.models import (
    FeatureType,
    FeaturePossibleValue,
    ModelType,
    ModelExtraConfig,
    Transformer,
    ModelTask,
)
import jaqpotpy
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
from onnxruntime import InferenceSession

from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.models import Preprocess
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.models import (
    FeatureType,
    FeaturePossibleValue,
    ModelType,
    ModelExtraConfig,
    Transformer,
    ModelTask,
)
import jaqpotpy

from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA


class SklearnModel(Model):
    def __init__(
        self,
        dataset: JaqpotpyDataset,
        model: Any,
        doa: Optional[DOA or list] = None,
        preprocessor: Preprocess = None,
        cv=None,
    ):
        self.dataset = dataset
        self.featurizer = dataset.featurizer
        self.model = model
        self.pipeline = None
        self.trained_model = None
        self.doa = doa if isinstance(doa, list) else [doa] if doa else []
        self.preprocess = preprocessor
        self.preprocessing_y = None
        self.transformers_y = {}
        self.libraries = None
        self.version = [sklearn.__version__]
        self.jaqpotpy_version = jaqpotpy.__version__
        self.task = self.dataset.task
        self.initial_types = None
        self.onnx_model = None
        self.onnx_opset = None
        self.type = ModelType("SKLEARN")
        self.independentFeatures = None
        self.dependentFeatures = None
        self.extra_config = ModelExtraConfig()
        self.cv = cv
        self.test_metrics = {}
        self.train_metrics = {}
        self.cross_val_metrics = {}

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
                categories = self.dataset.X[feature["key"]].unique()
                feature["possible_values"] = list(
                    FeaturePossibleValue(category, category) for category in categories
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

        if self.doa:
            self.extra_config.doa = []
            # if not isinstance(self.doa, list):
            #     self.doa = [self.doa]
            for doa_method in self.doa:
                doa_method.fit(X=X)
                self._add_class_to_extraconfig(doa_method, "doa")

        if len(self.dataset.y_cols) == 1:
            y = y.to_numpy().ravel()

        # if preprocessing was applied to either X,y or both
        if self.preprocess is not None:
            self.pipeline = sklearn.pipeline.Pipeline(steps=[])
            # Apply preprocessing on design matrix X
            pre_keys = self.preprocess.classes.keys()
            if len(pre_keys) > 0:
                for preprocessor in pre_keys:
                    self.pipeline.steps.append(
                        (preprocessor, self.preprocess.classes.get(preprocessor))
                    )
            self.pipeline.steps.append(("model", self.model))

            # Apply preprocessing of response vector y
            pre_y_keys = self.preprocess.classes_y.keys()

            if len(pre_y_keys) > 0:
                if (
                    self.task == "BINARY_CLASSIFICATION"
                    or self.task == "MULTICLASS_CLASSIFICATION"
                ):
                    raise ValueError(
                        "Target labels cannot be preprocessed for classification tasks. Remove any assigned preprocessing for y."
                    )
                else:
                    preprocess_names_y = []
                    preprocess_classes_y = []
                    y_scaled = self.dataset.__get_Y__()
                    self.extra_config.preprocessors = []
                    for pre_y_key in pre_y_keys:
                        pre_y_function = self.preprocess.classes_y.get(pre_y_key)
                        y_scaled = pre_y_function.fit_transform(y_scaled)
                        self.preprocess.register_fitted_class_y(
                            pre_y_key, pre_y_function
                        )
                        preprocess_names_y.append(pre_y_key)
                        preprocess_classes_y.append(pre_y_function)
                        self._add_class_to_extraconfig(pre_y_function, "preprocessor")

                    if len(self.dataset.y_cols) == 1:
                        y_scaled = y_scaled.ravel()
                    self.preprocessing_y = preprocess_classes_y
                    self.trained_model = self.pipeline.fit(X, y_scaled)
            else:
                self.trained_model = self.pipeline.fit(X, y)

        # case where no preprocessing was provided
        else:
            self.trained_model = self.model.fit(X, y)

        y_pred = self.predict(self.dataset)
        self.train_metrics = self._get_metrics(y, y_pred)
        print("Goodness-of-fit metrics on training set:")
        print(self.train_metrics)
        if self.cv:
            ...

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
                for feature in self.dataset.x_cols
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
        return self

    def predict(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sklearn_prediction = self.trained_model.predict(dataset.X)
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y[::-1]:
                    if len(self.dataset.y_cols) == 1:
                        sklearn_prediction = f.inverse_transform(
                            sklearn_prediction.reshape(1, -1)
                        ).flatten()
                    else:
                        sklearn_prediction = f.inverse_transform(sklearn_prediction)
        return sklearn_prediction

    def predict_proba(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        if self.task == "regression":
            raise ValueError("predict_proba is available only for classification tasks")
        sklearn_probs = self.trained_model.predict_proba(dataset.X)
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
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y[::-1]:
                    onnx_prediction[0] = f.inverse_transform(onnx_prediction[0])
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

    def _cross_val(self, X, y_true):
        if self.task.upper() == "REGRESSION":
            ...
        else:
            ...
        return metrics

    def evaluate(self, X, y_true):
        if self.task.upper() == "REGRESSION":
            ...
        else:
            ...
        return metrics

    def _get_metrics(self, y_true, y_pred):
        if self.task.upper() == "REGRESSION":
            return SklearnModel._get_regression_metrics(y_true, y_pred)
        else:
            return SklearnModel._get_classification_metrics(y_true, y_pred)

    @staticmethod
    def _get_classification_metrics(y_true, y_pred):
        eval_metrics = {
            "accuracy": metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
            "balanced_accuracy": metrics.balanced_accuracy_score(
                y_true=y_true, y_pred=y_pred
            ),
            "precision": metrics.precision_score(y_true=y_true, y_pred=y_pred),
            "recall": metrics.recall_score(y_true=y_true, y_pred=y_pred),
            "f1": metrics.f1_score(y_true=y_true, y_pred=y_pred),
            "f1_micro": metrics.f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
            "f1_macro": metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
            "jaccard": metrics.jaccard_score(y_true=y_true, y_pred=y_pred),
        }
        return eval_metrics

    @staticmethod
    def _get_regression_metrics(y_true, y_pred):
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(y_true, list):
            y_true = np.array(y_true)

        mae = metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2 = metrics.r2_score(y_true=y_true, y_pred=y_pred)
        rmse = metrics.root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        # Corellation coefficient squared R2
        cor_num_1 = y_true - y_true.mean()
        cor_num_2 = y_pred - y_pred.mean()
        cor_den_1 = ((y_true - y_true.mean()) ** 2).sum()
        cor_den_2 = ((y_pred - y_pred.mean()) ** 2).sum()
        cor_coeff = (cor_num_1 * cor_num_2).sum() / np.sqrt(cor_den_1 * cor_den_2)
        cor_coeff_2 = cor_coeff**2

        # Calculate k and k_hat
        k = ((y_true * y_pred).sum()) / ((y_pred) ** 2).sum()
        k_hat = ((y_true * y_pred).sum()) / ((y_true) ** 2).sum()

        # Calculate R0^2 , R'0^2
        # Calc y_r0, y_hat_r0 # CHECK
        y_r0 = k * y_pred
        y_hat_r0 = k_hat * y_true

        # Calculate R0^2 # CHECK
        R0_2_num = ((y_pred - y_r0) ** 2).sum()
        R0_2_den = ((y_pred - y_pred.mean()) ** 2).sum()
        R0_2 = 1 - R0_2_num / R0_2_den

        # Calculate R'0^2 # CHECK
        R0_2_hat_num = ((y_true - y_hat_r0) ** 2).sum()
        R0_2_hat_den = ((y_true - y_true.mean()) ** 2).sum()
        R0_2_hat = 1 - R0_2_hat_num / R0_2_hat_den

        eval_metrics = {
            "R^2 ": r2,
            "MAE": mae,
            "RMSE": rmse,
            "(R^2 - R0^2_ / R^2 ": (cor_coeff_2 - R0_2) / cor_coeff_2,
            "(R^2-R0_hat^2)/R2": (cor_coeff_2 - R0_2_hat) / cor_coeff_2,
            "|R02^-R0_hat^2|": abs(R0_2 - R0_2_hat),
            "k": k,
            "k_hat": k_hat,
        }

        return eval_metrics

    def deploy_on_jaqpot(self, jaqpot, name, description, visibility, upload_metrics):
        jaqpot.deploy_sklearn_model(
            model=self,
            name=name,
            description=description,
            visibility=visibility,
            upload_metrics=upload_metrics,
        )
