from typing import Any, Dict, Optional, List, Union
from sklearn import preprocessing, pipeline, compose, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
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
from jaqpotpy.api.openapi.models import (
    ModelScores,
    Scores,
    RegressionScores,
    MulticlassClassificationScores,
    BinaryClassificationScores,
)
from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.models import (
    FeatureType,
    FeaturePossibleValue,
    ModelType,
    Transformer,
    Doa,
)
from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa import DOA


class SklearnModel(Model):
    """
    A class to represent a Scikit-learn model within the Jaqpot framework.

    Attributes
    ----------
    dataset : JaqpotpyDataset
        The dataset used for training the model.
    model : Any
        The Scikit-learn model to be trained.
    doa : Optional[Union[DOA, list]], optional
        Domain of Applicability methods, by default None.
    preprocess_x : Optional[Union[BaseEstimator, List[BaseEstimator]]], optional
        Preprocessors for the input features, by default None.
    preprocess_y : Optional[Union[BaseEstimator, List[BaseEstimator]]], optional
        Preprocessors for the target variable, by default None.
    random_seed : Optional[int], optional
        Random seed for reproducibility, by default 1311.
    pipeline : sklearn.pipeline.Pipeline
        The pipeline that includes preprocessing steps and the model.
    trained_model : Any
        The trained Scikit-learn model.
    transformers_y : dict
        Dictionary to store transformers for the target variable.
    libraries : list
        List of installed libraries.
    jaqpotpy_version : str
        Version of the Jaqpotpy library.
    task : str
        The task type (e.g., regression, classification).
    initial_types : list
        Initial types for ONNX conversion.
    onnx_model : onnx.ModelProto
        The ONNX model.
    type : ModelType
        The type of the model.
    independentFeatures : list
        List of independent features.
    dependentFeatures : list
        List of dependent features.
    featurizers :list
        List of featurizers for the model.
    preprocessors :list
        List of preprocessors for the model.
    test_scores : dict
        Dictionary to store test scores.
    train_scores : dict
        Dictionary to store training scores.
    average_cross_val_scores : dict
        Dictionary to store average cross-validation scores.
    cross_val_scores : dict
        Dictionary to store cross-validation scores.
    randomization_test_results : dict
        Dictionary to store randomization test results.

    Methods
    -------
    _dtypes_to_jaqpotypes():
        Converts data types to Jaqpot feature types.
    _extract_attributes(trained_class, trained_class_type):
        Extracts attributes from a trained class.
    _add_transformer(added_class, added_class_type):
        Adds a class to the transformers list.
    _map_onnx_dtype(dtype, shape=1):
        Maps data types to ONNX tensor types.
    _create_onnx(onnx_options=None):
        Creates an ONNX model.
    fit(onnx_options=None):
        Fits the model to the dataset.
    predict(dataset):
        Predicts using the trained model.
    predict_proba(dataset):
        Predicts probabilities using the trained model.
    predict_onnx(dataset):
        Predicts using the ONNX model.
    predict_proba_onnx(dataset):
        Predicts probabilities using the ONNX model.
    deploy_on_jaqpot(jaqpot, name, description, visibility):
        Deploys the model on the Jaqpot platform.
    check_preprocessor(preprocessor_list, feat_type):
        Checks if the preprocessors are valid.
    cross_validate(dataset, n_splits=5):
        Performs cross-validation on the dataset.
    evaluate(dataset):
        Evaluates the model on a given dataset.
    randomization_test(train_dataset, test_dataset, n_iters=10):
        Performs a randomization test.
    _get_metrics(y_true, y_pred):
        Computes evaluation metrics.
    _get_classification_metrics(y_true, y_pred, binary=True):
        Computes classification metrics.
    _get_regression_metrics(y_true, y_pred):
        Computes regression metrics.
    """

    def __init__(
        self,
        dataset: JaqpotpyDataset,
        model: Any,
        doa: Optional[Union[DOA, list]] = None,
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

    def _add_transformer(self, added_class, added_class_type):
        configurations = {}

        for attr_name, attr_value in self._extract_attributes(
            added_class, added_class_type
        ).items():
            configurations[attr_name] = attr_value

        if added_class_type == "preprocessor":
            self.preprocessors.append(
                Transformer(name=added_class.__class__.__name__, config=configurations)
            )
        elif added_class_type == "featurizer":
            self.featurizers.append(
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

    def _create_onnx_preprocessor(self, onnx_options: Optional[Dict] = None):
        name = self.preprocess_pipeline.__class__.__name__ + "_ONNX"
        self.initial_types_preprocessor = []
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
            self.initial_types_preprocessor = [
                (
                    "input",
                    self._map_onnx_dtype("float32", len(self.dataset.X.columns)),
                )
            ]
        else:
            for i, feature in enumerate(self.dataset.X.columns):
                self.initial_types_preprocessor.append(
                    (
                        self.dataset.X.columns[i],
                        self._map_onnx_dtype(self.dataset.X[feature].dtype.name),
                    )
                )
        self.onnx_preprocessor = convert_sklearn(
            self.preprocess_pipeline,
            initial_types=self.initial_types_preprocessor,
            name=name,
            options=onnx_options,
        )

    def _create_onnx_model(self, onnx_options: Optional[Dict] = None):
        name = self.model.__class__.__name__ + "_ONNX"
        self.initial_types = []
        self.initial_types = [
            (
                "input",
                self._map_onnx_dtype(
                    "float32", self.trained_model.named_steps["model"].n_features_in_
                ),
            )
        ]
        self.onnx_model = convert_sklearn(
            self.trained_model,
            initial_types=self.initial_types,
            name=name,
            options=onnx_options,
        )

    def _labels_are_strings(self, y):
        return (
            (
                self.task.upper()
                in ["BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"]
            )
            and y.dtype in ["object", "string"]
            and isinstance(self.preprocess_y[0], LabelEncoder)
        )

    def fit(self, onnx_options: Optional[Dict] = None):
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
        self.pipeline = pipeline.Pipeline(steps=[])
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

    def predict(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        if self.selected_features is not None:
            X_mat = dataset.X[self.selected_features]
        else:
            X_mat = dataset.X
        sklearn_prediction = self._predict_with_X(X_mat, self.trained_model)
        return sklearn_prediction

    def _predict_with_X(self, X, model):
        if self.preprocess_x:
            X_transformed = self.preprocess_pipeline.transform(X)
            X_transformed = pd.DataFrame(X_transformed)
        else:
            X_transformed = X
        sklearn_prediction = model.predict(X_transformed)
        if self.preprocess_y[0] is not None:
            for func in self.preprocess_y[::-1]:
                if len(self.dataset.y_cols) == 1:
                    if not isinstance(func, LabelEncoder):
                        sklearn_prediction = sklearn_prediction.reshape(-1, 1)
                    sklearn_prediction = func.inverse_transform(
                        sklearn_prediction
                    ).flatten()
                else:
                    sklearn_prediction = func.inverse_transform(sklearn_prediction)
        return sklearn_prediction

    def predict_proba(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        if self.task == "regression":
            raise ValueError("predict_proba is available only for classification tasks")
        if self.selected_features is not None:
            X_mat = dataset.X[self.selected_features]
        else:
            X_mat = dataset.X
        if self.preprocess_x:
            X_mat = self.preprocess_pipeline.transform(X_mat)

        sklearn_probs = self.trained_model.predict_proba(X_mat)

        sklearn_probs_list = [
            max(sklearn_probs[instance]) for instance in range(len(sklearn_probs))
        ]
        return sklearn_probs_list

    def predict_onnx(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sess = InferenceSession(self.onnx_model.SerializeToString())
        if self.preprocess_x:
            X = self.preprocess_pipeline.transform(dataset.X)
        else:
            X = dataset.X.values
        input_dtype = (
            "float32"
            if isinstance(self.initial_types[0][1], FloatTensorType)
            else "string"
        )
        input_data = {sess.get_inputs()[0].name: X.astype(input_dtype)}
        onnx_prediction = sess.run(None, input_data)
        if self.preprocess_y[0] is not None:
            for func in self.preprocess_y[::-1]:
                if len(self.dataset.y_cols) == 1 and not isinstance(func, LabelEncoder):
                    onnx_prediction[0] = onnx_prediction[0].reshape(-1, 1)
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
        if self.preprocess_x:
            X = self.preprocess_pipeline.transform(dataset.X)
        else:
            X = dataset.X.values
        if len(self.initial_types) == 1:
            input_dtype = (
                "float32"
                if isinstance(self.initial_types[0][1], FloatTensorType)
                else "string"
            )
            input_data = {sess.get_inputs()[0].name: X.astype(input_dtype)}
        else:
            input_data = {
                sess.get_inputs()[i].name: X[self.initial_types[i][0]].values.reshape(
                    -1, 1
                )
                for i in range(len(self.initial_types))
            }
        onnx_probs = sess.run(None, input_data)
        onnx_probs_list = [
            max(onnx_probs[1][instance].values())
            for instance in range(len(onnx_probs[1]))
        ]
        return onnx_probs_list

    def predict_doa(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")

        X = dataset.X
        if self.preprocess_x:
            X = self.preprocess_pipeline.transform(X)
        else:
            X = dataset.X.values
        doa_results = {}
        for doa_method in self.doa:
            doa_results[doa_method.__name__] = doa_method.predict(X)

        return doa_results

    def deploy_on_jaqpot(self, jaqpot, name, description, visibility):
        jaqpot.deploy_sklearn_model(
            model=self, name=name, description=description, visibility=visibility
        )

    def _create_jaqpot_scores(self, fit_scores, score_type="train", n_output=1):
        for output in range(n_output):
            y_name = self.dataset.y_cols[output]
            if (n_output - 1) == 0:
                scores = fit_scores
            else:
                scores = fit_scores[y_name]

            if self.task.upper() == "REGRESSION":
                jaqpotScores = Scores(
                    regression=RegressionScores(
                        y_name=y_name,
                        r2=scores["r2"],
                        mae=scores["mae"],
                        rmse=scores["rmse"],
                        rSquaredDiffRZero=scores["rSquaredDiffRZero"],
                        rSquaredDiffRZeroHat=scores["rSquaredDiffRZeroHat"],
                        absDiffRZeroHat=scores["absDiffRZeroHat"],
                        k=scores["k"],
                        kHat=scores["khat"],
                    )
                )
            elif self.task.upper() == "MULTICLASS_CLASSIFICATION":
                jaqpotScores = Scores(
                    multiclass_classification=MulticlassClassificationScores(
                        labels=[str(x) for x in self.trained_model.classes_],
                        y_name=y_name,
                        accuracy=scores["accuracy"],
                        balanced_accuracy=scores["balancedAccuracy"],
                        precision=scores["precision"],
                        recall=scores["recall"],
                        jaccard=scores["jaccard"],
                        f1_score=scores["f1Score"],
                        matthews_corr_coef=scores["matthewsCorrCoef"],
                        confusion_matrix=scores["confusionMatrix"],
                    )
                )
            elif self.task.upper() == "BINARY_CLASSIFICATION":
                jaqpotScores = Scores(
                    binary_classification=BinaryClassificationScores(
                        labels=[str(x) for x in self.trained_model.classes_],
                        y_name=y_name,
                        accuracy=scores["accuracy"],
                        balanced_accuracy=scores["balancedAccuracy"],
                        precision=scores["precision"],
                        recall=scores["recall"],
                        jaccard=scores["jaccard"],
                        f1_score=scores["f1Score"],
                        matthews_corr_coef=scores["matthewsCorrCoef"],
                        confusion_matrix=scores["confusionMatrix"],
                    )
                )

            if score_type == "train":
                if not hasattr(self.scores, "train") or self.scores.train is None:
                    self.scores.train = []
                self.scores.train.append(jaqpotScores)
            elif score_type == "test":
                if not hasattr(self.scores, "test") or self.scores.test is None:
                    self.scores.test = []
                self.scores.test.append(jaqpotScores)
            elif score_type == "cross_validation":
                if (
                    not hasattr(self.scores, "cross_validation")
                    or self.scores.cross_validation is None
                ):
                    self.scores.cross_validation = []
                self.scores.cross_validation.append(jaqpotScores)
            else:
                raise TypeError(
                    "The score_type should be either 'train', 'test' or 'cross_validation'."
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

    def cross_validate(self, dataset: JaqpotpyDataset, n_splits=5):
        if dataset.y.ndim > 1 and dataset.y.shape[1] > 1:
            for output in range(dataset.y.shape[1]):
                y_target = dataset.y.iloc[:, output]
                self.average_cross_val_scores[self.dataset.y_cols[output]] = (
                    self._single_cross_validation(
                        dataset=dataset,
                        y=y_target,
                        n_splits=n_splits,
                        n_output=(output + 1),
                    )
                )
        else:
            self.average_cross_val_scores = self._single_cross_validation(
                dataset=dataset, y=dataset.y, n_splits=n_splits, n_output=1
            )
        self._create_jaqpot_scores(
            self.average_cross_val_scores,
            score_type="cross_validation",
            n_output=dataset.y.shape[1],
        )

        return self.average_cross_val_scores

    def _single_cross_validation(
        self, dataset: JaqpotpyDataset, y, n_splits=5, n_output=1
    ):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        if not self.trained_model:
            raise ValueError(
                "You need to first run SklearnModel.fit() and train a model"
            )

        if self.selected_features is not None:
            X_mat = dataset.X[self.selected_features]
        else:
            X_mat = dataset.X
        sum_metrics = None

        fold = 1
        for train_index, test_index in kf.split(X_mat):
            X_train, X_test = X_mat.iloc[train_index, :], X_mat.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if self.preprocess_y[0] is not None:
                for preprocessor in self.preprocess_y:
                    if isinstance(preprocessor, LabelEncoder):
                        y_train = y_train.to_numpy().flatten()
                    y_train = preprocessor.fit_transform(y_train)
            else:
                y_train = y_train.to_numpy()
            if self.preprocess_x:
                X_train_transformed = self.preprocess_pipeline.transform(X_train)
                X_train_transformed = pd.DataFrame(X_train_transformed)
            else:
                X_train_transformed = X_train
            trained_model = self.pipeline.fit(X_train_transformed, y_train.ravel())
            y_pred = self._predict_with_X(X_test, trained_model).reshape(-1, 1)
            metrics_result = self._get_metrics(
                y_test.to_numpy().ravel(), y_pred.ravel()
            )
            self.cross_val_scores.setdefault(self.dataset.y_cols[n_output - 1], {})
            self.cross_val_scores[self.dataset.y_cols[n_output - 1]][
                "fold_" + str(fold)
            ] = metrics_result
            fold += 1

            if sum_metrics is None:
                sum_metrics = {k: 0 for k in metrics_result.keys()}

            for key, value in metrics_result.items():
                sum_metrics[key] += value

        avg_metrics = {key: value / n_splits for key, value in sum_metrics.items()}

        return avg_metrics

    def evaluate(self, dataset: JaqpotpyDataset):
        if not dataset.y_cols:
            raise ValueError("y_cols must be provided to obtain y_true")
        y_true = dataset.__get_Y__().to_numpy()
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            for output in range(y_true.shape[1]):
                self.test_scores[self.dataset.y_cols[output]] = (
                    self._evaluate_with_model(
                        y_true, dataset.X, self.trained_model, output=output
                    )
                )
        else:
            if y_true.ndim > 1:
                self.test_scores = self._evaluate_with_model(
                    y_true, dataset.X, self.trained_model, output=0
                )
            else:
                self.test_scores = self._evaluate_with_model(
                    y_true.reshape(-1, 1), dataset.X, self.trained_model, output=0
                )
        self._create_jaqpot_scores(
            self.test_scores, score_type="test", n_output=dataset.y.shape[1]
        )
        return self.test_scores

    def _evaluate_with_model(self, y_true, X_mat, model, output=1):
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        y_pred = self._predict_with_X(X_mat, model)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return self._get_metrics(y_true[:, output], y_pred[:, output])

    def randomization_test(
        self, train_dataset: JaqpotpyDataset, test_dataset: JaqpotpyDataset, n_iters=10
    ):
        for iteration in range(n_iters):
            np.random.seed(iteration)
            if train_dataset.y.ndim > 1 and train_dataset.y.shape[1] > 1:
                for output in range(train_dataset.y.shape[1]):
                    y_train_shuffled = np.random.permutation(train_dataset.y)
                    if self.preprocess_y[0] is not None:
                        for preprocessor in self.preprocess_y:
                            y_train_shuffled = preprocessor.fit_transform(
                                y_train_shuffled
                            )
                    if self.preprocess_x:
                        X_transformed = self.preprocess_pipeline.transform(
                            train_dataset.X
                        )
                        X_transformed = pd.DataFrame(X_transformed)
                    else:
                        X_transformed = train_dataset.X
                    trained_model = self.pipeline.fit(
                        X_transformed, y_train_shuffled.ravel()
                    )
                    y_pred_train = self._predict_with_X(
                        train_dataset.X, trained_model
                    ).reshape(-1, 1)
                    train_metrics_result = self._get_metrics(
                        y_train_shuffled.ravel(), y_pred_train.ravel()
                    )
                    test_metrics_result = self._evaluate_with_model(
                        test_dataset.y.to_numpy(),
                        test_dataset.X,
                        trained_model,
                        output=output,
                    )
                    self.randomization_test_results.setdefault(
                        "iteration_" + str(iteration), {}
                    )

                    self.randomization_test_results["iteration_" + str(iteration)][
                        self.dataset.y_cols[output]
                    ] = {"Train": train_metrics_result, "Test": test_metrics_result}

            else:
                y_train_shuffled = np.random.permutation(train_dataset.y)
                if self.preprocess_y[0] is not None:
                    for preprocessor in self.preprocess_y:
                        y_train_shuffled = preprocessor.fit_transform(y_train_shuffled)
                if self.preprocess_x:
                    X_transformed = self.preprocess_pipeline.transform(train_dataset.X)
                    X_transformed = pd.DataFrame(X_transformed)
                else:
                    X_transformed = train_dataset.X
                trained_model = self.pipeline.fit(
                    X_transformed, y_train_shuffled.ravel()
                )
                y_pred_train = self._predict_with_X(
                    train_dataset.X, trained_model
                ).reshape(-1, 1)
                train_metrics_result = self._get_metrics(
                    y_train_shuffled.ravel(), y_pred_train.ravel()
                )
                test_metrics_result = self._evaluate_with_model(
                    test_dataset.y.to_numpy(),
                    test_dataset.X,
                    trained_model,
                    output=0,
                )
                self.randomization_test_results["iteration_" + str(iteration)] = {
                    "Train": train_metrics_result,
                    "Test": test_metrics_result,
                }

        return self.randomization_test_results

    def _get_metrics(self, y_true, y_pred):
        if self.task.upper() == "REGRESSION":
            return SklearnModel._get_regression_metrics(y_true, y_pred)
        if self._labels_are_strings(y_pred):
            y_pred = self.preprocess_y[0].transform(y_pred)
        if self._labels_are_strings(y_true):
            y_true = self.preprocess_y[0].transform(y_true)
        if self.task.upper() == "MULTICLASS_CLASSIFICATION":
            return SklearnModel._get_classification_metrics(
                y_true, y_pred, binary=False
            )
        else:
            return SklearnModel._get_classification_metrics(y_true, y_pred, binary=True)

    @staticmethod
    def _get_classification_metrics(y_true, y_pred, binary=True):
        if binary:
            conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        else:
            conf_mat = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)

        eval_metrics = {
            "accuracy": metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
            "balancedAccuracy": metrics.balanced_accuracy_score(
                y_true=y_true, y_pred=y_pred
            ),
            "precision": metrics.recall_score(
                y_true=y_true, y_pred=y_pred, average=None
            ),
            "recall": metrics.precision_score(
                y_true=y_true, y_pred=y_pred, average=None
            ),
            "f1Score": metrics.f1_score(y_true=y_true, y_pred=y_pred, average=None),
            "jaccard": metrics.jaccard_score(
                y_true=y_true, y_pred=y_pred, average=None
            ),
            "matthewsCorrCoef": metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred),
            "confusionMatrix": conf_mat,
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
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "rSquaredDiffRZero": (cor_coeff_2 - R0_2) / cor_coeff_2,
            "rSquaredDiffRZeroHat": (cor_coeff_2 - R0_2_hat) / cor_coeff_2,
            "absDiffRZeroHat": abs(R0_2 - R0_2_hat),
            "k": k,
            "khat": k_hat,
        }

        return eval_metrics
