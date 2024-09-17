import sklearn.pipeline
import copy
from jaqpotpy.models.base_classes import Model
from jaqpotpy.doa.doa import DOA
from typing import Any, Dict, Optional
from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.models import Evaluator, Preprocess
from jaqpotpy.api.get_installed_libraries import get_installed_libraries
from jaqpotpy.api.openapi.jaqpot_api_client.models import FeatureType, ModelType, ModelExtraConfig, Preprocessor, Featurizer
from jaqpotpy.api.openapi.jaqpot_api_client.models.preprocessor_config import PreprocessorConfig
from jaqpotpy.api.openapi.jaqpot_api_client.models.featurizer_config import FeaturizerConfig
from jaqpotpy.api.openapi.jaqpot_api_client.models.preprocessor_config_additional_property import PreprocessorConfigAdditionalProperty
from jaqpotpy.api.openapi.jaqpot_api_client.models.featurizer_config_additional_property import FeaturizerConfigAdditionalProperty
import sklearn
from jaqpotpy.cfg import config
import jaqpotpy
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
import numpy as np
from sklearn.preprocessing import StandardScaler


class SklearnModel(Model):
    def __init__(
        self,
        dataset: JaqpotpyDataset,
        doa: DOA,
        model: Any,
        preprocessor: Preprocess = None,
        evaluator: Evaluator = None,
    ):
        self.x_cols = dataset.x_cols
        self.y_cols = dataset.y_cols
        self.dataset = dataset
        self.featurizer = dataset.featurizer
        self.model = model
        self.pipeline = None
        self.pipeline = None
        self.trained_model = None
        self.doa = doa
        self.trained_doa = None
        self.evaluator = evaluator
        self.preprocess = preprocessor
        self.preprocessing_y = None
        self.transformers_y = {}
        self.libraries = None
        self.version = [sklearn.__version__]
        self.jaqpotpy_version = jaqpotpy.__version__
        self.jaqpotpy_docker = config.jaqpotpy_docker
        self.task = self.dataset.task
        self.onnx_model = None
        self.onnx_opset = None
        self.type = ModelType("SKLEARN")
        self.independentFeatures = None
        self.dependentFeatures = None
        self.extra_config = ModelExtraConfig()

    def __dtypes_to_jaqpotypes__(self):
        for feature in self.independentFeatures + self.dependentFeatures:
            if feature["featureType"] in ["SMILES"]:
                feature["featureType"] = FeatureType.SMILES
            elif feature["featureType"] in ["int", "int64"]:
                feature["featureType"] = FeatureType.INTEGER
            elif feature["featureType"] in ["float", "float64"]:
                feature["featureType"] = FeatureType.FLOAT
            elif feature["featureType"] in ["string, object"]:
                feature["featureType"] = FeatureType.STRING
    
    def _extract_attributes(self, trained_class):
        attributes = trained_class.__dict__
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, (np.int64, np.float64)) else v) for k, v in attributes.items()}

    def _add_class_to_extraconfig(self,added_class, added_class_type):
        if added_class_type == 'preprocessor':
            config = PreprocessorConfig()
            additional_property_type = PreprocessorConfigAdditionalProperty()
        elif added_class_type == 'featurizer':
            config = FeaturizerConfig()
            additional_property_type = FeaturizerConfigAdditionalProperty()

        for attr_name, attr_value in self._extract_attributes(added_class).items():
            additional_property = type(additional_property_type)()
            additional_property.additional_properties['value'] = attr_value
            config.additional_properties[attr_name] = additional_property

        if added_class_type == 'preprocessor':
            self.extra_config.preprocessors.append(
                Preprocessor(
                    name=added_class.__class__.__name__, 
                    config=config
                )
            )
        elif added_class_type == 'featurizer':
            self.extra_config.featurizers.append(
                Featurizer(
                    name=added_class.__class__.__name__,
                    config=config
                )
            )

    def fit(self, onnx_options: Optional[Dict] = None):
        self.libraries = get_installed_libraries()
        if isinstance(self.featurizer, MolecularFeaturizer):
            self.extra_config.featurizers = []
            self._add_class_to_extraconfig(self.featurizer, 'featurizer')

        if self.dataset.y is None:
            raise TypeError(
                "dataset.y is None. Please provide a target variable for the model."
            )
        # Get X and y from dataset
        X = self.dataset.__get_X__()
        y = self.dataset.__get_Y__()
        
        if self.doa:
            self.trained_doa = self.doa.fit(X=X)

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
                if self.task == "classification":
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
                        self._add_class_to_extraconfig(pre_y_function, 'preprocessor')

                    if len(self.dataset.y_cols) == 1:
                        y_scaled = y_scaled.ravel()
                    self.preprocessing_y = preprocess_classes_y
                    self.trained_model = self.pipeline.fit(X.to_numpy(), y_scaled)
            else:
                self.trained_model = self.pipeline.fit(
                    X.to_numpy(), y
                )  # .to_numpy().ravel())
        # case where no preprocessing was provided
        else:
            self.trained_model = self.model.fit(X.to_numpy(), y)  # .to_numpy().ravel())

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
        self.__dtypes_to_jaqpotypes__()

        name = self.model.__class__.__name__ + "_ONNX"
        self.onnx_model = convert_sklearn(
            self.trained_model,
            initial_types=[
            ("float_input", FloatTensorType([None, X.to_numpy().shape[1]]))
            ],
            name=name,
            options={StandardScaler: {"div": "div_cast"}},
        )
        self.onnx_opset = self.onnx_model.opset_import[0].version

        if self.evaluator:
            self.__eval__()
        return self

    def predict(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sklearn_prediction = self.trained_model.predict(
            dataset.X.to_numpy().astype(np.float32)
        )
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y[::-1]:
                    if len(self.y_cols) == 1:
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
        sklearn_probs = self.trained_model.predict_proba(dataset.X.to_numpy())
        sklearn_probs_list = [
            max(sklearn_probs[instance]) for instance in range(len(sklearn_probs))
        ]
        return sklearn_probs_list

    def predict_onnx(self, dataset: JaqpotpyDataset):
        if not isinstance(dataset, JaqpotpyDataset):
            raise TypeError("Expected dataset to be of type JaqpotpyDataset")
        sess = InferenceSession(self.onnx_model.SerializeToString())
        onnx_prediction = sess.run(
            None, {"float_input": dataset.X.to_numpy().astype(np.float32)}
        )
        if len(self.y_cols) == 1:
            onnx_prediction[0] = onnx_prediction[0].reshape(-1, 1)
        if self.preprocess is not None:
            if self.preprocessing_y:
                for f in self.preprocessing_y[::-1]:
                    onnx_prediction[0] = f.inverse_transform(onnx_prediction[0])
        if len(self.y_cols) == 1:
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
        onnx_probs = sess.run(
            None, {"float_input": dataset.X.to_numpy().astype(np.float32)}
        )
        onnx_probs_list = [
            max(onnx_probs[1][instance].values())
            for instance in range(len(onnx_probs[1]))
        ]
        return onnx_probs_list

    def __eval__(self):
        if self.evaluator.dataset.df is not None:
            pass
        else:
            self.evaluator.dataset.create()
        X = self.evaluator.dataset.__get_X__()
        if self.preprocess:
            pre_keys = self.preprocess.classes.keys()
            for pre_key in pre_keys:
                preprocess_func = self.preprocess.fitted_classes.get(pre_key)
                X = preprocess_func.transform(X)
        sess = InferenceSession(self.onnx_model.SerializeToString())
        sess = InferenceSession(self.onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        X = np.array(X.astype(float).copy())
        preds = sess.run(None, {input_name: X.astype(np.float32)})
        preds = preds[0].flatten()
        # preds = self.trained_model.predict(X)
        preds_t = []
        for p in preds:
            try:
                if self.preprocessing_y:
                    for f in self.preprocessing_y:
                        p = f.inverse_transform(p.reshape(1, -1))
                        preds_t.append(p)
            except AttributeError:
                pass
            preds_t.append(p)
        eval_keys = self.evaluator.functions.keys()
        for eval_key in eval_keys:
            eval_function = self.evaluator.functions.get(eval_key)
            try:
                if self.preprocessing_y:
                    for f in self.preprocessing_y:
                        truth = f.inverse_transform(self.evaluator.dataset.__get_Y__())
                    print(eval_key + ": " + str(eval_function(truth, preds_t)))
            except AttributeError:
                print(
                    eval_key
                    + ": "
                    + str(eval_function(self.evaluator.dataset.__get_Y__(), preds_t))
                )
                pass
            # print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds_t)))
            # print(eval_key + ": " + str(eval_function(self.evaluator.dataset.__get_Y__(), preds)))

    def copy(self):
        copied_model = copy.deepcopy(self)
        copied_model.dataset = None
        return copied_model

    def deploy_on_jaqpot(self, jaqpot, name, description, visibility):
        jaqpot.deploy_sklearn_model(
            model=self, name=name, description=description, visibility=visibility
        )
