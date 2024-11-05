from typing import Any, Dict, Optional, List, Union
from sklearn.base import BaseEstimator
import numpy as np
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
import jaqpotpy
from jaqpotpy.datasets.jaqpotpy_dataset import JaqpotpyDataset
from jaqpotpy.api.openapi.models import (
    ModelType,
)
from jaqpotpy.doa import DOA
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from xgboost import XGBClassifier, XGBRegressor
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from jaqpotpy.models.sklearn import SklearnModel
from sklearn.preprocessing import StandardScaler


class XGBoostModel(SklearnModel):
    # overrides the parent method from the sklearnmodel for the xgboost model here
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
        if (
            self.task == "BINARY_CLASSIFICATION"
            or self.task == "MULTICLASS_CLASSIFICATION"
        ):
            self._convert_classifier()
        else:
            self._convert_regressor()
        self.onnx_model = convert_sklearn(
            self.trained_model,
            initial_types=self.initial_types,
            name=name,
            options={StandardScaler: {"div": "div_cast"}},
            target_opset={"": 15, "ai.onnx.ml": 1},
        )

        self.onnx_opset = self.onnx_model.opset_import[0].version

    def _convert_regressor(self):
        update_registered_converter(
            model=XGBRegressor,
            alias="XGBRegressor",
            shape_fct=calculate_linear_regressor_output_shapes,
            convert_fct=convert_xgboost,
        )

    def _convert_classifier(self):
        update_registered_converter(
            model=XGBClassifier,
            alias="XGBClassifier",
            shape_fct=calculate_linear_classifier_output_shapes,
            convert_fct=convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
