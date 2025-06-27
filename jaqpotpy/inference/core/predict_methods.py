"""
Core prediction methods for ONNX models.

This module contains the shared prediction logic used by both local development
and production inference, ensuring consistent results across all environments.
"""

import pandas as pd
import numpy as np
import onnx
import gc
from typing import Optional, Tuple, List, Dict, Any, Union
from onnxruntime import InferenceSession

from jaqpotpy.datasets.jaqpot_tensor_dataset import JaqpotTensorDataset
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.doa import (
    Leverage,
    BoundingBox,
    MeanVar,
    Mahalanobis,
    KernelBased,
    CityBlock,
)

from .preprocessor_utils import recreate_preprocessor


def calculate_doas(
    input_feed: Dict[str, np.ndarray], request
) -> Optional[List[Dict[str, Any]]]:
    """
    Calculate the Domain of Applicability (DoA) for given input data using specified methods.

    Args:
        input_feed (dict): A dictionary containing the input data under the key "input".
        request (object): An object containing the model information, specifically the DoA methods
                          and their corresponding data under the key "model".

    Returns:
        list: A list of dictionaries where each dictionary contains the DoA predictions for a single
              data instance. The keys in the dictionary are the names of the DoA methods used, and
              the values are the corresponding DoA predictions.
    """
    if not hasattr(request.model, "doas") or not request.model.doas:
        return None

    doas_results = []
    input_df = pd.DataFrame(input_feed["input"])

    for _, data_instance in input_df.iterrows():
        doa_instance_prediction = {}

        for doa_data in request.model.doas:
            doa_method = None

            if doa_data.method == "LEVERAGE":
                doa_method = Leverage()
                doa_method.h_star = doa_data.data["hStar"]
                doa_method.doa_matrix = doa_data.data["doaMatrix"]
            elif doa_data.method == "BOUNDING_BOX":
                doa_method = BoundingBox()
                doa_method.bounding_box = doa_data.data["boundingBox"]
            elif doa_data.method == "MEAN_VAR":
                doa_method = MeanVar()
                doa_method.bounds = doa_data.data["bounds"]
            elif doa_data.method == "MAHALANOBIS":
                doa_method = Mahalanobis()
                doa_method._mean_vector = doa_data.data["meanVector"]
                doa_method._inv_cov_matrix = doa_data.data["invCovMatrix"]
                doa_method._threshold = doa_data.data["threshold"]
            elif doa_data.method == "KERNEL_BASED":
                doa_method = KernelBased()
                doa_method._sigma = doa_data.data["sigma"]
                doa_method._gamma = doa_data.data.get("gamma", None)
                doa_method._threshold = doa_data.data["threshold"]
                doa_method._kernel_type = doa_data.data["kernelType"]
                doa_method._data = doa_data.data["dataPoints"]
            elif doa_data.method == "CITY_BLOCK":
                doa_method = CityBlock()
                doa_method._mean_vector = doa_data.data["meanVector"]
                doa_method._threshold = doa_data.data["threshold"]

            if doa_method:
                doa_instance_prediction[doa_method.__class__.__name__] = (
                    doa_method.predict(
                        pd.DataFrame(data_instance.values.reshape(1, -1))
                    )[0]
                )

        # Majority voting
        if len(request.model.doas) > 1:
            in_doa_values = [
                value["inDoa"] for value in doa_instance_prediction.values()
            ]
            doa_instance_prediction["majorityVoting"] = in_doa_values.count(True) > (
                len(in_doa_values) / 2
            )
        else:
            # Access the inDoa value that corresponds to True or False
            doa_instance_prediction["majorityVoting"] = list(
                doa_instance_prediction.values()
            )[0]["inDoa"]

        doas_results.append(doa_instance_prediction)

    return doas_results


def predict_sklearn_onnx(
    model: onnx.ModelProto,
    preprocessor: Optional[onnx.ModelProto],
    dataset: JaqpotTabularDataset,
    request,
) -> Tuple[
    np.ndarray, List[Optional[Dict[str, float]]], Optional[List[Dict[str, Any]]]
]:
    """
    Perform prediction using a scikit-learn ONNX model.

    Parameters:
        model (onnx.ModelProto): The ONNX model to be used for prediction.
        preprocessor (onnx.ModelProto, optional): Optional ONNX preprocessor model.
        dataset (JaqpotTabularDataset): The dataset containing the input features.
        request: A request object containing additional configuration for the prediction,
                including model-specific settings and preprocessors.

    Returns:
        tuple: A tuple containing:
            - ONNX model predictions (np.ndarray)
            - Probability estimates (List[Optional[Dict[str, float]]])
            - DOA results (Optional[List[Dict[str, Any]]])
    """

    # Determine which graph to use for input preparation
    onnx_graph = preprocessor if preprocessor else model

    # Prepare initial types for preprocessing
    input_feed = {}
    for independent_feature in onnx_graph.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if len(onnx_graph.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(np_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(np_dtype)
                .reshape(-1, 1)
            )

    # Apply preprocessor if available
    if preprocessor:
        preprocessor_session = InferenceSession(preprocessor.SerializeToString())
        input_feed = {"input": preprocessor_session.run(None, input_feed)[0]}

    # Calculate DOA if requested
    doas_results = (
        calculate_doas(input_feed, request) if hasattr(request.model, "doas") else None
    )

    # Run main model inference
    model_session = InferenceSession(model.SerializeToString())
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )

    input_feed = {
        model_session.get_inputs()[0].name: input_feed["input"].astype(np_dtype)
    }
    onnx_prediction = model_session.run(None, input_feed)

    # Apply inverse preprocessing if needed
    if hasattr(request.model, "preprocessors") and request.model.preprocessors:
        for preprocessor_config in reversed(request.model.preprocessors):
            preprocessor_name = preprocessor_config.name
            preprocessor_config_data = preprocessor_config.config
            preprocessor_recreated = recreate_preprocessor(
                preprocessor_name, preprocessor_config_data
            )
            if (
                len(request.model.dependent_features) == 1
                and preprocessor_name != "LabelEncoder"
            ):
                onnx_prediction[0] = preprocessor_recreated.inverse_transform(
                    onnx_prediction[0].reshape(-1, 1)
                )
            else:
                onnx_prediction[0] = preprocessor_recreated.inverse_transform(
                    onnx_prediction[0]
                )

    # Flatten predictions if single output
    if len(request.model.dependent_features) == 1:
        onnx_prediction[0] = onnx_prediction[0].flatten()

    # Calculate probabilities for classification tasks
    probs_list = []
    if hasattr(request.model, "task") and request.model.task.lower() in [
        "binary_classification",
        "multiclass_classification",
    ]:
        for instance in onnx_prediction[1]:
            if not isinstance(instance, dict):  # support vector classifier case
                instance = {i: float(p) for i, p in enumerate(instance)}
            rounded_instance = {k: round(v, 3) for k, v in instance.items()}

            # Handle label encoding
            if (
                hasattr(request.model, "preprocessors")
                and request.model.preprocessors
                and request.model.preprocessors[0].name == "LabelEncoder"
            ):
                labels = request.model.preprocessors[0].config["classes_"]
                rounded_instance = {labels[k]: v for k, v in rounded_instance.items()}

            probs_list.append(rounded_instance)
    else:
        probs_list = [None for _ in range(len(onnx_prediction[0]))]

    return onnx_prediction[0], probs_list, doas_results


def predict_torch_onnx(
    model: onnx.ModelProto,
    preprocessor: Optional[onnx.ModelProto],
    dataset: JaqpotTensorDataset,
    request,
) -> np.ndarray:
    """
    Perform prediction using a PyTorch ONNX model.

    Parameters:
        model (onnx.ModelProto): The ONNX model to be used for prediction.
        preprocessor (onnx.ModelProto, optional): Optional ONNX preprocessor model.
        dataset (JaqpotTensorDataset): The dataset containing the input features.
        request: A request object containing additional configuration for the prediction.

    Returns:
        np.ndarray: The ONNX model predictions.
    """

    # Determine which graph to use for input preparation
    onnx_graph = preprocessor if preprocessor else model

    # Prepare initial types for preprocessing
    input_feed = {}
    for independent_feature in onnx_graph.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if len(onnx_graph.graph.input) == 1:
            # Handle object dtype (e.g., arrays inside cells)
            if dataset.X.dtypes[0] == "object":
                input_feed[independent_feature.name] = np.stack(
                    [squeeze_first_dim(x) for x in dataset.X.iloc[:, 0].to_list()]
                ).astype(np_dtype)
            else:
                input_feed[independent_feature.name] = dataset.X.values.astype(np_dtype)
        else:
            if dataset.X[independent_feature.name].dtype == "object":
                input_feed[independent_feature.name] = np.stack(
                    [squeeze_first_dim(x) for x in dataset.X.iloc[:, 0].to_list()]
                ).astype(np_dtype)
            else:
                input_feed[independent_feature.name] = (
                    dataset.X[independent_feature.name]
                    .values.astype(np_dtype)
                    .reshape(-1, 1)
                )

    # Apply preprocessor if available
    if preprocessor:
        preprocessor_session = InferenceSession(preprocessor.SerializeToString())
        input_feed = {"input": preprocessor_session.run(None, input_feed)[0]}

    # Run main model inference
    model_session = InferenceSession(model.SerializeToString())
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )

    input_feed = {
        model_session.get_inputs()[0].name: input_feed["input"].astype(np_dtype)
    }
    onnx_prediction = model_session.run(None, input_feed)

    # Clean up memory
    del model
    del model_session
    gc.collect()  # Trigger garbage collection to free up memory

    return onnx_prediction[0]


def squeeze_first_dim(arr: np.ndarray) -> np.ndarray:
    """
    Remove the first dimension if it's a singleton dimension in 4D arrays.

    Args:
        arr: Input numpy array

    Returns:
        Array with first dimension squeezed if applicable
    """
    return arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr


def predict_torch_sequence(
    model: onnx.ModelProto,
    preprocessor: Optional[onnx.ModelProto],
    dataset: JaqpotTensorDataset,
    request,
) -> np.ndarray:
    """
    Perform prediction using a PyTorch sequence ONNX model (LSTM, RNN, etc.).

    Parameters:
        model (onnx.ModelProto): The ONNX model to be used for prediction.
        preprocessor (onnx.ModelProto, optional): Optional ONNX preprocessor model.
        dataset (JaqpotTensorDataset): The dataset containing the input features.
        request: A request object containing additional configuration for the prediction.

    Returns:
        np.ndarray: The ONNX model predictions.
    """
    # For now, use the same logic as torch_onnx
    # This can be specialized for sequence-specific processing if needed
    return predict_torch_onnx(model, preprocessor, dataset, request)


def predict_torch_geometric(
    model: onnx.ModelProto,
    preprocessor: Optional[onnx.ModelProto],
    dataset: Union[JaqpotTensorDataset, Any],  # Could be graph dataset
    request,
) -> np.ndarray:
    """
    Perform prediction using a PyTorch Geometric ONNX model.

    Parameters:
        model (onnx.ModelProto): The ONNX model to be used for prediction.
        preprocessor (onnx.ModelProto, optional): Optional ONNX preprocessor model.
        dataset: The dataset containing the input features (could be graph-specific).
        request: A request object containing additional configuration for the prediction.

    Returns:
        np.ndarray: The ONNX model predictions.
    """
    # For now, use the same logic as torch_onnx
    # This can be specialized for graph-specific processing if needed
    return predict_torch_onnx(model, preprocessor, dataset, request)
