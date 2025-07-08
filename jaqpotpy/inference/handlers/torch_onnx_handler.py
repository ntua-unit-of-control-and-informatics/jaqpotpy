"""
Torch ONNX prediction handler with proper image processing support.

This handler deals with the specific requirements of torch ONNX models
including image preprocessing and tensor output formatting.
"""

import numpy as np
import torch
import onnx
from typing import List, Dict, Any
import onnxruntime
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.feature_type import FeatureType


def handle_torch_onnx_prediction(model_data, dataset: Dataset) -> List[Any]:
    """
    Handle torch ONNX prediction with proper image processing and tensor formatting.

    This includes:
    - Image preprocessing (base64 → numpy arrays)
    - ONNX model inference
    - Proper tensor indexing and output formatting

    Args:
        model_data: OfflineModelData containing model and metadata
        dataset: Raw dataset with input data (may include base64 images)

    Returns:
        List: Raw prediction results properly indexed for each row
    """
    from ..utils.image_utils import validate_and_decode_image, tensor_to_base64_img
    from ..core.model_loader import load_onnx_model_from_bytes

    # Load ONNX model from raw bytes
    model = load_onnx_model_from_bytes(model_data.onnx_bytes)

    # Get preprocessor (raw object)
    preprocessor = model_data.preprocessor

    # Find image features
    image_features = [
        f
        for f in model_data.model_metadata.independent_features
        if f.feature_type == FeatureType.IMAGE
    ]

    # Preprocess dataset input (including image decoding)
    processed_input = []
    jaqpot_row_ids = []

    for row in dataset.input:
        jaqpot_row_ids.append(row.get("jaqpotRowId", ""))
        row_data = {k: v for k, v in row.items() if k != "jaqpotRowId"}

        # Decode images from base64 to numpy arrays
        for f in image_features:
            if f.key in row_data:
                pil_img = validate_and_decode_image(row_data[f.key])  # from base64
                np_img = np.array(pil_img)  # shape [H, W] or [H, W, C]

                # Ensure [H, W, C] with 3 channels
                if np_img.ndim == 2:
                    np_img = np.expand_dims(np_img, axis=-1)  # [H, W, 1]

                if np_img.shape[2] not in [1, 3]:
                    raise ValueError("Only 1 or 3 channel images are supported")

                # Store the numpy image (ready to convert to torch later)
                row_data[f.key] = np_img

        processed_input.append(row_data)

    # Build tensor dataset from processed input (matches original logic)
    tensor_dataset, jaqpot_row_ids = _build_tensor_dataset_from_processed_input(
        processed_input,
        model_data.model_metadata.independent_features,
        model_data.model_metadata.task,
    )

    # Run ONNX inference
    predictions = _run_torch_onnx_inference(model, preprocessor, tensor_dataset)

    # Return predictions in the format expected by the service formatter
    # This should be a list where each element can be indexed by row
    return predictions


def _run_torch_onnx_inference(model, preprocessor, tensor_dataset):
    """Run ONNX inference on torch tensor dataset."""
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
            if tensor_dataset.X.dtypes[0] == "object":
                input_feed[independent_feature.name] = np.stack(
                    [
                        _squeeze_first_dim(x)
                        for x in tensor_dataset.X.iloc[:, 0].to_list()
                    ]
                ).astype(np_dtype)
            else:
                input_feed[independent_feature.name] = tensor_dataset.X.values.astype(
                    np_dtype
                )
        else:
            if tensor_dataset.X[independent_feature.name].dtype == "object":
                input_feed[independent_feature.name] = np.stack(
                    [
                        _squeeze_first_dim(x)
                        for x in tensor_dataset.X.iloc[:, 0].to_list()
                    ]
                ).astype(np_dtype)
            else:
                input_feed[independent_feature.name] = (
                    tensor_dataset.X[independent_feature.name]
                    .values.astype(np_dtype)
                    .reshape(-1, 1)
                )

    # Apply preprocessor if available
    if preprocessor:
        preprocessor_session = onnxruntime.InferenceSession(
            preprocessor.SerializeToString()
        )
        input_feed = {"input": preprocessor_session.run(None, input_feed)[0]}

    # Run main model inference
    model_session = onnxruntime.InferenceSession(model.SerializeToString())
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
    import gc

    gc.collect()

    return onnx_prediction[0]


def _build_tensor_dataset_from_processed_input(
    processed_input, independent_features, task
):
    """
    Build tensor dataset from processed input (matches original build_tensor_dataset_from_request).

    Args:
        processed_input: List of processed row dictionaries
        independent_features: List of feature objects
        task: Model task

    Returns:
        tuple: (JaqpotTensorDataset, jaqpot_row_ids)
    """
    import pandas as pd
    from jaqpotpy.datasets.jaqpot_tensor_dataset import JaqpotTensorDataset

    # Add back jaqpotRowId for dataset creation
    for i, row in enumerate(processed_input):
        row["jaqpotRowId"] = str(i)

    # Create DataFrame (matches original logic)
    df = pd.DataFrame(processed_input)

    # Extract jaqpot_row_ids (matches original logic)
    jaqpot_row_ids = []
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])

    # Extract feature column names (matches original logic)
    x_cols = [feature.key for feature in independent_features]

    # Create tensor dataset (matches original logic)
    dataset = JaqpotTensorDataset(
        df=df,
        x_cols=x_cols,
        task=task,
    )

    return dataset, jaqpot_row_ids


def _squeeze_first_dim(arr: np.ndarray) -> np.ndarray:
    """Remove the first dimension if it's a singleton dimension in 4D arrays."""
    return arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr
