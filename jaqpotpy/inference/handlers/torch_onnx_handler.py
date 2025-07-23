"""
Torch ONNX prediction handler with proper image processing support.

This handler deals with the specific requirements of torch ONNX models
including image preprocessing and tensor output formatting.
"""

import numpy as np
import torch
import onnx
from typing import List, Any
import onnxruntime
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.feature_type import FeatureType


def convert_tensor_to_base64_image(image_array: torch.Tensor) -> str:
    """
    Converts a tensor of shape [C, H, W] to base64-encoded PNG.
    Supports grayscale (1 channel) or RGB (3 channels).
    """
    from ..utils.image_utils import tensor_to_base64_img

    if isinstance(image_array, np.ndarray):
        image_array = torch.tensor(image_array)

    if image_array.ndim == 3 and image_array.shape[0] in [1, 3]:
        return tensor_to_base64_img(image_array)

    raise ValueError("Expected tensor of shape [C, H, W] with 1 or 3 channels")


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
    from ..utils.image_utils import validate_and_decode_image

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
        row_data = dict(row)  # Keep all data including jaqpotRowId

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

    # Build tensor dataset using the same logic as the original build_tensor_dataset_from_request
    import pandas as pd
    from jaqpotpy.datasets.jaqpot_tensor_dataset import JaqpotTensorDataset

    # Create DataFrame from processed input (same as original)
    df = pd.DataFrame(processed_input)
    jaqpot_row_ids = []
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])

    x_cols = [feature.key for feature in model_data.model_metadata.independent_features]

    tensor_dataset = JaqpotTensorDataset(
        df=df,
        x_cols=x_cols,
        task=model_data.model_metadata.task,
    )

    # Run ONNX inference using the original method
    from ..core.predict_methods import predict_torch_onnx

    predicted_values = predict_torch_onnx(model_data, tensor_dataset)

    # Format predictions with proper image conversion
    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}

        value = predicted_values[jaqpot_row_id]

        for i, feature in enumerate(model_data.model_metadata.dependent_features):
            if isinstance(value, (np.ndarray, torch.Tensor)):
                tensor = torch.tensor(value) if isinstance(value, np.ndarray) else value

                if (
                    dataset.result_types is not None
                    and dataset.result_types.get(feature.key)
                    and feature.feature_type == FeatureType.IMAGE
                ):
                    if tensor.ndim == 4:  # remove batch dim if present
                        tensor = tensor.squeeze(0)

                    if tensor.ndim == 2:  # Handle 2D grayscale images
                        tensor = tensor.unsqueeze(
                            0
                        )  # Add channel dimension [H,W] -> [1,H,W]

                    if tensor.ndim == 3:
                        results[feature.key] = convert_tensor_to_base64_image(tensor)
                    else:
                        raise ValueError(
                            f"Unexpected image tensor shape: {tensor.shape}"
                        )
                else:
                    results[feature.key] = tensor.detach().cpu().numpy().tolist()
            elif isinstance(value, (np.integer, int)):
                results[feature.key] = int(value)
            elif isinstance(value, (np.floating, float)):
                results[feature.key] = float(value)
            else:
                results[feature.key] = value

        results["jaqpotMetadata"] = {"jaqpotRowId": jaqpot_row_id}
        predictions.append(results)

    return predictions


def _squeeze_first_dim(arr: np.ndarray) -> np.ndarray:
    """Remove the first dimension if it's a singleton dimension in 4D arrays."""
    return arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr
