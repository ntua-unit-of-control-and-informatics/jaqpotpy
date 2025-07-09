"""
Sklearn ONNX prediction handler with proper formatting.

This handler deals with sklearn ONNX models and formats the results
into the expected response structure.
"""

import numpy as np
import torch
from typing import List, Any
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.feature_type import FeatureType


def handle_sklearn_onnx_prediction(model_data, dataset: Dataset) -> List[Any]:
    """
    Handle sklearn ONNX prediction with proper formatting.

    Args:
        model_data: OfflineModelData containing model and metadata
        dataset: Raw dataset with input data

    Returns:
        List: Formatted prediction results
    """
    from ..core.predict_methods import predict_sklearn_onnx
    from jaqpotpy.datasets.jaqpot_tabular_dataset import JaqpotTabularDataset
    import pandas as pd

    # Build tabular dataset from input data
    df_data = []
    jaqpot_row_ids = []

    for row in dataset.input:
        jaqpot_row_ids.append(row.get("jaqpotRowId", ""))
        row_data = {k: v for k, v in row.items() if k != "jaqpotRowId"}
        df_data.append(row_data)

    df = pd.DataFrame(df_data)
    x_cols = [feature.key for feature in model_data.model_metadata.independent_features]

    dataset_obj = JaqpotTabularDataset(
        df=df,
        x_cols=x_cols,
        task=model_data.model_metadata.task,
    )

    # Run prediction
    predictions, probabilities, doa_results = predict_sklearn_onnx(
        model_data, dataset_obj
    )

    # Format predictions into expected structure
    formatted_predictions = []

    for i, jaqpot_row_id in enumerate(jaqpot_row_ids):
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}

        # Handle multi-dimensional predictions
        if hasattr(predictions, "ndim") and predictions.ndim > 1:
            if len(model_data.model_metadata.dependent_features) == 1:
                predictions = predictions.reshape(-1, 1)

            # Map predictions to feature keys
            for j, feature in enumerate(model_data.model_metadata.dependent_features):
                pred_value = (
                    predictions[i, j] if predictions.ndim > 1 else predictions[i]
                )

                # Convert to appropriate Python type
                if isinstance(pred_value, (np.int16, np.int32, np.int64, np.longlong)):
                    results[feature.key] = int(pred_value)
                elif isinstance(pred_value, (np.float16, np.float32, np.float64)):
                    results[feature.key] = float(pred_value)
                elif isinstance(pred_value, (torch.Tensor, np.ndarray)):
                    # For tensor/array outputs
                    if isinstance(pred_value, torch.Tensor):
                        results[feature.key] = (
                            pred_value.detach().cpu().numpy().tolist()
                        )
                    else:
                        results[feature.key] = pred_value.tolist()
                else:
                    results[feature.key] = pred_value
        else:
            # Single prediction per row
            pred_value = (
                predictions[i] if hasattr(predictions, "__getitem__") else predictions
            )

            for j, feature in enumerate(model_data.model_metadata.dependent_features):
                if isinstance(pred_value, (np.ndarray, torch.Tensor)):
                    tensor = (
                        torch.tensor(pred_value)
                        if isinstance(pred_value, np.ndarray)
                        else pred_value
                    )
                    results[feature.key] = tensor.detach().cpu().numpy().tolist()
                elif isinstance(
                    pred_value, (np.int16, np.int32, np.int64, np.longlong)
                ):
                    results[feature.key] = int(pred_value)
                elif isinstance(pred_value, (np.float16, np.float32, np.float64)):
                    results[feature.key] = float(pred_value)
                else:
                    results[feature.key] = pred_value

        # Add metadata
        metadata = {"jaqpotRowId": jaqpot_row_id}

        # Add probabilities if available
        if probabilities and i < len(probabilities) and probabilities[i] is not None:
            metadata["probabilities"] = probabilities[i]

        # Add DOA results if available
        if doa_results and i < len(doa_results):
            metadata.update(doa_results[i])

        results["jaqpotMetadata"] = metadata
        formatted_predictions.append(results)

    return formatted_predictions
