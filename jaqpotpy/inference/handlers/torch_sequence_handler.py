"""
Torch sequence prediction handler with proper formatting.

This handler deals with PyTorch sequence models (LSTM, RNN, etc.) and formats
the results into the expected response structure.
"""

import numpy as np
import torch
from typing import List, Any
from jaqpot_api_client.models.dataset import Dataset


def handle_torch_sequence_prediction(model_data, dataset: Dataset) -> List[Any]:
    """
    Handle torch sequence prediction with proper formatting.

    Args:
        model_data: OfflineModelData containing model and metadata
        dataset: Raw dataset with input data

    Returns:
        List: Formatted prediction results
    """
    from ..core.predict_methods import predict_torch_sequence
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
    predictions = predict_torch_sequence(model_data, dataset_obj)

    # Format predictions into expected structure
    formatted_predictions = []

    for i, jaqpot_row_id in enumerate(jaqpot_row_ids):
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}

        value = predictions[i] if hasattr(predictions, "__getitem__") else predictions

        for j, feature in enumerate(model_data.model_metadata.dependent_features):
            # Get the specific value for this feature from the prediction array
            feature_value = value[j] if j < len(value) else value[0]

            if isinstance(feature_value, (np.ndarray, torch.Tensor)):
                tensor = (
                    torch.tensor(feature_value)
                    if isinstance(feature_value, np.ndarray)
                    else feature_value
                )
                results[feature.key] = tensor.detach().cpu().numpy().tolist()
            elif isinstance(feature_value, (np.integer, int)):
                results[feature.key] = int(feature_value)
            elif isinstance(feature_value, (np.floating, float)):
                results[feature.key] = float(feature_value)
            else:
                results[feature.key] = feature_value

        results["jaqpotMetadata"] = {"jaqpotRowId": jaqpot_row_id}
        formatted_predictions.append(results)

    return formatted_predictions
