"""
Sklearn ONNX prediction handler with proper formatting.

This handler deals with sklearn ONNX models and formats the results
into the expected response structure.
"""

from typing import List

import numpy as np
import pandas as pd
from jaqpot_api_client.models.dataset import Dataset

from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.inference.core.preprocessor_utils import recreate_featurizer
from jaqpotpy.offline.offline_model_data import OfflineModelData


def handle_sklearn_onnx_prediction(model_data: OfflineModelData, dataset: Dataset):
    """
    Handle sklearn ONNX prediction with proper formatting.

    Args:
        model_data: OfflineModelData containing model and metadata
        dataset: Raw dataset with input data

    Returns:
        List: Formatted prediction results
    """
    from ..core.predict_methods import predict_sklearn_onnx

    dataset, jaqpot_row_ids = _build_tabular_dataset(model_data, dataset)
    predicted_values, probabilities, doa_predictions = predict_sklearn_onnx(
        model_data, dataset
    )

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        if len(model_data.model_metadata.dependent_features) == 1:
            predicted_values = predicted_values.reshape(-1, 1)
        jaqpot_row_id = int(jaqpot_row_id)
        results = {
            feature.key: int(predicted_values[jaqpot_row_id, i])
            if isinstance(
                predicted_values[jaqpot_row_id, i],
                (np.int16, np.int32, np.int64, np.longlong),
            )
            else float(predicted_values[jaqpot_row_id, i])
            if isinstance(
                predicted_values[jaqpot_row_id, i], (np.float16, np.float32, np.float64)
            )
            else predicted_values[jaqpot_row_id, i]
            for i, feature in enumerate(model_data.model_metadata.dependent_features)
        }
        results["jaqpotMetadata"] = {
            "doa": doa_predictions[jaqpot_row_id] if doa_predictions else None,
            "probabilities": probabilities[jaqpot_row_id],
            "jaqpotRowId": jaqpot_row_id,
        }
        predictions.append(results)

    return predictions


def _build_tabular_dataset(model_data: OfflineModelData, dataset: Dataset):
    import logging

    logger = logging.getLogger(__name__)

    df = pd.DataFrame(dataset.input)
    jaqpot_row_ids = []
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])
    independent_features = model_data.model_metadata.independent_features
    logger.info(
        f"Independent features from model metadata: {[(f.key, f.feature_type) for f in independent_features]}"
    )
    smiles_cols = [
        feature.key
        for feature in independent_features
        if feature.feature_type == "SMILES"
    ] or None
    x_cols = [
        feature.key
        for feature in independent_features
        if feature.feature_type != "SMILES"
    ]
    logger.info(f"SMILES columns identified: {smiles_cols}")
    logger.info(f"X columns identified: {x_cols}")
    logger.info(f"Input DataFrame columns: {list(df.columns)}")
    featurizers = []
    if model_data.model_metadata.featurizers:
        logger.info(
            f"Found {len(model_data.model_metadata.featurizers)} featurizers to recreate"
        )
        for i, featurizer in enumerate(model_data.model_metadata.featurizers):
            featurizer_name = featurizer.name
            featurizer_config = featurizer.config
            logger.info(
                f"Featurizer {i}: {featurizer_name} with config: {featurizer_config}"
            )
            recreated_featurizer = recreate_featurizer(
                featurizer_name, featurizer_config
            )
            logger.info(f"Successfully recreated featurizer {featurizer_name}")
            featurizers.append(recreated_featurizer)
    else:
        logger.info("No featurizers found in model metadata")
        featurizers = None

    dataset = JaqpotTabularDataset(
        df=df,
        smiles_cols=smiles_cols,
        x_cols=x_cols,
        task=model_data.model_metadata.task,
        featurizers=featurizers,
    )

    logger.info(f"Dataset features after creation: {list(dataset.X.columns)}")
    logger.info(f"Dataset shape after creation: {dataset.X.shape}")

    if (
        model_data.model_metadata.selected_features
        and len(model_data.model_metadata.selected_features) > 0
    ):
        logger.info(
            f"Attempting to select features: {model_data.model_metadata.selected_features}"
        )
        logger.info(f"Available features in dataset: {list(dataset.X.columns)}")
        missing_features = [
            f
            for f in model_data.model_metadata.selected_features
            if f not in dataset.X.columns
        ]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
        dataset.select_features(
            SelectColumns=model_data.model_metadata.selected_features
        )
    return dataset, jaqpot_row_ids
