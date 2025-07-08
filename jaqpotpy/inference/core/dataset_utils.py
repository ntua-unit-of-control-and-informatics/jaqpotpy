"""
Dataset utilities for building datasets from prediction requests.

This module contains utilities for converting prediction requests into
Jaqpot dataset objects, handling both tabular and tensor datasets.
"""

import pandas as pd
from typing import List, Tuple, Optional, Union, Any
from jaqpot_api_client import PredictionRequest
from jaqpotpy.datasets.jaqpot_tensor_dataset import JaqpotTensorDataset
from jaqpotpy.datasets import JaqpotTabularDataset

from .preprocessor_utils import recreate_featurizer


def build_tabular_dataset_from_request(
    request: PredictionRequest,
) -> Tuple[JaqpotTabularDataset, List[Any]]:
    """
    Build a JaqpotTabularDataset from a prediction request.

    Args:
        request (PredictionRequest): The prediction request containing dataset and model info

    Returns:
        tuple: A tuple containing:
            - JaqpotTabularDataset: The created dataset
            - List[Any]: List of jaqpot row IDs
    """
    df = pd.DataFrame(request.dataset.input)
    jaqpot_row_ids = []

    # Extract row IDs
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])

    independent_features = request.model.independent_features

    # Separate SMILES columns from other features
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

    # Recreate featurizers if present
    featurizers = []
    if hasattr(request.model, "featurizers") and request.model.featurizers:
        for featurizer in request.model.featurizers:
            featurizer_name = featurizer.name
            featurizer_config = featurizer.config
            featurizer_recreated = recreate_featurizer(
                featurizer_name, featurizer_config
            )
            featurizers.append(featurizer_recreated)
    else:
        featurizers = None

    # Create dataset
    dataset = JaqpotTabularDataset(
        df=df,
        smiles_cols=smiles_cols,
        x_cols=x_cols,
        task=request.model.task,
        featurizer=featurizers,
    )

    # Apply feature selection if specified
    if (
        hasattr(request.model, "selected_features")
        and request.model.selected_features is not None
        and len(request.model.selected_features) > 0
    ):
        dataset.select_features(SelectColumns=request.model.selected_features)

    return dataset, jaqpot_row_ids


def build_tensor_dataset_from_request(
    request: PredictionRequest,
) -> Tuple[JaqpotTensorDataset, List[Any]]:
    """
    Build a JaqpotTensorDataset from a prediction request.

    Args:
        request (PredictionRequest): The prediction request containing dataset and model info

    Returns:
        tuple: A tuple containing:
            - JaqpotTensorDataset: The created dataset
            - List[Any]: List of jaqpot row IDs
    """
    df = pd.DataFrame(request.dataset.input)
    jaqpot_row_ids = []

    # Extract row IDs
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])

    independent_features = request.model.independent_features
    x_cols = [feature.key for feature in independent_features]

    # Create tensor dataset
    dataset = JaqpotTensorDataset(
        df=df,
        x_cols=x_cols,
        task=request.model.task,
    )

    return dataset, jaqpot_row_ids


def extract_feature_columns(
    request: PredictionRequest, feature_type: Optional[str] = None
) -> List[str]:
    """
    Extract feature column names from a prediction request.

    Args:
        request (PredictionRequest): The prediction request
        feature_type (str, optional): Filter by feature type (e.g., "SMILES")

    Returns:
        List[str]: List of feature column names
    """
    independent_features = request.model.independent_features

    if feature_type:
        return [
            feature.key
            for feature in independent_features
            if feature.feature_type == feature_type
        ]
    else:
        return [feature.key for feature in independent_features]
