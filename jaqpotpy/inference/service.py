"""
Unified prediction service working with raw model data.

This service takes raw model data (ONNX bytes, preprocessors, metadata) and
produces predictions, ensuring consistent results across all environments.
"""

import logging

import pandas as pd
from jaqpot_api_client import PredictionResponse
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.feature_type import FeatureType

from .core.predict_methods import (
    predict_sklearn_onnx,
    predict_torch_onnx,
    predict_torch_sequence,
    predict_torch_geometric,
)
from jaqpotpy.datasets.jaqpot_tabular_dataset import JaqpotTabularDataset
from .core.model_loader import load_onnx_model_from_bytes
from ..offline.offline_model_data import OfflineModelData

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Unified prediction service working with raw model data.

    This service takes raw model data (ONNX bytes, preprocessors, metadata) and
    produces predictions, ensuring consistent results across all environments.
    """

    def __init__(self):
        """Initialize the prediction service."""
        # Model type to handler mapping
        self.handlers = {
            "SKLEARN_ONNX": self._predict_sklearn_onnx,
            "TORCH_ONNX": self._predict_torch_onnx,
            "TORCH_SEQUENCE_ONNX": self._predict_torch_sequence,
            "TORCH_GEOMETRIC_ONNX": self._predict_torch_geometric,
            "TORCHSCRIPT": self._predict_torch_geometric,  # TorchScript uses same handler
        }

    def predict(
        self, model_data: OfflineModelData, dataset: Dataset, model_type: str
    ) -> PredictionResponse:
        """
        Make predictions using raw model data.

        Args:
            model_data: Raw model data (OfflineModelData or similar structure)
            dataset: Dataset object with input data
            model_type: Type of model (e.g., "SKLEARN_ONNX")

        Returns:
            PredictionResponse: The prediction results

        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in self.handlers:
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            logger.info(f"Starting prediction for model type {model_type}")

            # Get appropriate handler
            handler = self.handlers[model_type]

            # Execute prediction with raw data - each handler formats its own predictions
            predictions = handler(model_data, dataset)

            # Build response
            response = PredictionResponse(predictions=predictions)

            logger.info(
                f"Prediction completed successfully for model type {model_type}"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction failed for model type {model_type}: {str(e)}")
            raise

    def _predict_sklearn_onnx(self, model_data, dataset: Dataset):
        """Handle sklearn ONNX model prediction with raw data."""
        from .handlers.sklearn_onnx_handler import handle_sklearn_onnx_prediction

        return handle_sklearn_onnx_prediction(model_data, dataset)

    def _predict_torch_onnx(self, model_data, dataset: Dataset):
        """Handle PyTorch ONNX model prediction with raw data."""

        # Use specialized handler that handles images and tensor processing
        from .handlers.torch_onnx_handler import handle_torch_onnx_prediction

        # Handler already returns formatted predictions
        return handle_torch_onnx_prediction(model_data, dataset)

    def _predict_torch_sequence(self, model_data, dataset: Dataset):
        """Handle PyTorch sequence model prediction with raw data."""
        from .handlers.torch_sequence_handler import handle_torch_sequence_prediction

        return handle_torch_sequence_prediction(model_data, dataset)

    def _predict_torch_geometric(self, model_data, dataset: Dataset):
        """Handle PyTorch Geometric model prediction with raw data."""
        from .handlers.torch_geometric_handler import handle_torch_geometric_prediction

        # Handler should format its own predictions
        return handle_torch_geometric_prediction(model_data, dataset)

    def get_supported_model_types(self) -> list:
        """
        Get list of supported model types.

        Returns:
            list: List of supported model type strings
        """
        return list(self.handlers.keys())

    def _build_tabular_dataset(
        self, model_data, dataset: Dataset
    ) -> JaqpotTabularDataset:
        """
        Build a JaqpotTabularDataset from raw model data and input dataset.

        This follows the exact same logic as build_tabular_dataset_from_request
        but works with raw model data instead of request objects.

        Args:
            model_data: OfflineModelData with model metadata
            dataset: Dataset with input data

        Returns:
            JaqpotTabularDataset: Dataset ready for prediction
        """
        from .core.preprocessor_utils import recreate_featurizer

        # Create DataFrame from input (matches original logic)
        df = pd.DataFrame(dataset.input)
        jaqpot_row_ids = []

        # Extract row IDs (matches original logic)
        for i in range(len(df)):
            jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])

        independent_features = model_data.model_metadata.independent_features

        # Separate SMILES columns from other features (matches original logic)
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

        # Recreate featurizers if present (matches original logic)
        featurizers = []
        if (
            hasattr(model_data.model_metadata, "featurizers")
            and model_data.model_metadata.featurizers
        ):
            for featurizer in model_data.model_metadata.featurizers:
                featurizer_name = featurizer.name
                featurizer_config = featurizer.config
                featurizer_recreated = recreate_featurizer(
                    featurizer_name, featurizer_config
                )
                featurizers.append(featurizer_recreated)
        else:
            featurizers = None

        # Create dataset (matches original logic)
        dataset_obj = JaqpotTabularDataset(
            df=df,
            smiles_cols=smiles_cols,
            x_cols=x_cols,
            task=model_data.model_metadata.task,
            featurizer=featurizers,
        )

        # Apply feature selection if specified (matches original logic)
        if (
            hasattr(model_data.model_metadata, "selected_features")
            and model_data.model_metadata.selected_features is not None
            and len(model_data.model_metadata.selected_features) > 0
        ):
            dataset_obj.select_features(
                SelectColumns=model_data.model_metadata.selected_features
            )

        return dataset_obj


# Global service instance
_service = None


def get_prediction_service() -> PredictionService:
    """
    Get the global prediction service instance.

    Returns:
        PredictionService: Service instance
    """
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
