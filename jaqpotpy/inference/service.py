"""
Unified prediction service working with raw model data.

This service takes raw model data (ONNX bytes, preprocessors, metadata) and
produces predictions, ensuring consistent results across all environments.
"""

import logging

import pandas as pd
from jaqpot_api_client import PredictionResponse
from jaqpot_api_client.models.dataset import Dataset

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

            # Execute prediction with raw data
            predictions, probabilities, doa_results = handler(model_data, dataset)

            # Build response
            response = PredictionResponse(
                predictions=predictions.tolist()
                if hasattr(predictions, "tolist")
                else predictions,
                probabilities=probabilities,
                doa=doa_results,
                model_id=getattr(model_data, "model_id", None),
            )

            logger.info(
                f"Prediction completed successfully for model type {model_type}"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction failed for model type {model_type}: {str(e)}")
            raise

    def _predict_sklearn_onnx(self, model_data, dataset: Dataset):
        """Handle sklearn ONNX model prediction with raw data."""
        # Build dataset directly from input data
        dataset_obj = self._build_tabular_dataset(model_data, dataset)

        # Run prediction using existing method
        return predict_sklearn_onnx(model_data, dataset_obj)

    def _predict_torch_onnx(self, model_data, dataset: Dataset):
        """Handle PyTorch ONNX model prediction with raw data."""
        # Build dataset directly from input data
        dataset_obj = self._build_tabular_dataset(model_data, dataset)

        # Run prediction using existing method
        predictions = predict_torch_onnx(model_data, dataset_obj)

        # PyTorch models don't return probabilities or DOA results
        return predictions, None, None

    def _predict_torch_sequence(self, model_data, dataset: Dataset):
        """Handle PyTorch sequence model prediction with raw data."""

        # Build dataset directly from input data
        dataset_obj = self._build_tabular_dataset(model_data, dataset)

        # Run prediction using existing method
        predictions = predict_torch_sequence(model_data, dataset_obj)

        # PyTorch models don't return probabilities or DOA results
        return predictions, None, None

    def _predict_torch_geometric(self, model_data, dataset: Dataset):
        """Handle PyTorch Geometric model prediction with raw data."""

        # Use specialized handler that bypasses JaqpotTabularDataset
        from .handlers.torch_geometric_handler import handle_torch_geometric_prediction

        predictions = handle_torch_geometric_prediction(model_data, dataset)

        # PyTorch models don't return probabilities or DOA results
        return predictions, None, None

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

        Args:
            model_data: OfflineModelData with model metadata
            dataset: Dataset with input data

        Returns:
            JaqpotTabularDataset: Dataset ready for prediction
        """
        # Extract jaqpotRowId and data
        jaqpot_row_ids = []
        input_data = []

        for row in dataset.input:
            jaqpot_row_ids.append(row.get("jaqpotRowId", ""))
            # Remove jaqpotRowId from the data for the actual features
            row_data = {k: v for k, v in row.items() if k != "jaqpotRowId"}
            input_data.append(row_data)

        # Get featurizers from model metadata (with safe default)
        featurizers = getattr(model_data.model_metadata, "featurizers", []) or []

        # Extract feature names from feature objects
        x_col_names = [
            getattr(feat, "key", str(feat)) for feat in model_data.independent_features
        ]

        # Create the dataset
        dataset_obj = JaqpotTabularDataset(
            df=pd.DataFrame(input_data),
            x_cols=x_col_names,
            task=model_data.task,
            featurizer=featurizers,
        )

        # Apply feature selection if specified
        selected_features = getattr(
            model_data.model_metadata, "selected_features", None
        )
        if selected_features is not None and len(selected_features) > 0:
            dataset_obj.select_features(SelectColumns=selected_features)

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
