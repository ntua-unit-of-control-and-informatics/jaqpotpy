"""
Unified prediction service working with raw model data.

This service takes raw model data (ONNX bytes, preprocessors, metadata) and
produces predictions, ensuring consistent results across all environments.
"""

import logging

import pandas as pd
from jaqpot_api_client import PredictionResponse
from jaqpot_api_client.models.dataset import Dataset
from jaqpotpy.datasets.jaqpot_tabular_dataset import JaqpotTabularDataset
from .handlers.sklearn_onnx_handler import handle_sklearn_onnx_prediction
from .handlers.torch_geometric_handler import handle_torch_geometric_prediction
from .handlers.torch_onnx_handler import handle_torch_onnx_prediction
from .handlers.torch_sequence_handler import handle_torch_sequence_prediction
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
            "SKLEARN_ONNX": handle_sklearn_onnx_prediction,
            "TORCH_ONNX": handle_torch_onnx_prediction,
            "TORCH_SEQUENCE_ONNX": handle_torch_sequence_prediction,
            "TORCH_GEOMETRIC_ONNX": handle_torch_geometric_prediction,
            "TORCHSCRIPT": handle_torch_geometric_prediction,  # TorchScript uses same handler
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

    def get_supported_model_types(self) -> list:
        """
        Get list of supported model types.

        Returns:
            list: List of supported model type strings
        """
        return list(self.handlers.keys())


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
