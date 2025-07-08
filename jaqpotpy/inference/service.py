"""
Unified prediction service working with raw model data.

This service takes raw model data (ONNX bytes, preprocessors, metadata) and
produces predictions, ensuring consistent results across all environments.
"""

import logging
from jaqpot_api_client import PredictionResponse

from .core.predict_methods import (
    predict_sklearn_onnx,
    predict_torch_onnx,
    predict_torch_sequence,
    predict_torch_geometric,
)
from .core.dataset_utils import build_tabular_dataset_from_request
from .core.model_loader import load_onnx_model_from_bytes


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
        }

    def predict(self, model_data, dataset, model_type: str) -> PredictionResponse:
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

    def _predict_sklearn_onnx(self, model_data, dataset):
        """Handle sklearn ONNX model prediction with raw data."""
        # Load ONNX model from raw bytes
        onnx_model = load_onnx_model_from_bytes(model_data.onnx_bytes)

        # Get preprocessor (raw object)
        preprocessor = model_data.preprocessor

        # Create a mock request for dataset building compatibility
        mock_request = type(
            "MockRequest", (), {"model": model_data.model_metadata, "dataset": dataset}
        )()

        # Build dataset using existing utilities
        dataset_obj, _ = build_tabular_dataset_from_request(mock_request)

        # Run prediction using existing method
        return predict_sklearn_onnx(onnx_model, preprocessor, dataset_obj, mock_request)

    def _predict_torch_onnx(self, model_data, dataset):
        """Handle PyTorch ONNX model prediction with raw data."""
        # Load ONNX model from raw bytes
        onnx_model = load_onnx_model_from_bytes(model_data.onnx_bytes)

        # Get preprocessor (raw object)
        preprocessor = model_data.preprocessor

        # Create a mock request for dataset building compatibility
        mock_request = type(
            "MockRequest", (), {"model": model_data.model_metadata, "dataset": dataset}
        )()

        # Build dataset using existing utilities
        dataset_obj, _ = build_tabular_dataset_from_request(mock_request)

        # Run prediction using existing method
        return predict_torch_onnx(onnx_model, preprocessor, dataset_obj, mock_request)

    def _predict_torch_sequence(self, model_data, dataset):
        """Handle PyTorch sequence model prediction with raw data."""
        # Load ONNX model from raw bytes
        onnx_model = load_onnx_model_from_bytes(model_data.onnx_bytes)

        # Get preprocessor (raw object)
        preprocessor = model_data.preprocessor

        # Create a mock request for dataset building compatibility
        mock_request = type(
            "MockRequest", (), {"model": model_data.model_metadata, "dataset": dataset}
        )()

        # Build dataset using existing utilities
        dataset_obj, _ = build_tabular_dataset_from_request(mock_request)

        # Run prediction using existing method
        return predict_torch_sequence(
            onnx_model, preprocessor, dataset_obj, mock_request
        )

    def _predict_torch_geometric(self, model_data, dataset):
        """Handle PyTorch Geometric model prediction with raw data."""
        # Load ONNX model from raw bytes
        onnx_model = load_onnx_model_from_bytes(model_data.onnx_bytes)

        # Get preprocessor (raw object)
        preprocessor = model_data.preprocessor

        # Create a mock request for dataset building compatibility
        mock_request = type(
            "MockRequest", (), {"model": model_data.model_metadata, "dataset": dataset}
        )()

        # Build dataset using existing utilities
        dataset_obj, _ = build_tabular_dataset_from_request(mock_request)

        # Run prediction using existing method
        return predict_torch_geometric(
            onnx_model, preprocessor, dataset_obj, mock_request
        )

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
