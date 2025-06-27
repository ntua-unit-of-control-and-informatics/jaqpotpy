"""
Unified prediction service for both local and production inference.

This service provides a single entry point for all model prediction logic,
ensuring consistent results between local development and production deployment.
"""

import logging
from typing import Optional, Dict, Any, Union
from jaqpot_api_client import PredictionRequest, PredictionResponse

from .core.predict_methods import (
    predict_sklearn_onnx,
    predict_torch_onnx,
    predict_torch_sequence,
    predict_torch_geometric,
)
from .core.dataset_utils import (
    build_tabular_dataset_from_request,
    build_tensor_dataset_from_request,
)
from .core.model_loader import retrieve_onnx_model_from_request


logger = logging.getLogger(__name__)


class PredictionService:
    """
    Unified prediction service supporting both local and production inference.

    This service routes prediction requests to appropriate handlers based on
    model type and provides consistent preprocessing and postprocessing.
    """

    def __init__(self, local_mode: bool = False, jaqpot_client=None, s3_client=None):
        """
        Initialize the prediction service.

        Args:
            local_mode (bool): If True, enables local development mode features
            jaqpot_client: Optional Jaqpot client for local model downloads
            s3_client: Optional S3 client for model downloads (production mode)
        """
        self.local_mode = local_mode
        self.jaqpot_client = jaqpot_client
        self.s3_client = s3_client

        # Model type to handler mapping
        self.handlers = {
            "SKLEARN_ONNX": self._predict_sklearn_onnx,
            "TORCH_ONNX": self._predict_torch_onnx,
            "TORCH_SEQUENCE_ONNX": self._predict_torch_sequence,
            "TORCH_GEOMETRIC_ONNX": self._predict_torch_geometric,
            "TORCHSCRIPT": self._predict_torch_geometric,  # Alias for torch geometric
        }

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Execute prediction for the given request.

        Args:
            request (PredictionRequest): The prediction request

        Returns:
            PredictionResponse: The prediction response

        Raises:
            ValueError: If model type is not supported
            Exception: If prediction fails
        """
        try:
            model_type = getattr(request.model, "model_type", None) or getattr(
                request.model, "type", None
            )

            # Handle ModelType enum values
            if hasattr(model_type, "value"):
                model_type_str = model_type.value
            else:
                model_type_str = str(model_type)

            if model_type_str not in self.handlers:
                raise ValueError(f"Unsupported model type: {model_type}")

            logger.info(
                f"Processing prediction request for model {request.model.id} of type {model_type}"
            )

            # Route to appropriate handler
            handler = self.handlers[model_type_str]
            predictions, probabilities, doa_results = handler(request)

            # Build response
            response = PredictionResponse(
                predictions=predictions.tolist()
                if hasattr(predictions, "tolist")
                else predictions,
                probabilities=probabilities,
                doa=doa_results,
                model_id=request.model.id,
            )

            logger.info(
                f"Prediction completed successfully for model {request.model.id}"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction failed for model {request.model.id}: {str(e)}")
            raise

    def _predict_sklearn_onnx(self, request: PredictionRequest):
        """Handle sklearn ONNX model prediction."""
        # Load model and preprocessor
        model = retrieve_onnx_model_from_request(request, self.s3_client)
        preprocessor = None
        if (
            hasattr(request.model, "raw_preprocessor")
            and request.model.raw_preprocessor
        ):
            # Handle preprocessor loading similar to main model
            preprocessor_request = type(request.model)(
                id=request.model.id, raw_model=request.model.raw_preprocessor
            )
            preprocessor = retrieve_onnx_model_from_request(
                type(request)(model=preprocessor_request, dataset=request.dataset),
                self.s3_client,
            )

        # Build dataset
        dataset, _ = build_tabular_dataset_from_request(request)

        # Run prediction
        return predict_sklearn_onnx(model, preprocessor, dataset, request)

    def _predict_torch_onnx(self, request: PredictionRequest):
        """Handle PyTorch ONNX model prediction."""
        # Load model and preprocessor
        model = retrieve_onnx_model_from_request(request, self.s3_client)
        preprocessor = None
        if (
            hasattr(request.model, "raw_preprocessor")
            and request.model.raw_preprocessor
        ):
            # Handle preprocessor loading
            preprocessor_request = type(request.model)(
                id=request.model.id, raw_model=request.model.raw_preprocessor
            )
            preprocessor = retrieve_onnx_model_from_request(
                type(request)(model=preprocessor_request, dataset=request.dataset),
                self.s3_client,
            )

        # Build dataset
        dataset, _ = build_tensor_dataset_from_request(request)

        # Run prediction
        predictions = predict_torch_onnx(model, preprocessor, dataset, request)
        return predictions, [None] * len(predictions), None

    def _predict_torch_sequence(self, request: PredictionRequest):
        """Handle PyTorch sequence model prediction."""
        # Load model and preprocessor
        model = retrieve_onnx_model_from_request(request, self.s3_client)
        preprocessor = None
        if (
            hasattr(request.model, "raw_preprocessor")
            and request.model.raw_preprocessor
        ):
            # Handle preprocessor loading
            preprocessor_request = type(request.model)(
                id=request.model.id, raw_model=request.model.raw_preprocessor
            )
            preprocessor = retrieve_onnx_model_from_request(
                type(request)(model=preprocessor_request, dataset=request.dataset),
                self.s3_client,
            )

        # Build dataset
        dataset, _ = build_tensor_dataset_from_request(request)

        # Run prediction
        predictions = predict_torch_sequence(model, preprocessor, dataset, request)
        return predictions, [None] * len(predictions), None

    def _predict_torch_geometric(self, request: PredictionRequest):
        """Handle PyTorch Geometric model prediction."""
        # Load model and preprocessor
        model = retrieve_onnx_model_from_request(request, self.s3_client)
        preprocessor = None
        if (
            hasattr(request.model, "raw_preprocessor")
            and request.model.raw_preprocessor
        ):
            # Handle preprocessor loading
            preprocessor_request = type(request.model)(
                id=request.model.id, raw_model=request.model.raw_preprocessor
            )
            preprocessor = retrieve_onnx_model_from_request(
                type(request)(model=preprocessor_request, dataset=request.dataset),
                self.s3_client,
            )

        # Build dataset (could be tensor or graph dataset)
        dataset, _ = build_tensor_dataset_from_request(request)

        # Run prediction
        predictions = predict_torch_geometric(model, preprocessor, dataset, request)
        return predictions, [None] * len(predictions), None

    def validate_request(self, request: PredictionRequest) -> bool:
        """
        Validate a prediction request.

        Args:
            request (PredictionRequest): The request to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            if not request.model or not request.dataset:
                return False

            if not hasattr(request.model, "model_type") or not request.model.model_type:
                return False

            if request.model.model_type not in self.handlers:
                return False

            # Check dataset has input data
            if not hasattr(request.dataset, "input") or not request.dataset.input:
                return False

            return True

        except Exception:
            return False

    def get_supported_model_types(self) -> list:
        """
        Get list of supported model types.

        Returns:
            list: List of supported model type strings
        """
        return list(self.handlers.keys())


# Global prediction service instances for different modes
_local_service = None
_production_service = None


def get_prediction_service(local_mode: bool = False, **kwargs) -> PredictionService:
    """
    Get a prediction service instance.

    Args:
        local_mode (bool): Whether to use local mode
        **kwargs: Additional arguments for service initialization

    Returns:
        PredictionService: Service instance
    """
    global _local_service, _production_service

    if local_mode:
        if _local_service is None:
            _local_service = PredictionService(local_mode=True, **kwargs)
        return _local_service
    else:
        if _production_service is None:
            _production_service = PredictionService(local_mode=False, **kwargs)
        return _production_service
