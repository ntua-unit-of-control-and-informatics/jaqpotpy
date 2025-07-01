import base64
from typing import Union, Dict, Any, List

import numpy as np
from jaqpot_api_client.models.prediction_response import PredictionResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_model import PredictionModel
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.dataset_type import DatasetType

from ..inference.service import get_prediction_service


class DownloadedModelPredictor:
    """
    Handles predictions using downloaded models.

    This class is responsible for making predictions using models that have been
    downloaded from the Jaqpot platform, ensuring identical results to production inference.
    """

    def __init__(self, jaqpot_client):
        self.jaqpot_client = jaqpot_client

    def predict(
        self,
        model_data: Union[str, Dict[str, Any]],
        data: Union[np.ndarray, List, Dict],
        model_downloader=None,
    ) -> PredictionResponse:
        """
        Make predictions using a downloaded model.

        This method uses the shared inference service to ensure identical results
        to production inference while maintaining the same external API.

        Args:
            model_data: Either model_id (str) or model data dict from download_model
            data: Input data for prediction (numpy array, list, or dict)
            model_downloader: Optional model downloader instance for fetching models by ID

        Returns:
            PredictionResponse with predictions in same format as Jaqpot API
        """
        # Handle model_id string input
        if isinstance(model_data, str):
            if model_downloader is None:
                raise ValueError(
                    "model_downloader is required when model_data is a model_id string"
                )

            model_id = model_data
            if model_id not in model_downloader._cached_models:
                model_data = model_downloader.download_model(model_id)
            else:
                model_data = model_downloader._cached_models[model_id]

        # Get model metadata
        model_metadata = model_data["model_metadata"]

        # Convert downloaded model format to PredictionRequest format for shared service
        request = self._create_prediction_request(model_metadata, data, model_data)

        # Use shared prediction service (downloaded model mode with jaqpot client)
        prediction_service = get_prediction_service(
            local_mode=True, jaqpot_client=self.jaqpot_client
        )

        # Execute prediction using shared logic
        return prediction_service.predict(request)

    def _create_prediction_request(
        self, model_metadata, data, model_data
    ) -> PredictionRequest:
        """
        Convert downloaded model format and data to PredictionRequest format.

        Args:
            model_metadata: Model metadata from the Jaqpot API
            data: Input data (numpy array, list, or dict)
            model_data: Downloaded model data dict containing onnx_bytes and preprocessor

        Returns:
            PredictionRequest: Request object for shared inference service
        """
        # Convert data to the expected format for the dataset
        if isinstance(data, dict):
            # Convert dict to list of lists format expected by Dataset
            input_data = [list(data.values())]
        elif isinstance(data, list):
            # Ensure it's a list of lists
            if len(data) > 0 and not isinstance(data[0], list):
                input_data = [data]
            else:
                input_data = data
        elif isinstance(data, np.ndarray):
            # Convert numpy array to list of lists
            if data.ndim == 1:
                input_data = [data.tolist()]
            else:
                input_data = data.tolist()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Create dataset object with all required fields
        dataset = Dataset(
            type=DatasetType.PREDICTION, entryType="ARRAY", input=input_data
        )

        # Prepare downloaded model data for inference service
        # The model downloader provides:
        # - onnx_bytes: Raw ONNX bytes that need to be base64 encoded
        # - preprocessor: Deserialized Python object that needs to be pickled and base64 encoded
        raw_model_b64 = None
        raw_preprocessor_b64 = None

        # Encode downloaded ONNX model bytes to base64 (PredictionModel expects base64 string)
        if model_data.get("onnx_bytes"):
            raw_model_b64 = base64.b64encode(model_data["onnx_bytes"]).decode("utf-8")

        # Encode downloaded preprocessor to base64 (PredictionModel expects base64 string)
        if model_data.get("preprocessor"):
            import pickle

            preprocessor_bytes = pickle.dumps(model_data["preprocessor"])
            raw_preprocessor_b64 = base64.b64encode(preprocessor_bytes).decode("utf-8")

        # Create prediction model with downloaded and properly encoded data
        # Get required fields with proper defaults
        independent_features = getattr(model_metadata, "independent_features", [])
        dependent_features = getattr(model_metadata, "dependent_features", [])
        task = getattr(model_metadata, "task", None)

        # Ensure we have the required fields
        if not independent_features:
            independent_features = []
        if not dependent_features:
            dependent_features = []
        if task is None:
            # Default to a reasonable task if not specified
            from jaqpot_api_client.models.model_task import ModelTask

            task = ModelTask.REGRESSION

        prediction_model = PredictionModel(
            id=model_metadata.id,
            type=model_metadata.type,
            independent_features=independent_features,  # Required field
            dependent_features=dependent_features,  # Required field
            task=task,  # Required field
            raw_model=raw_model_b64,  # Base64 encoded ONNX bytes from downloaded model
            raw_preprocessor=raw_preprocessor_b64,  # Base64 encoded pickled preprocessor
            doas=getattr(model_metadata, "doas", None),
            selected_features=getattr(model_metadata, "selected_features", None),
            featurizers=getattr(model_metadata, "featurizers", None),
            preprocessors=getattr(model_metadata, "preprocessors", None),
        )

        return PredictionRequest(model=prediction_model, dataset=dataset)
