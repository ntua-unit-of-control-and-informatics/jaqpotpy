from typing import Union, Dict, Any, List

import numpy as np
from jaqpot_api_client.models.prediction_response import PredictionResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.dataset import Dataset

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

        # Convert local data format to PredictionRequest format for shared service
        request = self._create_prediction_request(model_metadata, data)

        # Use shared prediction service (downloaded model mode with jaqpot client)
        prediction_service = get_prediction_service(
            local_mode=True, jaqpot_client=self.jaqpot_client
        )

        # Execute prediction using shared logic
        return prediction_service.predict(request)

    def _create_prediction_request(self, model_metadata, data) -> PredictionRequest:
        """
        Convert downloaded model format and data to PredictionRequest format.

        Args:
            model_metadata: Model metadata from the Jaqpot API
            data: Input data (numpy array, list, or dict)

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

        # Create dataset object
        dataset = Dataset(input=input_data)

        # Create prediction request
        # Note: For downloaded model mode, we'll set raw_model to the base64 encoded model
        # that we downloaded, so the shared service doesn't need to download again
        model_with_data = type(model_metadata)(
            id=model_metadata.id,
            type=model_metadata.type,
            raw_model=None,  # Will be loaded by presigned URL in downloaded model mode
            raw_preprocessor=getattr(model_metadata, "raw_preprocessor", None),
            doas=getattr(model_metadata, "doas", None),
            # Copy other relevant fields
            **{
                k: v
                for k, v in model_metadata.__dict__.items()
                if k not in ["raw_model", "raw_preprocessor", "doas"]
            },
        )

        return PredictionRequest(model=model_with_data, dataset=dataset)
