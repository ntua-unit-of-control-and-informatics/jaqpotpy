import base64
import io
import pickle
from typing import Union, Dict, Any, List, Optional

import numpy as np
import onnxruntime as rt
from jaqpot_api_client.api.model_api import ModelApi
from jaqpot_api_client.models.prediction_response import PredictionResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.dataset import Dataset

from ..inference.service import get_prediction_service


class JaqpotDownloadedModel:
    def __init__(self, jaqpot_client):
        self.jaqpot_client = jaqpot_client
        self._cached_models = {}
        self._cached_preprocessors = {}

    def download_model(self, model_id: str, cache: bool = True) -> Dict[str, Any]:
        """
        Download a model from Jaqpot platform for local use.

        Args:
            model_id (str): The ID of the model to download
            cache (bool): Whether to cache the downloaded model

        Returns:
            Dict containing model metadata and ONNX bytes
        """
        if cache and model_id in self._cached_models:
            return self._cached_models[model_id]

        model_api = ModelApi(self.jaqpot_client.http_client)
        model = model_api.get_model(model_id=model_id)

        # Download ONNX model bytes
        onnx_bytes = self._download_model_bytes(model)

        # Download preprocessor if available
        preprocessor = None
        if hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
            preprocessor = self._download_preprocessor_bytes(model)

        model_data = {
            "model_metadata": model,
            "onnx_bytes": onnx_bytes,
            "preprocessor": preprocessor,
            "model_id": model_id,
        }

        if cache:
            self._cached_models[model_id] = model_data

        return model_data

    def _download_model_bytes(self, model) -> bytes:
        """
        Download ONNX model bytes from base64 encoding or S3 presigned URL.
        """
        # First try to get model from database (small models)
        if hasattr(model, "raw_model") and model.raw_model:
            try:
                return model.raw_model
            except Exception:
                # If it's already bytes, return as is
                if isinstance(model.raw_model, bytes):
                    return model.raw_model
                pass

        # Try to get presigned download URL for S3 models
        try:
            # Try to get presigned download URL (this assumes the API client has been updated)
            # For now we'll handle this manually until the API client is regenerated
            import requests

            # Make direct API call to get presigned URL using LocalModelService
            auth_header = self.jaqpot_client.http_client.default_headers.get(
                "Authorization", ""
            )
            headers = {"Authorization": f"Bearer {auth_header.replace('Bearer ', '')}"}
            api_host = self.jaqpot_client.http_client.configuration.host
            response = requests.get(
                f"{api_host}/v1/models/{model.id}/local/download-url",
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                download_info = response.json()
                download_url = download_info["downloadUrl"]

                # Download model from presigned URL
                model_response = requests.get(
                    download_url, timeout=300
                )  # 5 minutes for large models
                model_response.raise_for_status()
                return model_response.content
            elif response.status_code == 404:
                # Model not in S3, might be in database only
                pass
            else:
                response.raise_for_status()

        except Exception as e:
            print(f"Warning: Could not download model from S3 using presigned URL: {e}")

        raise ValueError("Model data not available in database or S3 storage")

    def _download_preprocessor_bytes(self, model) -> Optional[Any]:
        """
        Download and deserialize preprocessor from base64 encoding or S3 presigned URL.
        """
        # First try to get preprocessor from database (small preprocessors)
        if hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
            try:
                preprocessor_bytes = base64.b64decode(model.raw_preprocessor)
                return pickle.loads(preprocessor_bytes)
            except Exception:
                # If base64 decode fails, continue to try presigned URL
                pass

        # Try to get presigned download URL for S3 preprocessors
        try:
            import requests

            # Make direct API call to get presigned URL for preprocessor using LocalModelService
            auth_header = self.jaqpot_client.http_client.default_headers.get(
                "Authorization", ""
            )
            headers = {"Authorization": f"Bearer {auth_header.replace('Bearer ', '')}"}
            api_host = self.jaqpot_client.http_client.configuration.host
            response = requests.get(
                f"{api_host}/v1/models/{model.id}/local/preprocessor/download-url",
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                download_info = response.json()
                download_url = download_info["downloadUrl"]

                # Download preprocessor from presigned URL
                preprocessor_response = requests.get(download_url, timeout=60)
                preprocessor_response.raise_for_status()
                return pickle.loads(preprocessor_response.content)
            elif response.status_code == 404:
                # Preprocessor not found
                return None
            else:
                response.raise_for_status()

        except Exception as e:
            print(
                f"Warning: Could not download preprocessor from S3 using presigned URL: {e}"
            )

        # Fallback: check if we have raw_preprocessor in database
        if hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
            try:
                preprocessor_bytes = base64.b64decode(model.raw_preprocessor)
                return pickle.loads(preprocessor_bytes)
            except Exception as e:
                print(f"Warning: Could not load preprocessor from database: {e}")

        return None

    def predict_local(
        self,
        model_data: Union[str, Dict[str, Any]],
        data: Union[np.ndarray, List, Dict],
    ) -> PredictionResponse:
        """
        Make predictions using a locally downloaded model.

        This method now uses the shared inference service to ensure identical results
        to production inference while maintaining the same external API.

        Args:
            model_data: Either model_id (str) or model data dict from download_model
            data: Input data for prediction (numpy array, list, or dict)

        Returns:
            PredictionResponse with predictions in same format as Jaqpot API
        """
        # Handle model_id string input
        if isinstance(model_data, str):
            model_id = model_data
            if model_id not in self._cached_models:
                model_data = self.download_model(model_id)
            else:
                model_data = self._cached_models[model_id]

        # Get model metadata
        model_metadata = model_data["model_metadata"]

        # Convert local data format to PredictionRequest format for shared service
        request = self._create_prediction_request(model_metadata, data)

        # Use shared prediction service (local mode with jaqpot client)
        prediction_service = get_prediction_service(
            local_mode=True, jaqpot_client=self.jaqpot_client
        )

        # Execute prediction using shared logic
        return prediction_service.predict(request)

    def _create_prediction_request(self, model_metadata, data) -> PredictionRequest:
        """
        Convert local model format and data to PredictionRequest format.

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
        # Note: For local mode, we'll set raw_model to the base64 encoded model
        # that we downloaded, so the shared service doesn't need to download again
        model_with_data = type(model_metadata)(
            id=model_metadata.id,
            type=model_metadata.type,
            raw_model=None,  # Will be loaded by presigned URL in local mode
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

    def clear_cache(self):
        """
        Clear cached models and preprocessors.
        """
        self._cached_models.clear()
        self._cached_preprocessors.clear()

    def list_cached_models(self) -> List[str]:
        """
        Get list of cached model IDs.
        """
        return list(self._cached_models.keys())
