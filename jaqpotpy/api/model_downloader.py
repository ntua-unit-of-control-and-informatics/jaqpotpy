import base64
import io
import pickle
from typing import Union, Dict, Any, List, Optional

import numpy as np
import onnxruntime as rt
from jaqpot_api_client.api.model_api import ModelApi
from jaqpot_api_client.api.model_download_api import ModelDownloadApi


class JaqpotModelDownloader:
    def __init__(self, jaqpot_client):
        self.jaqpot_client = jaqpot_client
        self._cached_models = {}
        self._cached_preprocessors = {}

    def download_model(self, model_id: int, cache: bool = True) -> Dict[str, Any]:
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
        model = model_api.get_model_by_id(id=model_id)

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

        # Try to get presigned download URLs for S3 models using official API client
        try:
            import requests

            model_download_api = ModelDownloadApi(self.jaqpot_client.http_client)
            download_response = model_download_api.get_model_download_urls(
                model_id=model.id, expiration_minutes=30
            )

            if download_response and download_response.model_url:
                download_url = download_response.model_url

                # Download model from presigned URL
                model_response = requests.get(
                    download_url, timeout=300
                )  # 5 minutes for large models
                model_response.raise_for_status()
                return model_response.content

        except Exception as e:
            print(f"Warning: Could not download model from S3 using official API: {e}")

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

        # Try to get presigned download URLs for S3 preprocessors using official API client
        try:
            import requests

            model_download_api = ModelDownloadApi(self.jaqpot_client.http_client)
            download_response = model_download_api.get_model_download_urls(
                model_id=model.id, expiration_minutes=30
            )

            if download_response and download_response.preprocessor_url:
                download_url = download_response.preprocessor_url

                # Download preprocessor from presigned URL
                preprocessor_response = requests.get(download_url, timeout=60)
                preprocessor_response.raise_for_status()
                return pickle.loads(preprocessor_response.content)

        except Exception as e:
            print(
                f"Warning: Could not download preprocessor from S3 using official API: {e}"
            )

        # Fallback: check if we have raw_preprocessor in database
        if hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
            try:
                preprocessor_bytes = base64.b64decode(model.raw_preprocessor)
                return pickle.loads(preprocessor_bytes)
            except Exception as e:
                print(f"Warning: Could not load preprocessor from database: {e}")

        return None

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
