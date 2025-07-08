import pickle
from typing import Dict, List, Optional

from jaqpot_api_client import Model
from jaqpot_api_client.api.model_api import ModelApi
from jaqpot_api_client.api.model_download_api import ModelDownloadApi

from .offline_model_data import OfflineModelData


class JaqpotModelDownloader:
    def __init__(self, jaqpot) -> None:
        self.jaqpot = jaqpot
        self._cached_models: Dict[int, OfflineModelData] = {}

    def download_onnx_model(
        self, model_id: int, cache: bool = True
    ) -> OfflineModelData:
        """
        Download a model from Jaqpot platform for offline use.

        Args:
            model_id (int): The ID of the model to download
            cache (bool): Whether to cache the offline model

        Returns:
            OfflineModelData instance containing model metadata and ONNX bytes
        """
        if cache and model_id in self._cached_models:
            return self._cached_models[model_id]

        model_api = ModelApi(self.jaqpot.http_client)
        model = model_api.get_model_by_id(id=model_id)

        # Download ONNX model bytes
        onnx_bytes = self._download_model_bytes(model)

        # Download preprocessor if available
        preprocessor = None
        if hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
            preprocessor = self._download_preprocessor_bytes(model)

        # Create OfflineModelData instance
        offline_model_data = OfflineModelData(
            model_id=model_id,
            model_metadata=model,
            onnx_bytes=onnx_bytes,
            preprocessor=preprocessor,
        )

        if cache:
            self._cached_models[model_id] = offline_model_data

        return offline_model_data

    def _download_model_bytes(self, model: Model) -> bytes:
        """
        Download ONNX model bytes from S3 presigned URL.
        """

        # Try to get presigned download URLs for S3 models using official API client
        try:
            import requests

            model_download_api = ModelDownloadApi(self.jaqpot.http_client)
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

        raise ValueError(f"Failed to download model {model.id} from S3.")

    def _download_preprocessor_bytes(self, model: Model) -> Optional[bytes]:
        """
        Download and deserialize preprocessor from S3 presigned URL.
        """
        # Try to get presigned download URLs for S3 preprocessors using official API client
        try:
            import requests

            model_download_api = ModelDownloadApi(self.jaqpot.http_client)
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

        # No database fallback - only S3 downloads supported
        return None

    def clear_cache(self) -> None:
        """
        Clear cached models.
        """
        self._cached_models.clear()

    def list_cached_models(self) -> List[int]:
        """
        Get list of cached model IDs.
        """
        return list(self._cached_models.keys())
