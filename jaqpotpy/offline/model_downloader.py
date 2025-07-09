import pickle
from typing import Dict, List, Optional

from jaqpot_api_client import Model, GetModelDownloadUrls200Response
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

        model_download_api = ModelDownloadApi(self.jaqpot.http_client)
        model_download_urls = model_download_api.get_model_download_urls(
            model_id=model_id, expiration_minutes=30
        )
        offline_model_data = self._download_model_files(model_download_urls, model_id)

        if cache:
            self._cached_models[model_id] = offline_model_data

        return offline_model_data

    def _download_model_files(
        self, model_download_urls: GetModelDownloadUrls200Response, model_id: int
    ) -> OfflineModelData:
        """
        Download ONNX model bytes from S3 presigned URL.
        """
        import requests

        # Download ONNX model bytes
        model_bytes = None
        if model_download_urls.model_url:
            try:
                model_response = requests.get(model_download_urls.model_url, timeout=60)
                model_response.raise_for_status()
                model_bytes = model_response.content
            except Exception as e:
                raise ValueError(f"Failed to download model {model_id} from S3: {e}")
        else:
            raise ValueError(f"No model URL provided for model {model_id}")

        # Download preprocessor if available
        preprocessor = None
        if model_download_urls.preprocessor_url:
            try:
                preprocessor_response = requests.get(
                    model_download_urls.preprocessor_url, timeout=60
                )
                preprocessor_response.raise_for_status()
                preprocessor = preprocessor_response.content
            except Exception as e:
                print(
                    f"Warning: Could not download preprocessor for model {model_id}: {e}"
                )

        # Download DOA files if available
        doas = None
        if model_download_urls.doa_urls:
            doas = []
            for doa_url_info in model_download_urls.doa_urls:
                if doa_url_info.download_url:
                    try:
                        doa_response = requests.get(
                            doa_url_info.download_url, timeout=60
                        )
                        doa_response.raise_for_status()
                        doa_bytes = doa_response.content
                        doas.append(doa_bytes)
                    except Exception as e:
                        print(
                            f"Warning: Could not download DOA file for model {model_id}: {e}"
                        )

        # Get model metadata
        model_api = ModelApi(self.jaqpot.http_client)
        model_metadata = model_api.get_model_by_id(model_id)

        return OfflineModelData(
            model_id=model_id,
            model_metadata=model_metadata,
            model_bytes=model_bytes,
            preprocessor=preprocessor,
            doas=doas,
        )

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
