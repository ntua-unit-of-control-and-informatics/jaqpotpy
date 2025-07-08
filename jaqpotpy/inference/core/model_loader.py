"""
Model loading utilities for ONNX models.

This module contains utilities for loading ONNX models from Jaqpot
"""

import base64
import io
from typing import Optional, Tuple, Union, BinaryIO
import onnx
from jaqpot_api_client import PredictionRequest


def load_onnx_model_from_bytes(model_bytes: bytes) -> onnx.ModelProto:
    """
    Load an ONNX model from raw bytes.

    Args:
        model_bytes (bytes): Raw model bytes

    Returns:
        onnx.ModelProto: Loaded ONNX model

    Raises:
        Exception: If model loading fails
    """
    try:
        return onnx.load_from_string(model_bytes)
    except Exception as e:
        raise Exception(f"Failed to load ONNX model from bytes: {str(e)}")


def _download_model_via_presigned_url(model, jaqpot_client) -> bytes:
    """
    Download model bytes using presigned URLs (offline mode).

    Args:
        model: Model object with id attribute
        jaqpot_client: Jaqpot client for making API calls

    Returns:
        bytes: Raw model bytes

    Raises:
        Exception: If download fails
    """
    import requests

    try:
        # Get presigned download URL
        auth_header = jaqpot_client.http_client.default_headers.get("Authorization", "")
        headers = {"Authorization": f"Bearer {auth_header.replace('Bearer ', '')}"}
        api_host = jaqpot_client.http_client.configuration.host
        response = requests.get(
            f"{api_host}/v1/models/{model.id}/offline/download-url",
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
            raise Exception(f"Model {model.id} not found in S3 storage")
        else:
            response.raise_for_status()

    except Exception as e:
        raise Exception(f"Failed to download model via presigned URL: {str(e)}")
