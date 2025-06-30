"""
Model loading utilities for ONNX models.

This module contains utilities for loading ONNX models from various sources
including base64 encoded strings, S3 storage, and local files.
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


def load_onnx_model_from_base64(base64_string: str) -> onnx.ModelProto:
    """
    Load an ONNX model from a base64 encoded string.

    Args:
        base64_string (str): Base64 encoded model string

    Returns:
        onnx.ModelProto: Loaded ONNX model

    Raises:
        Exception: If model decoding or loading fails
    """
    try:
        model_bytes = base64.b64decode(base64_string)
        return load_onnx_model_from_bytes(model_bytes)
    except Exception as e:
        raise Exception(f"Failed to load ONNX model from base64: {str(e)}")


def load_onnx_model_from_file(file_obj: Union[BinaryIO, io.BytesIO]) -> onnx.ModelProto:
    """
    Load an ONNX model from a file object.

    Args:
        file_obj: File object containing the ONNX model

    Returns:
        onnx.ModelProto: Loaded ONNX model

    Raises:
        Exception: If model loading fails
    """
    try:
        return onnx.load(file_obj)
    except Exception as e:
        raise Exception(f"Failed to load ONNX model from file: {str(e)}")


def retrieve_onnx_model_from_request(
    request: PredictionRequest, s3_client=None, jaqpot_client=None
) -> onnx.ModelProto:
    """
    Retrieve an ONNX model from a prediction request.

    This function handles both inline models (base64 encoded), S3-stored models,
    and presigned URL downloads for local development.

    Args:
        request (PredictionRequest): The prediction request
        s3_client: Optional S3 client for downloading models from S3 (production mode)
        jaqpot_client: Optional Jaqpot client for presigned URLs (local mode)

    Returns:
        onnx.ModelProto: The loaded ONNX model

    Raises:
        Exception: If model retrieval or loading fails
    """
    if request.model.raw_model is None:
        # Model is stored in S3 - use different strategies based on available clients
        if jaqpot_client is not None:
            # Local mode: Use presigned URLs
            model_bytes = _download_model_via_presigned_url(
                request.model, jaqpot_client
            )
            return load_onnx_model_from_bytes(model_bytes)
        elif s3_client is not None:
            # Production mode: Direct S3 download
            file_obj, error = s3_client.download_file(str(request.model.id))
            if file_obj is None:
                raise Exception(f"Failed to download model from S3: {error}")
            return load_onnx_model_from_file(file_obj)
        else:
            raise Exception(
                "Either S3 client (production) or Jaqpot client (local) required for downloading model from S3"
            )
    else:
        # Model is inline (base64 encoded)
        return load_onnx_model_from_base64(request.model.raw_model)


def _download_model_via_presigned_url(model, jaqpot_client) -> bytes:
    """
    Download model bytes using presigned URLs (local development mode).

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
        # Get presigned download URL using LocalModelService
        auth_header = jaqpot_client.http_client.default_headers.get("Authorization", "")
        headers = {"Authorization": f"Bearer {auth_header.replace('Bearer ', '')}"}
        api_host = jaqpot_client.http_client.configuration.host
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
            raise Exception(f"Model {model.id} not found in S3 storage")
        else:
            response.raise_for_status()

    except Exception as e:
        raise Exception(f"Failed to download model via presigned URL: {str(e)}")


def retrieve_raw_model_from_request(
    request: PredictionRequest, s3_client=None, jaqpot_client=None
) -> bytes:
    """
    Retrieve raw model bytes from a prediction request.

    Args:
        request (PredictionRequest): The prediction request
        s3_client: Optional S3 client for downloading models from S3 (production mode)
        jaqpot_client: Optional Jaqpot client for presigned URLs (local mode)

    Returns:
        bytes: Raw model bytes

    Raises:
        Exception: If model retrieval fails
    """
    if request.model.raw_model is None:
        # Model is stored in S3 - use different strategies based on available clients
        if jaqpot_client is not None:
            # Local mode: Use presigned URLs
            return _download_model_via_presigned_url(request.model, jaqpot_client)
        elif s3_client is not None:
            # Production mode: Direct S3 download
            file_obj, error = s3_client.download_file(str(request.model.id))
            if file_obj is None:
                raise Exception(f"Failed to download model from S3: {error}")

            # Read bytes from file object
            if hasattr(file_obj, "read"):
                return file_obj.read()
            else:
                return file_obj
        else:
            raise Exception(
                "Either S3 client (production) or Jaqpot client (local) required for downloading model from S3"
            )
    else:
        # Model is inline (base64 encoded)
        return base64.b64decode(request.model.raw_model)


def validate_onnx_model(model: onnx.ModelProto) -> bool:
    """
    Validate an ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model to validate

    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        onnx.checker.check_model(model)
        return True
    except Exception:
        return False
