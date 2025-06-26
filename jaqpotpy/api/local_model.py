import base64
import io
import pickle
import requests
from typing import Union, Dict, Any, List, Optional

import numpy as np
import onnxruntime as rt
from jaqpot_api_client.api.model_api import ModelApi
from jaqpot_api_client.models.prediction_response import PredictionResponse


class JaqpotLocalModel:
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
        Download ONNX model bytes from base64 encoding or S3 URL.
        """
        if hasattr(model, "actual_url") and model.actual_url:
            # Large model stored in S3
            response = requests.get(model.actual_url)
            response.raise_for_status()
            return response.content
        elif hasattr(model, "raw_model") and model.raw_model:
            # Small model stored as base64
            try:
                return base64.b64decode(model.raw_model)
            except Exception as e:
                # If it's already bytes, return as is
                if isinstance(model.raw_model, bytes):
                    return model.raw_model
                raise ValueError(f"Could not decode model bytes: {e}")
        else:
            raise ValueError("Model does not contain ONNX data")

    def _download_preprocessor_bytes(self, model) -> Optional[Any]:
        """
        Download and deserialize preprocessor from base64 encoding or S3 URL.
        """
        try:
            if (
                hasattr(model, "actual_preprocessor_url")
                and model.actual_preprocessor_url
            ):
                # Large preprocessor stored in S3
                response = requests.get(model.actual_preprocessor_url)
                response.raise_for_status()
                return pickle.loads(response.content)
            elif hasattr(model, "raw_preprocessor") and model.raw_preprocessor:
                # Small preprocessor stored as base64
                preprocessor_bytes = base64.b64decode(model.raw_preprocessor)
                return pickle.loads(preprocessor_bytes)
        except Exception as e:
            print(f"Warning: Could not load preprocessor: {e}")
            return None
        return None

    def predict_local(
        self,
        model_data: Union[str, Dict[str, Any]],
        data: Union[np.ndarray, List, Dict],
    ) -> PredictionResponse:
        """
        Make predictions using a locally downloaded model.

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

        # Preprocess data if preprocessor is available
        processed_data = self._preprocess_data(model_data, data)

        # Run ONNX inference
        predictions = self._run_onnx_inference(model_data, processed_data)

        # Format response
        return PredictionResponse(predictions=predictions)

    def _preprocess_data(
        self, model_data: Dict[str, Any], data: Union[np.ndarray, List, Dict]
    ) -> np.ndarray:
        """
        Preprocess input data using the model's preprocessor if available.
        """
        # Convert input data to numpy array if needed
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, dict):
            # Assume dict has feature names as keys
            data = np.array([list(data.values())])

        # Ensure 2D array for sklearn compatibility
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Apply preprocessor if available
        preprocessor = model_data.get("preprocessor")
        if preprocessor is not None:
            try:
                data = preprocessor.transform(data)
            except Exception as e:
                print(f"Warning: Preprocessing failed, using raw data: {e}")

        return data

    def _run_onnx_inference(self, model_data: Dict[str, Any], data: np.ndarray) -> List:
        """
        Run ONNX inference on preprocessed data.
        """
        onnx_bytes = model_data["onnx_bytes"]

        # Create ONNX runtime session
        sess = rt.InferenceSession(onnx_bytes)

        # Get input/output info
        input_name = sess.get_inputs()[0].name
        input_dtype = sess.get_inputs()[0].type

        # Convert data to appropriate dtype
        if "float" in input_dtype:
            data = data.astype(np.float32)
        elif "int" in input_dtype:
            data = data.astype(np.int32)

        # Run inference
        try:
            outputs = sess.run(None, {input_name: data})
            predictions = outputs[0]

            # Convert to list for JSON serialization
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

            return predictions

        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")

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
