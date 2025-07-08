import base64
from typing import Union, Dict, Any, List

import numpy as np
from jaqpot_api_client.models.prediction_response import PredictionResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_model import PredictionModel
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.dataset_type import DatasetType

from jaqpotpy.inference.service import get_prediction_service
from .offline_model_data import OfflineModelData


class OfflineModelPredictor:
    """
    Handles predictions using downloaded models.

    This class is responsible for making predictions using models that have been
    downloaded from the Jaqpot platform, ensuring identical results to production inference.
    """

    def __init__(self, jaqpot_client):
        self.jaqpot_client = jaqpot_client

    def predict(
        self,
        model_data: Union[str, OfflineModelData],
        input: Union[np.ndarray, List, Dict],
        model_downloader=None,
    ) -> PredictionResponse:
        """
        Make predictions using a downloaded model.

        This method uses the shared inference service to ensure identical results
        to production inference while maintaining the same external API.

        Args:
            model_data: Either model_id (str) or OfflineModelData instance
            input: Input data for prediction (numpy array, list, or dict)
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
                model_data = model_downloader.download_onnx_model(model_id)
            else:
                model_data = model_downloader._cached_models[model_id]

        # Now model_data is guaranteed to be an OfflineModelData instance
        assert isinstance(
            model_data, OfflineModelData
        ), f"Expected OfflineModelData, got {type(model_data)}"

        # Convert downloaded model format to PredictionRequest format for shared service
        request = self._create_prediction_request(model_data, input)

        # Use shared prediction service (downloaded model mode with jaqpot client)
        prediction_service = get_prediction_service(
            local_mode=True, jaqpot_client=self.jaqpot_client
        )

        # Execute prediction using shared logic
        return prediction_service.predict(request)

    def _create_prediction_request(
        self, model_data: OfflineModelData, data
    ) -> PredictionRequest:
        """
        Convert OfflineModelData and input data to PredictionRequest format.

        Args:
            model_data: OfflineModelData instance containing model components
            data: Input data (numpy array, list, or dict)

        Returns:
            PredictionRequest: Request object for shared inference service
        """
        # Convert data to the expected format for the dataset with jaqpotRowId
        processed_input_data = []

        if isinstance(data, dict):
            # Single dict input - add jaqpotRowId
            row_data = data.copy()
            row_data["jaqpotRowId"] = "row_0"
            processed_input_data = [row_data]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # List of dicts - add jaqpotRowId to each
                for i, row in enumerate(data):
                    row_data = row.copy()
                    row_data["jaqpotRowId"] = f"row_{i}"
                    processed_input_data.append(row_data)
            elif len(data) > 0 and isinstance(data[0], list):
                # List of lists - convert to dicts with jaqpotRowId
                for i, row in enumerate(data):
                    row_data = {f"feature_{j}": val for j, val in enumerate(row)}
                    row_data["jaqpotRowId"] = f"row_{i}"
                    processed_input_data.append(row_data)
            else:
                # Single list - convert to dict with jaqpotRowId
                row_data = {f"feature_{i}": val for i, val in enumerate(data)}
                row_data["jaqpotRowId"] = "row_0"
                processed_input_data = [row_data]
        elif isinstance(data, np.ndarray):
            # Convert numpy array to dicts with jaqpotRowId
            if data.ndim == 1:
                # Check if array contains dicts (edge case: np.array([{...}]))
                if len(data) > 0 and isinstance(data[0], dict):
                    # Single dict wrapped in numpy array
                    row_data = data[0].copy()
                    row_data["jaqpotRowId"] = "row_0"
                    processed_input_data = [row_data]
                else:
                    # Single row of numeric values
                    row_data = {
                        f"feature_{i}": float(val) for i, val in enumerate(data)
                    }
                    row_data["jaqpotRowId"] = "row_0"
                    processed_input_data = [row_data]
            else:
                # Multiple rows
                for i, row in enumerate(data):
                    if isinstance(row, dict):
                        # Row is already a dict
                        row_data = row.copy()
                        row_data["jaqpotRowId"] = f"row_{i}"
                        processed_input_data.append(row_data)
                    else:
                        # Row is numeric values
                        row_data = {
                            f"feature_{j}": float(val) for j, val in enumerate(row)
                        }
                        row_data["jaqpotRowId"] = f"row_{i}"
                        processed_input_data.append(row_data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        input_data = processed_input_data

        # Create dataset object with all required fields
        dataset = Dataset(
            type=DatasetType.PREDICTION, entryType="ARRAY", input=input_data
        )

        # Prepare downloaded model data for inference service
        # Use OfflineModelData properties for clean access to base64 encoded data
        raw_model_b64 = model_data.onnx_base64
        raw_preprocessor_b64 = model_data.preprocessor_base64

        # Create prediction model with downloaded and properly encoded data
        # Use OfflineModelData properties for clean access to metadata
        independent_features = model_data.independent_features
        dependent_features = model_data.dependent_features
        task = model_data.task

        # Ensure we have the required fields with proper defaults
        if task is None:
            # Default to a reasonable task if not specified
            from jaqpot_api_client.models.model_task import ModelTask

            task = ModelTask.REGRESSION

        # Ensure optional fields are lists, not None
        selected_features = getattr(
            model_data.model_metadata, "selected_features", None
        )
        if selected_features is None:
            selected_features = []

        featurizers = getattr(model_data.model_metadata, "featurizers", None)
        if featurizers is None:
            featurizers = []

        preprocessors = getattr(model_data.model_metadata, "preprocessors", None)
        if preprocessors is None:
            preprocessors = []

        prediction_model = PredictionModel(
            id=model_data.model_metadata.id,
            type=model_data.model_type,
            independent_features=independent_features,  # Required field
            dependent_features=dependent_features,  # Required field
            task=task,  # Required field
            raw_model=raw_model_b64,  # Base64 encoded ONNX bytes from downloaded model
            raw_preprocessor=raw_preprocessor_b64,  # Base64 encoded pickled preprocessor
            doas=getattr(model_data.model_metadata, "doas", None),
            selected_features=selected_features,  # Ensured to be list, not None
            featurizers=featurizers,  # Ensured to be list, not None
            preprocessors=preprocessors,  # Ensured to be list, not None
        )

        return PredictionRequest(model=prediction_model, dataset=dataset)
