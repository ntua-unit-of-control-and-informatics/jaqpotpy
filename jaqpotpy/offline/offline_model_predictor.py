from typing import Union, Dict, Any, List, Optional, TYPE_CHECKING

import numpy as np
from jaqpot_api_client.models.prediction_response import PredictionResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_model import PredictionModel
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.dataset_type import DatasetType

from jaqpotpy.inference.service import get_prediction_service
from .offline_model_data import OfflineModelData

if TYPE_CHECKING:
    from jaqpotpy.jaqpot import Jaqpot
    from .model_downloader import JaqpotModelDownloader


class OfflineModelPredictor:
    """
    Handles predictions using offline models.

    This class is responsible for making predictions using offline models from
    the Jaqpot platform, ensuring identical results to production inference.
    """

    def __init__(self, jaqpot_client: "Jaqpot") -> None:
        self.jaqpot_client = jaqpot_client

    def predict(
        self,
        model_data: Union[str, OfflineModelData],
        input: Union[np.ndarray, List, Dict],
        model_downloader: Optional["JaqpotModelDownloader"] = None,
    ) -> PredictionResponse:
        """
        Make predictions using an offline model.

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

        # Convert input data to dataset format
        dataset = self._create_dataset(input)

        # Get the unified prediction service
        prediction_service = get_prediction_service()

        # Execute prediction using raw model data
        return prediction_service.predict(
            model_data=model_data, dataset=dataset, model_type=model_data.model_type
        )

    def _create_dataset(self, data: Union[np.ndarray, List, Dict]) -> Dataset:
        """
        Convert input data to Dataset format.

        Args:
            data: Input data (numpy array, list, or dict)

        Returns:
            Dataset: Dataset object for shared inference service
        """
        # Convert data to the expected format for the dataset with jaqpotRowId
        processed_input_data = []

        if isinstance(data, dict):
            # Single dict input - add jaqpotRowId
            row_data = data.copy()
            row_data["jaqpotRowId"] = "0"
            processed_input_data = [row_data]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # List of dicts - add jaqpotRowId to each
                for i, row in enumerate(data):
                    row_data = row.copy()
                    row_data["jaqpotRowId"] = str(i)
                    processed_input_data.append(row_data)
            elif len(data) > 0 and isinstance(data[0], list):
                # List of lists - convert to dicts with jaqpotRowId
                for i, row in enumerate(data):
                    row_data = {f"feature_{j}": val for j, val in enumerate(row)}
                    row_data["jaqpotRowId"] = str(i)
                    processed_input_data.append(row_data)
            else:
                # Single list - convert to dict with jaqpotRowId
                row_data = {f"feature_{i}": val for i, val in enumerate(data)}
                row_data["jaqpotRowId"] = "0"
                processed_input_data = [row_data]
        elif isinstance(data, np.ndarray):
            # Convert numpy array to dicts with jaqpotRowId
            if data.ndim == 1:
                # Check if array contains dicts (edge case: np.array([{...}]))
                if len(data) > 0 and isinstance(data[0], dict):
                    # Single dict wrapped in numpy array
                    row_data = data[0].copy()
                    row_data["jaqpotRowId"] = "0"
                    processed_input_data = [row_data]
                else:
                    # Single row of numeric values
                    row_data = {
                        f"feature_{i}": float(val) for i, val in enumerate(data)
                    }
                    row_data["jaqpotRowId"] = "0"
                    processed_input_data = [row_data]
            else:
                # Multiple rows
                for i, row in enumerate(data):
                    if isinstance(row, dict):
                        # Row is already a dict
                        row_data = row.copy()
                        row_data["jaqpotRowId"] = str(i)
                        processed_input_data.append(row_data)
                    else:
                        # Row is numeric values
                        row_data = {
                            f"feature_{j}": float(val) for j, val in enumerate(row)
                        }
                        row_data["jaqpotRowId"] = str(i)
                        processed_input_data.append(row_data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        input_data = processed_input_data

        # Create dataset object with all required fields
        return Dataset(type=DatasetType.PREDICTION, entryType="ARRAY", input=input_data)
