import polling2
import os

from jaqpotpy.api.jaqpot_api_client_builder import JaqpotApiHttpClientBuilder
from jaqpotpy.api.model_to_b64encoding import file_to_b64encoding
from jaqpotpy.api.openapi import (
    ModelApi,
    DatasetApi,
    Dataset,
    DatasetType,
    DatasetCSV,
    Configuration,
)
from jaqpotpy.exceptions.exceptions import (
    JaqpotApiException,
    JaqpotPredictionTimeoutException,
    JaqpotPredictionFailureException,
)
from jaqpotpy.helpers.logging import init_logger
from jaqpotpy.models import Model
from jaqpotpy.utils.url_utils import add_subdomain

JAQPOT_API_KEY = os.getenv("JAQPOT_API_KEY")
JAQPOT_API_SECRET = os.getenv("JAQPOT_API_SECRET")
QSARTOOLBOX_CALCULATOR_MODEL_ID = 6
QSARTOOLBOX_MODEL_MODEL_ID = 1837
QSAR_PROFILER_MODEL_ID = 1842


class JaqpotApiClient:
    def __init__(
        self,
        base_url=None,
        api_url=None,
        create_logs=False,
    ):
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.log = init_logger(
            __name__, testing_mode=False, output_log_file=create_logs
        )
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "https://jaqpot.org"
        self.api_url = api_url or add_subdomain(self.base_url, "api")
        self.http_client = (
            JaqpotApiHttpClientBuilder(host=self.api_url)
            .build_with_api_keys(JAQPOT_API_KEY, JAQPOT_API_SECRET)
            .build()
        )

    def get_model_by_id(self, model_id) -> Model:
        """Get model from Jaqpot.

        Parameters
        ----------
        model_id : model_id is the id of the model on Jaqpot

        """
        model_api = ModelApi(self.http_client)
        response = model_api.get_model_by_id_with_http_info(id=model_id)
        if response.status_code < 300:
            return response.data.to_dict()
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def get_model_summary(self, model_id):
        """Get model summary from Jaqpot.

        Parameters
        ----------
        model_id : model_id is the id of the model on Jaqpot

        """
        response = self.get_model_by_id(model_id)
        if response.status_code < 300:
            model = response.data
            model_summary = {
                "name": model.name,
                "modelId": model.id,
                "description": model.description,
                "type": model.type,
                "independentFeatures": [
                    feature.name for feature in model.independent_features
                ],
                "dependentFeatures": [
                    feature.name for feature in model.dependent_features
                ],
            }
            return model_summary
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def get_shared_models(self, page=None, size=None, sort=None, organization_id=None):
        """Get shared models from Jaqpot.

        Parameters
        ----------
        page : page number
        size : number of models per page
        sort : sort models by
        organization_id : organization id

        """
        model_api = ModelApi(self.http_client)
        response = model_api.get_shared_models_with_http_info(
            page=page, size=size, sort=sort, organization_id=organization_id
        )
        if response.status_code < 300:
            return response
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def get_dataset_by_id(self, dataset_id) -> Dataset:
        """Get dataset from Jaqpot.

        Parameters
        ----------
        dataset_id : dataset_id is the id of the dataset on Jaqpot

        """
        dataset_api = DatasetApi(self.http_client)
        response = dataset_api.get_dataset_by_id_with_http_info(id=dataset_id)
        if response.status_code < 300:
            return response.data
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def predict_sync(self, model_id, dataset):
        """Predict with model on Jaqpot.

        Parameters
        ----------
        model_id : model_id is the id of the model on Jaqpot
        dataset : dataset to predict

        """
        dataset = Dataset(
            type=DatasetType.PREDICTION,
            entry_type="ARRAY",
            input=dataset,
        )

        model_api = ModelApi(self.http_client)
        response = model_api.predict_with_model_with_http_info(
            model_id=model_id, dataset=dataset
        )
        if response.status_code < 300:
            dataset = self._get_dataset_with_polling(response)
            if dataset.status == "SUCCESS":
                return dataset.result
            elif dataset.status == "FAILURE":
                raise JaqpotPredictionFailureException(dataset.failure_reason)
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def predict_async(self, model_id, dataset):
        """
        Asynchronously predicts using a specified model and dataset.
        This method sends a prediction request to the server using the provided model ID and dataset.
        It constructs a Dataset object with the type set to PREDICTION and entry_type set to "ARRAY".
        The method then uses the ModelApi to send the prediction request.
        Args:
            model_id (str): The ID of the model to use for prediction.
            dataset (list or dict): The input data for prediction.
        Returns:
            int: The ID of the dataset containing the prediction results.
        Raises:
            JaqpotApiException: If the prediction request fails, an exception is raised with the error message
                    and status code from the response.
        """

        dataset = Dataset(
            type=DatasetType.PREDICTION,
            entry_type="ARRAY",
            input=dataset,
        )

        model_api = ModelApi(self.http_client)
        response = model_api.predict_with_model_with_http_info(
            model_id=model_id, dataset=dataset
        )
        if response.status_code < 300:
            dataset_location = response.headers["Location"]
            dataset_id = int(dataset_location.split("/")[-1])
            return dataset_id
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def predict_with_csv_sync(self, model_id, csv_path):
        """Predict with model on Jaqpot.

        Parameters
        ----------
        :param model_id:
        :param csv_path:

        """

        b64_dataset_csv = file_to_b64encoding(csv_path)
        dataset_csv = DatasetCSV(
            type=DatasetType.PREDICTION, input_file=b64_dataset_csv
        )
        model_api = ModelApi(self.http_client)
        response = model_api.predict_with_model_csv_with_http_info(
            model_id=model_id, dataset_csv=dataset_csv
        )
        if response.status_code < 300:
            dataset = self._get_dataset_with_polling(response)
            if dataset.status == "SUCCESS":
                return dataset.result
            elif dataset.status == "FAILURE":
                raise JaqpotPredictionFailureException(message=dataset.failure_reason)
        raise JaqpotApiException(
            message=response.data.to_dict().message,
            status_code=response.status_code.value,
        )

    def _get_dataset_with_polling(self, response):
        """
        Retrieves a dataset by polling until the dataset is ready or a timeout occurs.
        This method extracts the dataset location from the response headers, retrieves the dataset ID,
        and then polls the server to check the status of the dataset. If the status is either "SUCCESS"
        or "FAILURE", the polling stops. If the polling times out, a JaqpotPredictionTimeoutException is raised.
        Args:
            response (requests.Response): The HTTP response object containing the dataset location in the headers.
        Returns:
            dict: The dataset retrieved from the server.
        Raises:
            JaqpotPredictionTimeoutException: If polling times out while waiting for the dataset to be ready.
        """

        dataset_location = response.headers["Location"]
        dataset_id = int(dataset_location.split("/")[-1])
        try:
            polling2.poll(
                lambda: self.get_dataset_by_id(dataset_id).status
                in ["SUCCESS", "FAILURE"],
                step=3,
                timeout=60,
            )
        except polling2.TimeoutException:
            raise JaqpotPredictionTimeoutException(
                message="Polling timed out while waiting for prediction result."
            )
        dataset = self.get_dataset_by_id(dataset_id)
        return dataset

    def qsartoolbox_calculator_predict_sync(self, smiles, calculator_guid):
        """
        Synchronously predicts using the QSAR Toolbox calculator.
        Args:
            smiles (str): The SMILES string representing the chemical structure.
            calculator_guid (str): The unique identifier for the QSAR Toolbox calculator.
        Returns:
            dict: The prediction result from the QSAR Toolbox calculator.
        """

        dataset = [{"smiles": smiles, "calculatorGuid": calculator_guid}]
        prediction = self.predict_sync(QSARTOOLBOX_CALCULATOR_MODEL_ID, dataset)
        return prediction

    def qsartoolbox_qsar_model_predict_sync(self, smiles, qsar_guid):
        """
        Synchronously predicts QSAR model results using the QSAR Toolbox.
        Args:
            smiles (str): The SMILES string representing the chemical structure.
            qsar_guid (str): The unique identifier for the QSAR model.
        Returns:
            dict: The prediction results from the QSAR model.
        """

        dataset = [{"smiles": smiles, "qsarGuid": qsar_guid}]
        prediction = self.predict_sync(QSARTOOLBOX_MODEL_MODEL_ID, dataset)
        return prediction

    def qsartoolbox_profiler_predict_sync(self, smiles, profiler_guid):
        """
        Predicts the QSAR toolbox profiler synchronously.
        Parameters:
        smiles (str): The SMILES string representing the chemical structure.
        profiler_guid (str): The unique identifier for the profiler.
        Returns:
        dict: The prediction result from the QSAR profiler model.
        """

        dataset = [{"smiles": smiles, "profilerGuid": profiler_guid}]
        prediction = self.predict_sync(QSAR_PROFILER_MODEL_ID, dataset)
        return prediction
