# JaqpotApiClient Documentation

## Class: JaqpotApiClient

Client for interacting with the Jaqpot API.

This client provides methods to interact with various endpoints of the Jaqpot API, including retrieving models and datasets, and making synchronous and asynchronous predictions.

### Attributes

- **base_url** : str
  - The base URL for the Jaqpot API.
- **api_url** : str
  - The API URL for the Jaqpot API.
- **http_client** : object
  - The HTTP client used to make requests to the Jaqpot API.
- **log** : object
  - The logger object for logging messages.

### Methods

#### `__init__(self, base_url=None, api_url=None, create_logs=False)`

Initialize the JaqpotApiClient.

**Parameters:**

- **base_url** : str, optional
  - The base URL for the Jaqpot API. Defaults to "https://jaqpot.org".
- **api_url** : str, optional
  - The API URL for the Jaqpot API. If not provided, it is constructed using the base URL.
- **create_logs** : bool, optional
  - Whether to create logs. Defaults to False.

#### `get_model_by_id(self, model_id) -> Model`

Get model from Jaqpot.

**Parameters:**

- **model_id** : int
  - The ID of the model on Jaqpot.

**Returns:**

- **Model**
  - The model object retrieved from Jaqpot.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.

#### `get_model_summary(self, model_id)`

Get model summary from Jaqpot.

**Parameters:**

- **model_id** : int
  - The ID of the model on Jaqpot.

**Returns:**

- **dict**
  - A dictionary containing the model summary.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.

#### `get_shared_models(self, page=None, size=None, sort=None, organization_id=None)`

Get shared models from Jaqpot.

**Parameters:**

- **page** : int, optional
  - Page number for pagination.
- **size** : int, optional
  - Number of models per page.
- **sort** : str, optional
  - Sort order for models.
- **organization_id** : int, optional
  - Organization ID to filter models.

**Returns:**

- **list**
  - A list of shared models.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.

#### `get_dataset_by_id(self, dataset_id) -> Dataset`

Get dataset from Jaqpot.

**Parameters:**

- **dataset_id** : int
  - The ID of the dataset on Jaqpot.

**Returns:**

- **Dataset**
  - The dataset object retrieved from Jaqpot.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.

#### `predict_sync(self, model_id, dataset)`

Predict with model on Jaqpot.

**Parameters:**

- **model_id** : int
  - The ID of the model on Jaqpot.
- **dataset** : list or dict
  - The dataset to predict.

**Returns:**

- **dict**
  - The prediction result.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.
- **JaqpotPredictionFailureException**
  - If the prediction fails, an exception is raised with the failure reason.

#### `predict_async(self, model_id, dataset)`

Asynchronously predict with model on Jaqpot.

**Parameters:**

- **model_id** : int
  - The ID of the model on Jaqpot.
- **dataset** : list or dict
  - The dataset to predict.

**Returns:**

- **int**
  - The ID of the dataset containing the prediction results.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.

#### `predict_with_csv_sync(self, model_id, csv_path)`

Predict with model on Jaqpot using a CSV file.

**Parameters:**

- **model_id** : int
  - The ID of the model on Jaqpot.
- **csv_path** : str
  - The path to the CSV file.

**Returns:**

- **dict**
  - The prediction result.

**Raises:**

- **JaqpotApiException**
  - If the request fails, an exception is raised with the error message and status code.
- **JaqpotPredictionFailureException**
  - If the prediction fails, an exception is raised with the failure reason.

#### `_get_dataset_with_polling(self, response)`

Retrieve a dataset by polling until it is ready or a timeout occurs.

**Parameters:**

- **response** : requests.Response
  - The HTTP response object containing the dataset location in the headers.

**Returns:**

- **dict**
  - The dataset retrieved from the server.

**Raises:**

- **JaqpotPredictionTimeoutException**
  - If polling times out while waiting for the dataset to be ready.

#### `qsartoolbox_calculator_predict_sync(self, smiles, calculator_guid)`

Synchronously predict using the QSAR Toolbox calculator.

**Parameters:**

- **smiles** : str
  - The SMILES string representing the chemical structure.
- **calculator_guid** : str
  - The unique identifier for the QSAR Toolbox calculator.

**Returns:**

- **dict**
  - The prediction result from the QSAR Toolbox calculator.

#### `qsartoolbox_qsar_model_predict_sync(self, smiles, qsar_guid)`

Synchronously predict QSAR model results using the QSAR Toolbox.

**Parameters:**

- **smiles** : str
  - The SMILES string representing the chemical structure.
- **qsar_guid** : str
  - The unique identifier for the QSAR model.

**Returns:**

- **dict**
  - The prediction results from the QSAR model.

#### `qsartoolbox_profiler_predict_sync(self, smiles, profiler_guid)`

Synchronously predict using the QSAR Toolbox profiler.

**Parameters:**

- **smiles** : str
  - The SMILES string representing the chemical structure.
- **profiler_guid** : str
  - The unique identifier for the profiler.

**Returns:**

- **dict**
  - The prediction result from the QSAR profiler model.