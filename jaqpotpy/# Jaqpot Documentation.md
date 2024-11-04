# Jaqpot Documentation

## Jaqpot Class

Deploys sklearn models on Jaqpot.

### Parameters

- `base_url` : str, optional
  - The url on which Jaqpot services are deployed.
- `app_url` : str, optional
  - The url for the Jaqpot application.
- `login_url` : str, optional
  - The url for the Jaqpot login.
- `api_url` : str, optional
  - The url for the Jaqpot API.
- `keycloak_realm` : str, optional
  - The Keycloak realm name.
- `keycloak_client_id` : str, optional
  - The Keycloak client ID.
- `create_logs` : bool, optional
  - Whether to create logs.

## Methods

### `login()`

Log in to Jaqpot using Keycloak.

This method opens a browser window for the user to log in via Keycloak, then exchanges the authorization code for an access token.

### `deploy_sklearn_model(model, name, description, visibility)`

Deploy an sklearn model on Jaqpot.

#### Parameters

- `model` : object
  - The sklearn model to be deployed.
- `name` : str
  - The name of the model.
- `description` : str
  - A description of the model.
- `visibility` : str
  - The visibility of the model (e.g., 'public', 'private').

#### Returns

- `None`

### `deploy_torch_model(onnx_model, featurizer, name, description, target_name, visibility, task)`

Deploy a PyTorch model on Jaqpot.

#### Parameters

- `onnx_model` : object
  - The ONNX model to be deployed.
- `featurizer` : object
  - The featurizer used for preprocessing.
- `name` : str
  - The name of the model.
- `description` : str
  - A description of the model.
- `target_name` : str
  - The name of the target feature.
- `visibility` : str
  - The visibility of the model (e.g., 'public', 'private').
- `task` : str
  - The task type (e.g., 'binary_classification', 'regression').

#### Returns

- `None`