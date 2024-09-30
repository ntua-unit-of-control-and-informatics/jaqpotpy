# openapi_client.ModelApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_model**](ModelApi.md#create_model) | **POST** /v1/models | Create a new model
[**delete_model_by_id**](ModelApi.md#delete_model_by_id) | **DELETE** /v1/models/{id} | Delete a Model
[**get_legacy_model_by_id**](ModelApi.md#get_legacy_model_by_id) | **GET** /v1/models/legacy/{id} | Get a legacy model
[**get_model_by_id**](ModelApi.md#get_model_by_id) | **GET** /v1/models/{id} | Get a Model
[**get_models**](ModelApi.md#get_models) | **GET** /v1/user/models | Get paginated models
[**get_shared_models**](ModelApi.md#get_shared_models) | **GET** /v1/user/shared-models | Get paginated shared models
[**partially_update_model**](ModelApi.md#partially_update_model) | **PATCH** /v1/models/{id}/partial | Partially update specific fields of a model
[**predict_with_model**](ModelApi.md#predict_with_model) | **POST** /v1/models/{modelId}/predict | Predict with Model
[**predict_with_model_csv**](ModelApi.md#predict_with_model_csv) | **POST** /v1/models/{modelId}/predict/csv | Predict using CSV with Model
[**search_models**](ModelApi.md#search_models) | **GET** /v1/models/search | Search for models


# **create_model**
> create_model(model)

Create a new model

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.model import Model
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    model = openapi_client.Model() # Model | 

    try:
        # Create a new model
        api_instance.create_model(model)
    except Exception as e:
        print("Exception when calling ModelApi->create_model: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model** | [**Model**](Model.md)|  | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Model created successfully |  -  |
**400** | Invalid input |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_model_by_id**
> delete_model_by_id(id)

Delete a Model

Delete a single model by its ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    id = 0 # int | The ID of the model to delete

    try:
        # Delete a Model
        api_instance.delete_model_by_id(id)
    except Exception as e:
        print("Exception when calling ModelApi->delete_model_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the model to delete | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**204** | Model deleted successfully |  -  |
**404** | Model not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_legacy_model_by_id**
> Model get_legacy_model_by_id(id)

Get a legacy model

Retrieve a single model by its ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.model import Model
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    id = 'id_example' # str | The ID of the model to retrieve

    try:
        # Get a legacy model
        api_response = api_instance.get_legacy_model_by_id(id)
        print("The response of ModelApi->get_legacy_model_by_id:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->get_legacy_model_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **str**| The ID of the model to retrieve | 

### Return type

[**Model**](Model.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**404** | Model not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_model_by_id**
> Model get_model_by_id(id)

Get a Model

Retrieve a single model by its ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.model import Model
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    id = 0 # int | The ID of the model to retrieve

    try:
        # Get a Model
        api_response = api_instance.get_model_by_id(id)
        print("The response of ModelApi->get_model_by_id:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->get_model_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the model to retrieve | 

### Return type

[**Model**](Model.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**404** | Model not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_models**
> GetModels200Response get_models(page=page, size=size, sort=sort)

Get paginated models

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.get_models200_response import GetModels200Response
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    page = 0 # int |  (optional) (default to 0)
    size = 10 # int |  (optional) (default to 10)
    sort = ['[\"field1|asc\",\"field2|desc\"]'] # List[str] |  (optional)

    try:
        # Get paginated models
        api_response = api_instance.get_models(page=page, size=size, sort=sort)
        print("The response of ModelApi->get_models:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->get_models: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **page** | **int**|  | [optional] [default to 0]
 **size** | **int**|  | [optional] [default to 10]
 **sort** | [**List[str]**](str.md)|  | [optional] 

### Return type

[**GetModels200Response**](GetModels200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Paginated list of models |  -  |
**400** | Invalid input |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_shared_models**
> GetModels200Response get_shared_models(page=page, size=size, sort=sort, organization_id=organization_id)

Get paginated shared models

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.get_models200_response import GetModels200Response
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    page = 0 # int |  (optional) (default to 0)
    size = 10 # int |  (optional) (default to 10)
    sort = ['[\"field1|asc\",\"field2|desc\"]'] # List[str] |  (optional)
    organization_id = 56 # int |  (optional)

    try:
        # Get paginated shared models
        api_response = api_instance.get_shared_models(page=page, size=size, sort=sort, organization_id=organization_id)
        print("The response of ModelApi->get_shared_models:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->get_shared_models: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **page** | **int**|  | [optional] [default to 0]
 **size** | **int**|  | [optional] [default to 10]
 **sort** | [**List[str]**](str.md)|  | [optional] 
 **organization_id** | **int**|  | [optional] 

### Return type

[**GetModels200Response**](GetModels200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Paginated list of shared models |  -  |
**400** | Invalid input |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **partially_update_model**
> Model partially_update_model(id, partially_update_model_request)

Partially update specific fields of a model

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.model import Model
from openapi_client.models.partially_update_model_request import PartiallyUpdateModelRequest
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    id = 56 # int | 
    partially_update_model_request = openapi_client.PartiallyUpdateModelRequest() # PartiallyUpdateModelRequest | 

    try:
        # Partially update specific fields of a model
        api_response = api_instance.partially_update_model(id, partially_update_model_request)
        print("The response of ModelApi->partially_update_model:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->partially_update_model: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**|  | 
 **partially_update_model_request** | [**PartiallyUpdateModelRequest**](PartiallyUpdateModelRequest.md)|  | 

### Return type

[**Model**](Model.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Model fields updated successfully |  -  |
**404** | Model not found |  -  |
**400** | Invalid input |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **predict_with_model**
> predict_with_model(model_id, dataset)

Predict with Model

Submit a dataset for prediction using a specific model

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.dataset import Dataset
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    model_id = 0 # int | The ID of the model to use for prediction
    dataset = openapi_client.Dataset() # Dataset | 

    try:
        # Predict with Model
        api_instance.predict_with_model(model_id, dataset)
    except Exception as e:
        print("Exception when calling ModelApi->predict_with_model: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **int**| The ID of the model to use for prediction | 
 **dataset** | [**Dataset**](Dataset.md)|  | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Prediction created successfully |  -  |
**400** | Invalid Request |  -  |
**404** | Model not found |  -  |
**500** | Internal Server Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **predict_with_model_csv**
> predict_with_model_csv(model_id, dataset_csv)

Predict using CSV with Model

Submit a dataset for prediction using a specific model

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.dataset_csv import DatasetCSV
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    model_id = 0 # int | The ID of the model to use for prediction
    dataset_csv = openapi_client.DatasetCSV() # DatasetCSV | 

    try:
        # Predict using CSV with Model
        api_instance.predict_with_model_csv(model_id, dataset_csv)
    except Exception as e:
        print("Exception when calling ModelApi->predict_with_model_csv: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **int**| The ID of the model to use for prediction | 
 **dataset_csv** | [**DatasetCSV**](DatasetCSV.md)|  | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Prediction created successfully |  -  |
**400** | Invalid Request |  -  |
**404** | Model not found |  -  |
**500** | Internal Server Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **search_models**
> GetModels200Response search_models(query, page=page, size=size)

Search for models

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.get_models200_response import GetModels200Response
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.ModelApi(api_client)
    query = 'query_example' # str | 
    page = 0 # int |  (optional) (default to 0)
    size = 10 # int |  (optional) (default to 10)

    try:
        # Search for models
        api_response = api_instance.search_models(query, page=page, size=size)
        print("The response of ModelApi->search_models:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelApi->search_models: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **query** | **str**|  | 
 **page** | **int**|  | [optional] [default to 0]
 **size** | **int**|  | [optional] [default to 10]

### Return type

[**GetModels200Response**](GetModels200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Paginated list of models |  -  |
**400** | Invalid input |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

