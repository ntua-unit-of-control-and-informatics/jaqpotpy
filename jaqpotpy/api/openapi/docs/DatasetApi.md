# jaqpotpy.api.openapi.DatasetApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_dataset_by_id**](DatasetApi.md#get_dataset_by_id) | **GET** /v1/datasets/{id} | Get a Dataset
[**get_datasets**](DatasetApi.md#get_datasets) | **GET** /v1/user/datasets | Get Datasets by User ID


# **get_dataset_by_id**
> Dataset get_dataset_by_id(id)

Get a Dataset

Retrieve a single dataset by its ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.dataset import Dataset
from jaqpotpy.api.openapi.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = jaqpotpy.api.openapi.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = jaqpotpy.api.openapi.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with jaqpotpy.api.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = jaqpotpy.api.openapi.DatasetApi(api_client)
    id = 0 # int | The ID of the dataset to retrieve

    try:
        # Get a Dataset
        api_response = api_instance.get_dataset_by_id(id)
        print("The response of DatasetApi->get_dataset_by_id:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetApi->get_dataset_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the dataset to retrieve | 

### Return type

[**Dataset**](Dataset.md)

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

# **get_datasets**
> GetDatasets200Response get_datasets(page=page, size=size, sort=sort)

Get Datasets by User ID

Retrieve all datasets associated with a specific user ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.get_datasets200_response import GetDatasets200Response
from jaqpotpy.api.openapi.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.jaqpot.org
# See configuration.py for a list of all supported configuration parameters.
configuration = jaqpotpy.api.openapi.Configuration(
    host = "https://api.jaqpot.org"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): bearerAuth
configuration = jaqpotpy.api.openapi.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with jaqpotpy.api.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = jaqpotpy.api.openapi.DatasetApi(api_client)
    page = 0 # int |  (optional) (default to 0)
    size = 10 # int |  (optional) (default to 10)
    sort = ['[\"field1|asc\",\"field2|desc\"]'] # List[str] |  (optional)

    try:
        # Get Datasets by User ID
        api_response = api_instance.get_datasets(page=page, size=size, sort=sort)
        print("The response of DatasetApi->get_datasets:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetApi->get_datasets: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **page** | **int**|  | [optional] [default to 0]
 **size** | **int**|  | [optional] [default to 10]
 **sort** | [**List[str]**](str.md)|  | [optional] 

### Return type

[**GetDatasets200Response**](GetDatasets200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**404** | User or datasets not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

