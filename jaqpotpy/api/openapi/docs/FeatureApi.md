# jaqpotpy.api.openapi.FeatureApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**partially_update_model_feature**](FeatureApi.md#partially_update_model_feature) | **PATCH** /v1/models/{modelId}/features/{featureId} | Update a feature for a specific model


# **partially_update_model_feature**
> Feature partially_update_model_feature(model_id, feature_id, partially_update_model_feature_request)

Update a feature for a specific model

Update the name, description, and feature type of an existing feature within a specific model

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.partially_update_model_feature_request import PartiallyUpdateModelFeatureRequest
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
    api_instance = jaqpotpy.api.openapi.FeatureApi(api_client)
    model_id = 56 # int | The ID of the model containing the feature
    feature_id = 56 # int | The ID of the feature to update
    partially_update_model_feature_request = jaqpotpy.api.openapi.PartiallyUpdateModelFeatureRequest() # PartiallyUpdateModelFeatureRequest | 

    try:
        # Update a feature for a specific model
        api_response = api_instance.partially_update_model_feature(model_id, feature_id, partially_update_model_feature_request)
        print("The response of FeatureApi->partially_update_model_feature:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling FeatureApi->partially_update_model_feature: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_id** | **int**| The ID of the model containing the feature | 
 **feature_id** | **int**| The ID of the feature to update | 
 **partially_update_model_feature_request** | [**PartiallyUpdateModelFeatureRequest**](PartiallyUpdateModelFeatureRequest.md)|  | 

### Return type

[**Feature**](Feature.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Feature updated successfully |  -  |
**400** | Invalid input |  -  |
**404** | Model or feature not found |  -  |
**401** | Unauthorized |  -  |
**403** | Forbidden |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

