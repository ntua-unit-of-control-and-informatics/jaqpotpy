# jaqpotpy.api.openapi.AuthApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**validate_jwt**](AuthApi.md#validate_jwt) | **GET** /v1/auth/validate | Validate JWT


# **validate_jwt**
> validate_jwt()

Validate JWT

Validate a JWT token

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
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
    api_instance = jaqpotpy.api.openapi.AuthApi(api_client)

    try:
        # Validate JWT
        api_instance.validate_jwt()
    except Exception as e:
        print("Exception when calling AuthApi->validate_jwt: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

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
**200** | JWT is valid |  -  |
**401** | Unauthorized |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

