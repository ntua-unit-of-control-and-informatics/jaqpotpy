# jaqpotpy.api.openapi.LeadApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_lead**](LeadApi.md#create_lead) | **POST** /v1/leads | Create a Lead
[**delete_lead_by_id**](LeadApi.md#delete_lead_by_id) | **DELETE** /v1/leads/{id} | Delete a Lead by ID
[**get_all_leads**](LeadApi.md#get_all_leads) | **GET** /v1/leads | Get All Leads
[**get_lead_by_id**](LeadApi.md#get_lead_by_id) | **GET** /v1/leads/{id} | Get a Lead by ID
[**update_lead_by_id**](LeadApi.md#update_lead_by_id) | **PUT** /v1/leads/{id} | Update a Lead by ID


# **create_lead**
> create_lead()

Create a Lead

Create a new lead

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
    api_instance = jaqpotpy.api.openapi.LeadApi(api_client)

    try:
        # Create a Lead
        api_instance.create_lead()
    except Exception as e:
        print("Exception when calling LeadApi->create_lead: %s\n" % e)
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
**201** | Lead created successfully |  -  |
**400** | Invalid request data |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_lead_by_id**
> delete_lead_by_id(id)

Delete a Lead by ID

Delete a single lead by its ID

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
    api_instance = jaqpotpy.api.openapi.LeadApi(api_client)
    id = 0 # int | The ID of the lead to delete

    try:
        # Delete a Lead by ID
        api_instance.delete_lead_by_id(id)
    except Exception as e:
        print("Exception when calling LeadApi->delete_lead_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the lead to delete | 

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
**204** | Lead deleted successfully |  -  |
**404** | Lead not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_all_leads**
> List[Lead] get_all_leads()

Get All Leads

Retrieve all leads

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.lead import Lead
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
    api_instance = jaqpotpy.api.openapi.LeadApi(api_client)

    try:
        # Get All Leads
        api_response = api_instance.get_all_leads()
        print("The response of LeadApi->get_all_leads:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling LeadApi->get_all_leads: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**List[Lead]**](Lead.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_lead_by_id**
> Lead get_lead_by_id(id)

Get a Lead by ID

Retrieve a single lead by its ID

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.lead import Lead
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
    api_instance = jaqpotpy.api.openapi.LeadApi(api_client)
    id = 0 # int | The ID of the lead to retrieve

    try:
        # Get a Lead by ID
        api_response = api_instance.get_lead_by_id(id)
        print("The response of LeadApi->get_lead_by_id:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling LeadApi->get_lead_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the lead to retrieve | 

### Return type

[**Lead**](Lead.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**404** | Lead not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_lead_by_id**
> update_lead_by_id(id, lead)

Update a Lead by ID

Update the details of an existing lead

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.lead import Lead
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
    api_instance = jaqpotpy.api.openapi.LeadApi(api_client)
    id = 0 # int | The ID of the lead to update
    lead = jaqpotpy.api.openapi.Lead() # Lead | 

    try:
        # Update a Lead by ID
        api_instance.update_lead_by_id(id, lead)
    except Exception as e:
        print("Exception when calling LeadApi->update_lead_by_id: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| The ID of the lead to update | 
 **lead** | [**Lead**](Lead.md)|  | 

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
**200** | Lead updated successfully |  -  |
**400** | Invalid request data |  -  |
**404** | Lead not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

