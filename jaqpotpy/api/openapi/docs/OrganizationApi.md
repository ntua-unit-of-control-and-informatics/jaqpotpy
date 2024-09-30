# jaqpotpy.api.openapi.OrganizationApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_organization**](OrganizationApi.md#create_organization) | **POST** /v1/organizations | Create a new organization
[**get_all_organizations_by_user**](OrganizationApi.md#get_all_organizations_by_user) | **GET** /v1/user/organizations | Get all user organizations
[**get_all_organizations_for_user**](OrganizationApi.md#get_all_organizations_for_user) | **GET** /v1/organizations | Get all organizations for a specific user
[**get_organization_by_name**](OrganizationApi.md#get_organization_by_name) | **GET** /v1/organizations/{name} | Get organization by name
[**partial_update_organization**](OrganizationApi.md#partial_update_organization) | **PATCH** /v1/organizations/{id}/partial | Partially update an existing organization


# **create_organization**
> create_organization(organization)

Create a new organization

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.organization import Organization
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
    api_instance = jaqpotpy.api.openapi.OrganizationApi(api_client)
    organization = jaqpotpy.api.openapi.Organization() # Organization | 

    try:
        # Create a new organization
        api_instance.create_organization(organization)
    except Exception as e:
        print("Exception when calling OrganizationApi->create_organization: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **organization** | [**Organization**](Organization.md)|  | 

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
**201** | Organization created successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_all_organizations_by_user**
> List[Organization] get_all_organizations_by_user()

Get all user organizations

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.organization import Organization
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
    api_instance = jaqpotpy.api.openapi.OrganizationApi(api_client)

    try:
        # Get all user organizations
        api_response = api_instance.get_all_organizations_by_user()
        print("The response of OrganizationApi->get_all_organizations_by_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationApi->get_all_organizations_by_user: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**List[Organization]**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_all_organizations_for_user**
> List[Organization] get_all_organizations_for_user()

Get all organizations for a specific user

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.organization import Organization
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
    api_instance = jaqpotpy.api.openapi.OrganizationApi(api_client)

    try:
        # Get all organizations for a specific user
        api_response = api_instance.get_all_organizations_for_user()
        print("The response of OrganizationApi->get_all_organizations_for_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationApi->get_all_organizations_for_user: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**List[Organization]**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_organization_by_name**
> Organization get_organization_by_name(name)

Get organization by name

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.organization import Organization
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
    api_instance = jaqpotpy.api.openapi.OrganizationApi(api_client)
    name = 'name_example' # str | 

    try:
        # Get organization by name
        api_response = api_instance.get_organization_by_name(name)
        print("The response of OrganizationApi->get_organization_by_name:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationApi->get_organization_by_name: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**|  | 

### Return type

[**Organization**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**404** | Organization not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **partial_update_organization**
> Organization partial_update_organization(id, partial_update_organization_request)

Partially update an existing organization

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import jaqpotpy.api.openapi
from jaqpotpy.api.openapi.models.organization import Organization
from jaqpotpy.api.openapi.models.partial_update_organization_request import PartialUpdateOrganizationRequest
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
    api_instance = jaqpotpy.api.openapi.OrganizationApi(api_client)
    id = 56 # int | 
    partial_update_organization_request = jaqpotpy.api.openapi.PartialUpdateOrganizationRequest() # PartialUpdateOrganizationRequest | 

    try:
        # Partially update an existing organization
        api_response = api_instance.partial_update_organization(id, partial_update_organization_request)
        print("The response of OrganizationApi->partial_update_organization:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationApi->partial_update_organization: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**|  | 
 **partial_update_organization_request** | [**PartialUpdateOrganizationRequest**](PartialUpdateOrganizationRequest.md)|  | 

### Return type

[**Organization**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Organization updated successfully |  -  |
**404** | Organization not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

