# openapi_client.OrganizationInvitationApi

All URIs are relative to *https://api.jaqpot.org*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_invitations**](OrganizationInvitationApi.md#create_invitations) | **POST** /v1/organizations/{orgName}/invitations | Create new invitations for an organization
[**get_all_invitations**](OrganizationInvitationApi.md#get_all_invitations) | **GET** /v1/organizations/{orgName}/invitations | Get all invitations for an organization
[**get_invitation**](OrganizationInvitationApi.md#get_invitation) | **GET** /v1/organizations/{name}/invitations/{uuid} | Get the status of an invitation
[**resend_invitation**](OrganizationInvitationApi.md#resend_invitation) | **POST** /v1/organizations/{orgId}/invitations/{id}/resend | Resend an invitation email
[**update_invitation**](OrganizationInvitationApi.md#update_invitation) | **PUT** /v1/organizations/{name}/invitations/{uuid} | Update the status of an invitation


# **create_invitations**
> create_invitations(org_name, create_invitations_request)

Create new invitations for an organization

This endpoint allows an organization admin to create new invitations for users.

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.create_invitations_request import CreateInvitationsRequest
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
    api_instance = openapi_client.OrganizationInvitationApi(api_client)
    org_name = 'org_name_example' # str | Name of the organization
    create_invitations_request = openapi_client.CreateInvitationsRequest() # CreateInvitationsRequest | Invitation request payload

    try:
        # Create new invitations for an organization
        api_instance.create_invitations(org_name, create_invitations_request)
    except Exception as e:
        print("Exception when calling OrganizationInvitationApi->create_invitations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_name** | **str**| Name of the organization | 
 **create_invitations_request** | [**CreateInvitationsRequest**](CreateInvitationsRequest.md)| Invitation request payload | 

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
**201** | Invitations created successfully |  -  |
**400** | Bad request, invalid input |  -  |
**401** | Unauthorized, only admins can create invitations |  -  |
**429** | Too many requests, rate limit exceeded |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_all_invitations**
> List[OrganizationInvitation] get_all_invitations(org_name)

Get all invitations for an organization

This endpoint allows an organization admin to get all invitations for their organization.

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.organization_invitation import OrganizationInvitation
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
    api_instance = openapi_client.OrganizationInvitationApi(api_client)
    org_name = 'org_name_example' # str | Name of the organization

    try:
        # Get all invitations for an organization
        api_response = api_instance.get_all_invitations(org_name)
        print("The response of OrganizationInvitationApi->get_all_invitations:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationInvitationApi->get_all_invitations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_name** | **str**| Name of the organization | 

### Return type

[**List[OrganizationInvitation]**](OrganizationInvitation.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Invitations retrieved successfully |  -  |
**400** | Bad request, invalid input |  -  |
**401** | Unauthorized, only admins can access this endpoint |  -  |
**404** | Organization not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_invitation**
> OrganizationInvitation get_invitation(name, uuid)

Get the status of an invitation

This endpoint allows a user to check the status of an invitation.

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.organization_invitation import OrganizationInvitation
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
    api_instance = openapi_client.OrganizationInvitationApi(api_client)
    name = 'name_example' # str | Name of the organization
    uuid = 'uuid_example' # str | UUID of the invitation

    try:
        # Get the status of an invitation
        api_response = api_instance.get_invitation(name, uuid)
        print("The response of OrganizationInvitationApi->get_invitation:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationInvitationApi->get_invitation: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| Name of the organization | 
 **uuid** | **str**| UUID of the invitation | 

### Return type

[**OrganizationInvitation**](OrganizationInvitation.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Invitation status retrieved successfully |  -  |
**400** | Bad request, invalid input |  -  |
**404** | Invitation not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **resend_invitation**
> resend_invitation(org_id, id)

Resend an invitation email

This endpoint allows an organization admin to resend an invitation email if it has not expired. Only organization admins can access this endpoint.

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
    api_instance = openapi_client.OrganizationInvitationApi(api_client)
    org_id = 56 # int | ID of the organization
    id = 'id_example' # str | ID of the invitation

    try:
        # Resend an invitation email
        api_instance.resend_invitation(org_id, id)
    except Exception as e:
        print("Exception when calling OrganizationInvitationApi->resend_invitation: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **int**| ID of the organization | 
 **id** | **str**| ID of the invitation | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Invitation resent successfully |  -  |
**400** | Bad request, invalid input |  -  |
**401** | Unauthorized, only organization admins can access this endpoint |  -  |
**404** | Organization or invitation not found |  -  |
**410** | Gone, the invitation has expired |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_invitation**
> OrganizationInvitation update_invitation(name, uuid, organization_invitation)

Update the status of an invitation

This endpoint allows a user to update the status of an invitation.

### Example

* Bearer (JWT) Authentication (bearerAuth):

```python
import openapi_client
from openapi_client.models.organization_invitation import OrganizationInvitation
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
    api_instance = openapi_client.OrganizationInvitationApi(api_client)
    name = 'name_example' # str | Name of the organization
    uuid = 'uuid_example' # str | UUID of the invitation
    organization_invitation = openapi_client.OrganizationInvitation() # OrganizationInvitation | Invitation status update payload

    try:
        # Update the status of an invitation
        api_response = api_instance.update_invitation(name, uuid, organization_invitation)
        print("The response of OrganizationInvitationApi->update_invitation:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationInvitationApi->update_invitation: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| Name of the organization | 
 **uuid** | **str**| UUID of the invitation | 
 **organization_invitation** | [**OrganizationInvitation**](OrganizationInvitation.md)| Invitation status update payload | 

### Return type

[**OrganizationInvitation**](OrganizationInvitation.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Invitation status updated successfully |  -  |
**400** | Bad request, invalid input |  -  |
**404** | Invitation not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

