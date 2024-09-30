# PartialUpdateOrganizationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** |  | 
**description** | **str** |  | [optional] 
**contact_email** | **str** |  | 
**visibility** | [**OrganizationVisibility**](OrganizationVisibility.md) |  | 

## Example

```python
from openapi_client.models.partial_update_organization_request import PartialUpdateOrganizationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PartialUpdateOrganizationRequest from a JSON string
partial_update_organization_request_instance = PartialUpdateOrganizationRequest.from_json(json)
# print the JSON string representation of the object
print(PartialUpdateOrganizationRequest.to_json())

# convert the object into a dict
partial_update_organization_request_dict = partial_update_organization_request_instance.to_dict()
# create an instance of PartialUpdateOrganizationRequest from a dict
partial_update_organization_request_from_dict = PartialUpdateOrganizationRequest.from_dict(partial_update_organization_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


