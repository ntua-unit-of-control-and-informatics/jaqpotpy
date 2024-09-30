# OrganizationUser


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**user_id** | **str** |  | 
**username** | **str** |  | 
**email** | **str** |  | 
**association_type** | [**OrganizationUserAssociationType**](OrganizationUserAssociationType.md) |  | 

## Example

```python
from openapi_client.models.organization_user import OrganizationUser

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationUser from a JSON string
organization_user_instance = OrganizationUser.from_json(json)
# print the JSON string representation of the object
print(OrganizationUser.to_json())

# convert the object into a dict
organization_user_dict = organization_user_instance.to_dict()
# create an instance of OrganizationUser from a dict
organization_user_from_dict = OrganizationUser.from_dict(organization_user_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


