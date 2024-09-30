# Organization


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**name** | **str** |  | 
**creator** | [**User**](User.md) |  | [optional] 
**visibility** | [**OrganizationVisibility**](OrganizationVisibility.md) |  | 
**description** | **str** |  | [optional] 
**organization_members** | [**List[OrganizationUser]**](OrganizationUser.md) |  | [optional] 
**contact_email** | **str** |  | 
**contact_phone** | **str** |  | [optional] 
**website** | **str** |  | [optional] 
**address** | **str** |  | [optional] 
**can_edit** | **bool** | If the current user can edit the organization | [optional] 
**is_member** | **bool** | If the current user is a member of the organization | [optional] 
**created_at** | **str** |  | [optional] 
**updated_at** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.organization import Organization

# TODO update the JSON string below
json = "{}"
# create an instance of Organization from a JSON string
organization_instance = Organization.from_json(json)
# print the JSON string representation of the object
print(Organization.to_json())

# convert the object into a dict
organization_dict = organization_instance.to_dict()
# create an instance of Organization from a dict
organization_from_dict = Organization.from_dict(organization_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


