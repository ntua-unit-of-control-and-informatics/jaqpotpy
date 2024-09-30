# OrganizationInvitation


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | ID of the invitation | [optional] 
**user_id** | **str** | The user id associated with that invitation | [optional] 
**user_email** | **str** | Email address of the invited user | 
**status** | **str** | Status of the invitation | 
**expiration_date** | **str** | Expiration date of the invitation | 

## Example

```python
from jaqpotpy.api.openapi.models.organization_invitation import OrganizationInvitation

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationInvitation from a JSON string
organization_invitation_instance = OrganizationInvitation.from_json(json)
# print the JSON string representation of the object
print(OrganizationInvitation.to_json())

# convert the object into a dict
organization_invitation_dict = organization_invitation_instance.to_dict()
# create an instance of OrganizationInvitation from a dict
organization_invitation_from_dict = OrganizationInvitation.from_dict(organization_invitation_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


