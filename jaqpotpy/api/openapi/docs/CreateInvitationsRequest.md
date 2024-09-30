# CreateInvitationsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**emails** | **List[str]** | List of email addresses to invite | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.create_invitations_request import CreateInvitationsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateInvitationsRequest from a JSON string
create_invitations_request_instance = CreateInvitationsRequest.from_json(json)
# print the JSON string representation of the object
print(CreateInvitationsRequest.to_json())

# convert the object into a dict
create_invitations_request_dict = create_invitations_request_instance.to_dict()
# create an instance of CreateInvitationsRequest from a dict
create_invitations_request_from_dict = CreateInvitationsRequest.from_dict(create_invitations_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


