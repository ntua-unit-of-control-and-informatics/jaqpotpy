# OrganizationSummary


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | 
**name** | **str** |  | 

## Example

```python
from jaqpotpy.api.openapi.models.organization_summary import OrganizationSummary

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationSummary from a JSON string
organization_summary_instance = OrganizationSummary.from_json(json)
# print the JSON string representation of the object
print(OrganizationSummary.to_json())

# convert the object into a dict
organization_summary_dict = organization_summary_instance.to_dict()
# create an instance of OrganizationSummary from a dict
organization_summary_from_dict = OrganizationSummary.from_dict(organization_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


