# PartiallyUpdateModelRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** |  | 
**description** | **str** |  | [optional] 
**visibility** | [**ModelVisibility**](ModelVisibility.md) |  | 
**task** | [**ModelTask**](ModelTask.md) |  | 
**tags** | **str** |  | [optional] 
**shared_with_organization_ids** | **List[int]** |  | [optional] 

## Example

```python
from openapi_client.models.partially_update_model_request import PartiallyUpdateModelRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PartiallyUpdateModelRequest from a JSON string
partially_update_model_request_instance = PartiallyUpdateModelRequest.from_json(json)
# print the JSON string representation of the object
print(PartiallyUpdateModelRequest.to_json())

# convert the object into a dict
partially_update_model_request_dict = partially_update_model_request_instance.to_dict()
# create an instance of PartiallyUpdateModelRequest from a dict
partially_update_model_request_from_dict = PartiallyUpdateModelRequest.from_dict(partially_update_model_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


