# PartiallyUpdateModelFeatureRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | A name for the feature that will appear on top of the form field | 
**units** | **str** | The units that this feature is using | [optional] 
**description** | **str** |  | [optional] 
**feature_type** | [**FeatureType**](FeatureType.md) |  | 
**possible_values** | [**List[FeaturePossibleValue]**](FeaturePossibleValue.md) |  | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.partially_update_model_feature_request import PartiallyUpdateModelFeatureRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PartiallyUpdateModelFeatureRequest from a JSON string
partially_update_model_feature_request_instance = PartiallyUpdateModelFeatureRequest.from_json(json)
# print the JSON string representation of the object
print(PartiallyUpdateModelFeatureRequest.to_json())

# convert the object into a dict
partially_update_model_feature_request_dict = partially_update_model_feature_request_instance.to_dict()
# create an instance of PartiallyUpdateModelFeatureRequest from a dict
partially_update_model_feature_request_from_dict = PartiallyUpdateModelFeatureRequest.from_dict(partially_update_model_feature_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


