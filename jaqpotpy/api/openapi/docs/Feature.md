# Feature


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**meta** | **Dict[str, object]** | A JSON object containing meta information. | [optional] 
**key** | **str** | A key that must start with a letter, followed by any combination of letters, digits, hyphens, or underscores. For example, &#39;abc123&#39;, &#39;abc-test&#39;, or &#39;Abc_test&#39;. It cannot start with a digit. | 
**name** | **str** | A name for the feature that will appear on top of the form field | 
**units** | **str** | A name for the feature that will appear on top of the form field | [optional] 
**description** | **str** |  | [optional] 
**feature_type** | [**FeatureType**](FeatureType.md) |  | 
**feature_dependency** | **str** |  | [optional] 
**visible** | **bool** |  | [optional] 
**possible_values** | [**List[FeaturePossibleValue]**](FeaturePossibleValue.md) |  | [optional] 
**created_at** | **datetime** | The date and time when the feature was created. | [optional] 
**updated_at** | **str** | The date and time when the feature was last updated. | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.feature import Feature

# TODO update the JSON string below
json = "{}"
# create an instance of Feature from a JSON string
feature_instance = Feature.from_json(json)
# print the JSON string representation of the object
print(Feature.to_json())

# convert the object into a dict
feature_dict = feature_instance.to_dict()
# create an instance of Feature from a dict
feature_from_dict = Feature.from_dict(feature_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


