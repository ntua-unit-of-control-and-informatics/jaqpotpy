# ModelSummary


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | 
**name** | **str** |  | 
**visibility** | [**ModelVisibility**](ModelVisibility.md) |  | 
**description** | **str** |  | [optional] 
**creator** | [**User**](User.md) |  | [optional] 
**type** | [**ModelType**](ModelType.md) |  | 
**dependent_features_length** | **int** |  | [optional] 
**independent_features_length** | **int** |  | [optional] 
**shared_with_organizations** | [**List[OrganizationSummary]**](OrganizationSummary.md) |  | 
**created_at** | **datetime** | The date and time when the feature was created. | 
**updated_at** | **str** | The date and time when the feature was last updated. | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.model_summary import ModelSummary

# TODO update the JSON string below
json = "{}"
# create an instance of ModelSummary from a JSON string
model_summary_instance = ModelSummary.from_json(json)
# print the JSON string representation of the object
print(ModelSummary.to_json())

# convert the object into a dict
model_summary_dict = model_summary_instance.to_dict()
# create an instance of ModelSummary from a dict
model_summary_from_dict = ModelSummary.from_dict(model_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


