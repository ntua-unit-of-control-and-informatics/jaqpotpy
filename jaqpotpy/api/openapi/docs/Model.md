# Model


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**meta** | **Dict[str, object]** | A JSON object containing meta information. | [optional] 
**name** | **str** |  | 
**description** | **str** |  | [optional] 
**type** | [**ModelType**](ModelType.md) |  | 
**jaqpotpy_version** | **str** |  | 
**libraries** | [**List[Library]**](Library.md) |  | 
**dependent_features** | [**List[Feature]**](Feature.md) |  | 
**independent_features** | [**List[Feature]**](Feature.md) |  | 
**shared_with_organizations** | [**List[Organization]**](Organization.md) |  | [optional] 
**visibility** | [**ModelVisibility**](ModelVisibility.md) |  | 
**task** | [**ModelTask**](ModelTask.md) |  | 
**actual_model** | **bytearray** | A base64 representation of the actual model. | 
**creator** | [**User**](User.md) |  | [optional] 
**can_edit** | **bool** | If the current user can edit the model | [optional] 
**is_admin** | **bool** |  | [optional] 
**tags** | **str** |  | [optional] 
**legacy_prediction_service** | **str** |  | [optional] 
**extra_config** | [**ModelExtraConfig**](ModelExtraConfig.md) |  | [optional] 
**created_at** | **datetime** | The date and time when the feature was created. | [optional] 
**updated_at** | **str** | The date and time when the feature was last updated. | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.model import Model

# TODO update the JSON string below
json = "{}"
# create an instance of Model from a JSON string
model_instance = Model.from_json(json)
# print the JSON string representation of the object
print(Model.to_json())

# convert the object into a dict
model_dict = model_instance.to_dict()
# create an instance of Model from a dict
model_from_dict = Model.from_dict(model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


