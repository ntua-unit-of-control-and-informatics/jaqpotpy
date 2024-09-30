# ModelExtraConfig

A JSON object containing extra configuration for the model

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**torch_config** | **Dict[str, object]** |  | [optional] 
**preprocessors** | [**List[Transformer]**](Transformer.md) |  | [optional] 
**featurizers** | [**List[Transformer]**](Transformer.md) |  | [optional] 
**doa** | [**List[Transformer]**](Transformer.md) |  | [optional] 

## Example

```python
from openapi_client.models.model_extra_config import ModelExtraConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ModelExtraConfig from a JSON string
model_extra_config_instance = ModelExtraConfig.from_json(json)
# print the JSON string representation of the object
print(ModelExtraConfig.to_json())

# convert the object into a dict
model_extra_config_dict = model_extra_config_instance.to_dict()
# create an instance of ModelExtraConfig from a dict
model_extra_config_from_dict = ModelExtraConfig.from_dict(model_extra_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


