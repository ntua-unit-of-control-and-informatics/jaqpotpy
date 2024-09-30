# Dataset


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**type** | [**DatasetType**](DatasetType.md) |  | 
**entry_type** | **str** |  | 
**input** | **List[object]** |  | 
**result** | **List[object]** |  | [optional] 
**status** | **str** |  | [optional] 
**failure_reason** | **str** |  | [optional] 
**user_id** | **str** |  | [optional] 
**model_id** | **int** |  | [optional] 
**model_name** | **str** |  | [optional] 
**executed_at** | **str** |  | [optional] 
**execution_finished_at** | **str** |  | [optional] 
**created_at** | **str** |  | [optional] 
**updated_at** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.dataset import Dataset

# TODO update the JSON string below
json = "{}"
# create an instance of Dataset from a JSON string
dataset_instance = Dataset.from_json(json)
# print the JSON string representation of the object
print(Dataset.to_json())

# convert the object into a dict
dataset_dict = dataset_instance.to_dict()
# create an instance of Dataset from a dict
dataset_from_dict = Dataset.from_dict(dataset_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


