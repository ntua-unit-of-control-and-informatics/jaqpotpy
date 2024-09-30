# DatasetCSV


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**type** | [**DatasetType**](DatasetType.md) |  | 
**input_file** | **bytearray** | A base64 representation in CSV format of the input values. | 
**values** | **List[object]** |  | [optional] 
**status** | **str** |  | [optional] 
**failure_reason** | **str** |  | [optional] 
**model_id** | **int** |  | [optional] 
**model_name** | **str** |  | [optional] 
**executed_at** | **str** |  | [optional] 
**execution_finished_at** | **str** |  | [optional] 
**created_at** | **str** |  | [optional] 
**updated_at** | **str** |  | [optional] 

## Example

```python
from jaqpotpy.api.openapi.models.dataset_csv import DatasetCSV

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetCSV from a JSON string
dataset_csv_instance = DatasetCSV.from_json(json)
# print the JSON string representation of the object
print(DatasetCSV.to_json())

# convert the object into a dict
dataset_csv_dict = dataset_csv_instance.to_dict()
# create an instance of DatasetCSV from a dict
dataset_csv_from_dict = DatasetCSV.from_dict(dataset_csv_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


