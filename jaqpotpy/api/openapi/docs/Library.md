# Library


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **int** |  | [optional] 
**name** | **str** |  | 
**version** | **str** |  | 
**created_at** | **datetime** | The date and time when the feature was created. | [optional] 
**updated_at** | **str** | The date and time when the feature was last updated. | [optional] 

## Example

```python
from openapi_client.models.library import Library

# TODO update the JSON string below
json = "{}"
# create an instance of Library from a JSON string
library_instance = Library.from_json(json)
# print the JSON string representation of the object
print(Library.to_json())

# convert the object into a dict
library_dict = library_instance.to_dict()
# create an instance of Library from a dict
library_from_dict = Library.from_dict(library_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


