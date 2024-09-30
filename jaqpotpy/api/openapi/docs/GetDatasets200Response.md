# GetDatasets200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**content** | [**List[Dataset]**](Dataset.md) |  | [optional] 
**total_elements** | **int** |  | [optional] 
**total_pages** | **int** |  | [optional] 
**page_size** | **int** |  | [optional] 
**page_number** | **int** |  | [optional] 

## Example

```python
from openapi_client.models.get_datasets200_response import GetDatasets200Response

# TODO update the JSON string below
json = "{}"
# create an instance of GetDatasets200Response from a JSON string
get_datasets200_response_instance = GetDatasets200Response.from_json(json)
# print the JSON string representation of the object
print(GetDatasets200Response.to_json())

# convert the object into a dict
get_datasets200_response_dict = get_datasets200_response_instance.to_dict()
# create an instance of GetDatasets200Response from a dict
get_datasets200_response_from_dict = GetDatasets200Response.from_dict(get_datasets200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


