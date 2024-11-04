# get_installed_libraries

## Description
The `get_installed_libraries` function retrieves a list of installed Python libraries and their versions. It runs the `pip freeze` command to get a list of installed Python packages and their versions, then parses the output and creates a list of `Library` objects.

## Returns
- `list`: A list of `Library` objects, each containing the name and version of an installed Python package.

## Example
```python
from jaqpotpy.api.get_installed_libraries import get_installed_libraries

libraries = get_installed_libraries()
for library in libraries:
    print(f"Library: {library.name}, Version: {library.version}")