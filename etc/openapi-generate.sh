#!/bin/sh

# Install openapi-python-client
pip install openapi-python-client

# Replace all DateTime occurrences with string on the openapi.yaml
sed -i 's/DateTime/string/g' ../jaqpot-api/src/main/resources/openapi.yaml

# Generate the openapi client types
openapi-python-client generate --path ../jaqpot-api/src/main/resources/openapi.yaml --output-path ./jaqpotpy/api/openapi --overwrite

