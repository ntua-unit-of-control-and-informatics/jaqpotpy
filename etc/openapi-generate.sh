#!/bin/sh

JAQPOT_API_PATH=../jaqpot-api

# Install openapi-python-client according to these installation instructions:
# https://github.com/OpenAPITools/openapi-generator?tab=readme-ov-file#1---installation

cd $JAQPOT_API_PATH
git pull
cd -

# Replace all DateTime occurrences with string on the openapi.yaml
sed -i.bak 's/DateTime/string/g' $JAQPOT_API_PATH/src/main/resources/openapi.yaml && rm $JAQPOT_API_PATH/src/main/resources/openapi.yaml.bak

# Generate the openapi client types
openapi-generator generate -i $JAQPOT_API_PATH/src/main/resources/openapi.yaml -g python -o ./jaqpotpy/api/openapi


