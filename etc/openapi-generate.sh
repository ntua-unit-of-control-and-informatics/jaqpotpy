#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

JAQPOT_API_PATH=../jaqpot-api
TEMP_DIR=$(mktemp -d)
TARGET_DIR=./jaqpotpy/api/openapi

# Install openapi-generator-cli if not already installed
if ! command -v openapi-generator-cli &> /dev/null
then
    echo "openapi-generator could not be found, installing..."
    npm install @openapitools/openapi-generator-cli -g
fi

# Update Jaqpot API
cd $JAQPOT_API_PATH
git pull
cd -

# Remove all pattern validations to avoid throwing errors when a pattern is not satisfied
# legacy models do not satisfy most patterns
# TODO skip validation on pydantic responses when this https://github.com/OpenAPITools/openapi-generator/issues/19357 is implemented
sed -i.bak 's/pattern: "[^"]*"//g' $JAQPOT_API_PATH/src/main/resources/openapi.yaml && rm $JAQPOT_API_PATH/src/main/resources/openapi.yaml.bak
# Remove empty lines that might be left after pattern removal
sed -i.bak '/^[[:space:]]*$/d' $JAQPOT_API_PATH/src/main/resources/openapi.yaml && rm $JAQPOT_API_PATH/src/main/resources/openapi.yaml.bak


# Generate the OpenAPI client in a temporary directory
openapi-generator-cli generate \
    -i $JAQPOT_API_PATH/src/main/resources/openapi.yaml \
    -g python \
    -o $TEMP_DIR \
    --additional-properties packageName=jaqpotpy.api.openapi

# Ensure the target directory exists
mkdir -p $TARGET_DIR

# Move only the necessary files
cp -r $TEMP_DIR/jaqpotpy/api/openapi/* $TARGET_DIR/

# Clean up
rm -rf $TEMP_DIR

echo "OpenAPI client generated successfully in $TARGET_DIR"
