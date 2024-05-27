#!/bin/sh

# [DEPRECATED] Previous script to upload jaqpotpy library to pip
# This will be moved to a github action after creating a release

python setup.py bdist_wheel
twine upload dist/*
twine upload --repository testpypi dist/jaqpotpy-2.0.0b0-py3-none-any.whl
twine upload dist/jaqpotpy-1.0.84-py3-none-any.whl
docker build -t upcintua/jaqpotpy:1.0.3 --no-cache python setup.py bdist_wheel--build-arg tag=1.0.3 .
