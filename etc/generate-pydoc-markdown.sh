#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

pip install -U sphinx sphinx-markdown-builder
cd jaqpotpy/sphinx_docs
sphinx-build -M markdown . ./_build

