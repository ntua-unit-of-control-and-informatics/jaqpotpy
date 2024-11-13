#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

pip install -U sphinx myst-parser
cd jaqpotpy/sphinx_docs
make markdown

