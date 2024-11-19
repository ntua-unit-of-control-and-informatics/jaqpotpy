# Sphinx Docs Generation

This repository uses **Sphinx** to generate documentation in Markdown format.

## Prerequisites

Before generating the documentation, ensure you have the required packages installed:

```bash
pip install sphinx sphinx-markdown-builder
```

## Generating the Documentation

To generate the documentation in Markdown format, run:

```bash
make markdown
```

This command will create the Markdown files in the `_build` folder.

## Folder Structure

- **_build**: Contains the generated Markdown files.

## Initiation command

The command used to generate the docs was:

```shell
sphinx-apidoc -o sphinx_docs . sphinx-apidoc --full -A 'UPCI NTUA'; cd sphinx_docs;
```

from the root folder.

Also in the conf.py these lines were added to run sphinx from the root folder of the project:

```python
import os
import sys

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory of the docs folder
```
