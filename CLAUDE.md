# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jaqpotpy is a Python client library for deploying machine learning models to the Jaqpot platform. It supports scikit-learn, PyTorch, PyTorch Geometric, and XGBoost models with features for molecular descriptors, domain of applicability, and ONNX conversion.

## Core Architecture

### Main Components

- **jaqpot.py**: Main client class for authentication and model deployment to Jaqpot platform
- **models/**: Model wrappers and base classes
  - `base_classes.py`: Abstract Model class with common interfaces
  - `sklearn.py`: Scikit-learn model wrapper
  - `torch_models/`: PyTorch model implementations
  - `torch_geometric_models/`: Graph neural network models
  - `docker_model.py`: Docker-based model deployment
- **datasets/**: Dataset handling and preprocessing
  - `dataset_base.py`: Abstract BaseDataset class
  - `jaqpot_tabular_dataset.py`: Tabular data handling
  - `graph_pyg_dataset.py`: Graph data for PyTorch Geometric
- **descriptors/**: Molecular featurization
  - `molecular/`: RDKit, Mordred, MACCS keys fingerprints
  - `graph/`: Graph-based molecular features
- **doa/**: Domain of Applicability implementation
- **preprocessors/**: Data preprocessing utilities
- **transformers/**: Data transformation utilities

### Key Design Patterns

- Models inherit from base Model class and implement fit/predict interface
- Datasets inherit from BaseDataset with common preprocessing methods
- Descriptors use MolecularFeaturizer base class
- ONNX conversion integrated for model serialization
- S3 upload for large models via presigned URLs

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest path/to/test.py   # Run specific test file
```

### Linting and Formatting
```bash
ruff check               # Lint code (configured in ruff.toml)
ruff format              # Format code
flake8 .                 # Additional linting (CI uses specific flags)
```

### Building
```bash
pip install -e .         # Install in development mode
python -m build          # Build distribution packages
```

### Documentation
```bash
cd jaqpotpy/sphinx_docs
make html                # Build Sphinx documentation
```

## Key Configuration Files

- `pyproject.toml`: Project metadata and dependencies (uses hatchling build system)
- `requirements.txt`: Generated from requirements.in via pip-compile
- `ruff.toml`: Linting configuration (excludes api/openapi generated code)
- `.github/workflows/build.yml`: CI pipeline with pytest, flake8, and ruff

## Testing Structure

Tests are co-located with source code in `tests/` subdirectories:
- `jaqpotpy/datasets/tests/`
- `jaqpotpy/descriptors/tests/`
- `jaqpotpy/doa/tests/`
- `jaqpotpy/models/tests/`

## Examples and Usage

The `examples/` directory contains working examples for:
- sklearn model deployment
- PyTorch model training and deployment
- PyTorch Geometric graph models
- Docker model upload
- Advanced features like domain of applicability

## Release Process

Releases are automated via GitHub Actions when creating a release on GitHub. Follow semantic versioning (1.XX.YY format).