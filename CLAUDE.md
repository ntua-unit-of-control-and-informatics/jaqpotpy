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

## Local Model Development

### JaqpotLocalModel Class (`api/local_model.py`)
- Downloads models from Jaqpot platform for local testing
- Supports both database-stored and S3-stored models with presigned URLs
- Handles preprocessing pipeline reconstruction
- Provides identical inference logic to production service
- Includes model caching for improved performance

### Key Methods
- `download_model(model_id)`: Downloads model and preprocessor from platform
- `predict_local(model_data, data)`: Runs inference using ONNX runtime
- `_preprocess_data()`: Applies preprocessing transformations
- `_run_onnx_inference()`: Executes ONNX model inference

### Testing Local Models
```bash
# Test local model download and inference
python test_local_model_download.py --model-id <model_id> [--local]

# Test with localhost API
python test_local_model_download.py --model-id <model_id> --local
```

## Integration Points

### With jaqpot-api
- Authenticates using Keycloak OAuth2 flow
- Uploads models with metadata to platform
- Downloads models for local development and testing
- **NEW**: Uses presigned URLs for large model downloads from S3

### With jaqpotpy-inference
- **Critical**: Local model logic must stay synchronized with production inference
- Shares ONNX conversion and preprocessing logic
- Both use identical prediction algorithms and data handling
- **PLANNED**: Shared prediction logic extraction to eliminate duplication

## Major Refactoring in Progress

### Current Challenge: Code Duplication
The local model implementation in `jaqpotpy/api/local_model.py` and production inference in `jaqpotpy-inference/` have duplicated prediction logic. This creates maintenance burden and consistency risks.

### Planned Architecture Changes

#### Phase 1: Extract Shared Prediction Logic
```
jaqpotpy/inference/                    # 🆕 New shared inference package
├── handlers/                          # Model-type-specific handlers
│   ├── sklearn_handler.py            # sklearn ONNX prediction
│   ├── torch_handler.py              # PyTorch ONNX prediction
│   ├── torch_sequence_handler.py     # LSTM/RNN prediction
│   └── torch_geometric_handler.py    # Graph neural network prediction
├── core/                             # Core prediction algorithms
│   ├── predict_methods.py            # Moved from jaqpotpy-inference
│   ├── model_loader.py               # Enhanced ONNX loading
│   ├── dataset_utils.py              # Data format conversion
│   └── preprocessor_utils.py         # Preprocessor reconstruction
├── utils/                            # Utility functions
│   ├── image_utils.py                # Image processing for torch models
│   └── tensor_utils.py               # Tensor operations
└── service.py                        # Unified prediction orchestrator
```

#### Phase 2: Unified Prediction Service
```python
# Usage in local development
from jaqpotpy.inference.service import PredictionService

service = PredictionService(local_mode=True, jaqpot_client=jaqpot)
response = service.predict(prediction_request)

# Usage in production (jaqpotpy-inference)
from jaqpotpy.inference.service import PredictionService

service = PredictionService(local_mode=False)
response = service.predict(prediction_request)
```

### Benefits of Refactoring
1. **Single Source of Truth**: Prediction logic maintained in one place
2. **Guaranteed Consistency**: Local and production inference produce identical results
3. **Simplified Testing**: Test once in jaqpotpy, confidence in production
4. **Reduced Maintenance**: Changes applied once, affect both environments
5. **Enhanced Local Development**: Full production feature parity locally

### Key Files for Refactoring

#### Current Implementation
- `jaqpotpy/api/local_model.py`: Local model download and basic inference
- `jaqpotpy-inference/src/helpers/predict_methods.py`: Production prediction algorithms
- `jaqpotpy-inference/src/handlers/`: Model-type-specific production handlers

#### Target Implementation  
- `jaqpotpy/inference/service.py`: Unified prediction service
- `jaqpotpy/inference/handlers/`: Shared model-type handlers
- `jaqpotpy/api/local_model.py`: Enhanced to use shared inference logic
- `jaqpotpy-inference/`: Simplified to depend on jaqpotpy shared logic

### Synchronization Requirements

Changes to prediction logic in jaqpotpy local models must be reflected in jaqpotpy-inference:
- Model loading and caching strategies
- Data preprocessing steps
- ONNX runtime configuration
- Domain of Applicability calculations
- Error handling and fallback logic

**See**: `REFACTORING_PLAN.md` for detailed migration strategy
**See**: `SYNCHRONIZATION.md` in jaqpot-claude-context for current sync requirements

## Current Priorities

### Immediate Tasks
1. **Fix jaqpot-api compilation errors** (see `jaqpot-api/FIX_COMPILATION_ERRORS.md`)
2. **Test presigned URL download** using `test_local_model_download.py`
3. **Verify local model functionality** with enhanced S3 support

### Next Phase
1. **Extract prediction logic** from jaqpotpy-inference to jaqpotpy
2. **Create unified inference service** supporting both local and production modes
3. **Update jaqpotpy-inference** to use shared jaqpotpy logic
4. **Comprehensive integration testing** across all three repositories

## Testing Strategy

### Local Model Testing
```bash
# Install in development mode
pip install -e .

# Test basic functionality
python test_local_model_download.py --model-id <sklearn-model-id>
python test_local_model_download.py --model-id <torch-model-id>

# Test with different API environments
python test_local_model_download.py --model-id <model-id> --local           # localhost
python test_local_model_download.py --model-id <model-id>                   # production
```

### Integration Testing
```bash
# After refactoring: test consistency between local and production
python test_prediction_consistency.py --model-id <model-id>
```

## Release Process

Releases are automated via GitHub Actions when creating a release on GitHub. Follow semantic versioning (1.XX.YY format).

### Post-Refactoring Release Strategy
1. **Major Version Bump**: When shared inference logic is integrated (breaking changes)
2. **Minor Version Bump**: When new model types or features are added
3. **Patch Version Bump**: Bug fixes and performance improvements

## Dependencies Management

### Current Dependencies
Standard ML stack: numpy, pandas, scikit-learn, torch, onnx, etc.

### Post-Refactoring Dependencies
```python
# Additional dependencies for shared inference logic
"jaqpot-api-client",     # For PredictionRequest/Response types
"fastapi",               # For request/response models (optional)
"pillow",                # For image processing in torch models
```

**Important**: jaqpotpy-inference will depend on jaqpotpy post-refactoring, reversing the current independence.