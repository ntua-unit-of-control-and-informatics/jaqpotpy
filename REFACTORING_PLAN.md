# Jaqpotpy Refactoring Plan: Shared Prediction Logic

## Overview

This document outlines the plan to extract prediction logic from `jaqpotpy-inference` to `jaqpotpy` to create a shared
codebase for both local development and production inference.

## Goals

1. **Eliminate Code Duplication**: Single source of truth for prediction algorithms
2. **Ensure Consistency**: Identical predictions between local and production
3. **Simplify Maintenance**: Changes made once, applied everywhere
4. **Improve Testing**: Comprehensive testing in jaqpotpy affects production

## Current State Analysis

### jaqpotpy (Local Development)

```
jaqpotpy/api/local_model.py
├── download_model()          # ✅ Enhanced with S3 presigned URLs
├── _download_model_bytes()   # ✅ Database + S3 fallback
├── _download_preprocessor_bytes() # ✅ Database + S3 fallback  
├── predict_local()           # ⚠️ Basic ONNX inference only
├── _preprocess_data()        # ⚠️ Basic preprocessing
└── _run_onnx_inference()     # ⚠️ Single ONNX method for all model types
```

### jaqpotpy-inference (Production)

```
jaqpotpy-inference/src/
├── handlers/                 # 🔥 Model-type-specific prediction logic
│   ├── predict_sklearn_onnx.py
│   ├── predict_torch_onnx.py
│   ├── predict_torch_sequence.py
│   └── predict_torch_geometric.py
├── helpers/
│   ├── predict_methods.py    # 🔥 Core prediction algorithms
│   ├── dataset_utils.py      # 🔥 Data format conversion
│   ├── model_loader.py       # 🔥 ONNX model loading
│   ├── recreate_preprocessor.py # 🔥 Preprocessor reconstruction
│   ├── doa_calc.py          # ⚠️ DOA calculations (duplicates jaqpotpy.doa)
│   └── image_utils.py       # 🔥 Image processing for torch models
└── services/
    └── predict_service.py    # 🔥 Model type routing
```

**Legend**: ✅ = Good as-is, ⚠️ = Needs enhancement, 🔥 = Must extract to jaqpotpy

## Proposed New Architecture

### Enhanced jaqpotpy Structure

```
jaqpotpy/
├── api/
│   ├── local_model.py                    # ✅ Keep enhanced version
│   └── shared_inference.py               # 🆕 Shared inference logic
├── inference/                            # 🆕 New inference package
│   ├── __init__.py
│   ├── handlers/                         # 🔥 Moved from jaqpotpy-inference
│   │   ├── __init__.py
│   │   ├── sklearn_handler.py           # sklearn ONNX prediction
│   │   ├── torch_handler.py             # PyTorch ONNX prediction
│   │   ├── torch_sequence_handler.py    # LSTM/RNN prediction
│   │   └── torch_geometric_handler.py   # Graph neural network prediction
│   ├── core/                            # 🔥 Core prediction logic
│   │   ├── __init__.py
│   │   ├── predict_methods.py           # Core algorithms
│   │   ├── model_loader.py              # ONNX model loading
│   │   ├── dataset_utils.py             # Data format conversion
│   │   └── preprocessor_utils.py        # Preprocessor reconstruction
│   ├── utils/                           # 🔥 Utility functions
│   │   ├── __init__.py
│   │   ├── image_utils.py               # Image processing
│   │   └── tensor_utils.py              # Tensor operations
│   └── service.py                       # 🆕 Prediction service orchestrator
└── ...existing structure...
```

### Simplified jaqpotpy-inference Structure

```
jaqpotpy-inference/src/
├── api/
│   └── predict.py                       # FastAPI endpoint
├── services/
│   └── predict_service.py               # ⚠️ Simplified to use jaqpotpy
├── config/
│   └── config.py                        # Configuration
└── loggers/
    └── logger.py                        # Logging
```

## Detailed Migration Plan

### Phase 1: Extract Core Prediction Logic

#### 1.1 Create jaqpotpy.inference package

```bash
cd jaqpotpy
mkdir -p jaqpotpy/inference/{handlers,core,utils}
touch jaqpotpy/inference/__init__.py
touch jaqpotpy/inference/{handlers,core,utils}/__init__.py
```

#### 1.2 Move predict_methods.py

```python
# jaqpotpy/inference/core/predict_methods.py
# Copied from jaqpotpy-inference/src/helpers/predict_methods.py

def predict_sklearn_onnx(model, preprocessor, data, request):
    """Predict using sklearn ONNX model with preprocessing."""
    # Enhanced version that works for both local and production


def predict_torch_onnx(model, preprocessor, data, request):
    """Predict using PyTorch ONNX model with image support."""
    # Enhanced version with image processing


def predict_torch_sequence(model, preprocessor, data, request):
    """Predict using PyTorch sequence models (LSTM, RNN)."""
    # Sequence model prediction logic


def predict_torch_geometric(model, preprocessor, data, request):
    """Predict using PyTorch Geometric models."""
    # Graph neural network prediction logic


def calculate_doas(input_feed, request):
    """Calculate Domain of Applicability using jaqpotpy.doa classes."""
    # Use existing jaqpotpy.doa instead of duplicating
```

#### 1.3 Move dataset utilities

```python
# jaqpotpy/inference/core/dataset_utils.py
# Copied from jaqpotpy-inference/src/helpers/dataset_utils.py

def build_tabular_dataset_from_request(request):
    """Convert prediction request to tabular dataset."""


def build_tensor_dataset_from_request(request):
    """Convert prediction request to tensor dataset."""


def build_graph_dataset_from_request(request):
    """Convert prediction request to graph dataset."""
```

#### 1.4 Move model loading utilities

```python
# jaqpotpy/inference/core/model_loader.py
# Enhanced version combining jaqpotpy local_model + jaqpotpy-inference logic

def load_onnx_model_from_request(request):
    """Load ONNX model from request (base64 or S3 URL)."""


def load_onnx_model_from_metadata(model_metadata, jaqpot_client=None):
    """Load ONNX model from metadata for local development."""
```

### Phase 2: Create Model Type Handlers

#### 2.1 Sklearn Handler

```python
# jaqpotpy/inference/handlers/sklearn_handler.py

from ..core.predict_methods import predict_sklearn_onnx
from ..core.dataset_utils import build_tabular_dataset_from_request


def handle_sklearn_prediction(request, local_mode=False):
    """Handle sklearn ONNX prediction for both local and production."""
    if local_mode:
        # Use local model loading logic
        model_data = load_local_model(request.model)
    else:
        # Use production request-based loading
        model_data = load_onnx_model_from_request(request)

    return predict_sklearn_onnx(model_data.model, model_data.preprocessor, data, request)
```

#### 2.2 Torch Handlers (similar pattern)

```python
# jaqpotpy/inference/handlers/torch_handler.py
# jaqpotpy/inference/handlers/torch_sequence_handler.py  
# jaqpotpy/inference/handlers/torch_geometric_handler.py
```

### Phase 3: Create Unified Prediction Service

#### 3.1 Prediction Service Orchestrator

```python
# jaqpotpy/inference/service.py

from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse
from .handlers import (
    sklearn_handler,
    torch_handler,
    torch_sequence_handler,
    torch_geometric_handler
)


class PredictionService:
    """Unified prediction service for both local and production inference."""

    def __init__(self, local_mode=False, jaqpot_client=None):
        self.local_mode = local_mode
        self.jaqpot_client = jaqpot_client

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Route prediction to appropriate handler based on model type."""
        match request.model.type:
            case ModelType.SKLEARN_ONNX:
                return sklearn_handler.handle_sklearn_prediction(request, self.local_mode)
            case ModelType.TORCH_ONNX:
                return torch_handler.handle_torch_prediction(request, self.local_mode)
            case ModelType.TORCH_SEQUENCE_ONNX:
                return torch_sequence_handler.handle_sequence_prediction(request, self.local_mode)
            case ModelType.TORCH_GEOMETRIC_ONNX | ModelType.TORCHSCRIPT:
                return torch_geometric_handler.handle_geometric_prediction(request, self.local_mode)
            case _:
                raise ValueError(f"Model type {request.model.type} not supported")
```

### Phase 4: Update Local Model API

#### 4.1 Enhanced Local Model

```python
# jaqpotpy/api/downloaded_model.py (updated)

from ..inference.service import PredictionService
from jaqpot_api_client import PredictionRequest, ModelType


class JaqpotDownloadedModel:
    def __init__(self, jaqpot_client):
        self.jaqpot_client = jaqpot_client
        self._cached_models = {}
        self._cached_preprocessors = {}
        self.prediction_service = PredictionService(local_mode=True, jaqpot_client=jaqpot_client)

    def predict_local(self, model_data, data):
        """Enhanced prediction using shared inference logic."""
        # Create PredictionRequest from local data
        request = self._create_prediction_request(model_data, data)

        # Use shared prediction service
        response = self.prediction_service.predict(request)

        return response

    def _create_prediction_request(self, model_data, data):
        """Convert local model data and input to PredictionRequest format."""
        # Implementation to bridge local and production formats
```

### Phase 5: Simplify jaqpotpy-inference

#### 5.1 Updated FastAPI Service

```python
# jaqpotpy-inference/src/services/predict_service.py (simplified)

from jaqpotpy.inference.service import PredictionService
from jaqpot_api_client import PredictionRequest, PredictionResponse

# Global prediction service instance
prediction_service = PredictionService(local_mode=False)


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    """Simplified prediction service using jaqpotpy shared logic."""
    return prediction_service.predict(req)
```

## Migration Steps

### Step 1: Preparation

1. ✅ Test current jaqpotpy local model functionality
2. ✅ Fix jaqpot-api compilation errors
3. 🔄 Verify jaqpot-api presigned URL endpoints work

### Step 2: Extract Core Logic

1. Create jaqpotpy/inference package structure
2. Copy predict_methods.py from jaqpotpy-inference
3. Copy dataset_utils.py and model_loader.py
4. Update imports and dependencies

### Step 3: Create Handlers

1. Extract handler logic from jaqpotpy-inference
2. Create unified handlers in jaqpotpy
3. Add local_mode support to all handlers

### Step 4: Update Local Model

1. Integrate PredictionService with JaqpotDownloadedModel
2. Add model type detection and routing
3. Test local prediction with all model types

### Step 5: Update jaqpotpy-inference

1. Update jaqpotpy-inference to depend on jaqpotpy
2. Simplify prediction service to use shared logic
3. Remove duplicated code

### Step 6: Integration Testing

1. Test local predictions match production predictions
2. Performance testing for both paths
3. End-to-end testing across all repositories

## Dependencies and Requirements

### jaqpotpy New Dependencies

```python
# pyproject.toml additions
dependencies = [
    # ... existing dependencies ...
    "jaqpot-api-client",  # For PredictionRequest/Response types
    "onnxruntime",  # Already exists
    "torch",  # For tensor operations
    "pillow",  # For image processing
    # ... others from jaqpotpy-inference requirements.txt
]
```

### jaqpotpy-inference Updated Dependencies

```python
# requirements.txt simplified
jaqpotpy >= 1.
XX.YY  # Depend on jaqpotpy for prediction logic
fastapi
uvicorn
pydantic - settings
boto3  # For S3 operations
```

## Testing Strategy

### Unit Tests

```python
# tests/test_inference_service.py
def test_sklearn_prediction_consistency():
    """Test local vs production sklearn predictions match."""


def test_torch_prediction_consistency():
    """Test local vs production torch predictions match."""


# tests/test_handlers.py  
def test_sklearn_handler_local_mode():
    """Test sklearn handler in local mode."""


def test_sklearn_handler_production_mode():
    """Test sklearn handler in production mode."""
```

### Integration Tests

```python
# test_integration.py
def test_end_to_end_prediction_workflow():
    """Test complete workflow from jaqpotpy upload to jaqpotpy-inference prediction."""
```

## Risk Mitigation

### Backwards Compatibility

- Keep existing `JaqpotDownloadedModel.predict_local()` API unchanged
- Deprecate old methods gradually, don't break immediately

### Rollback Plan

- Keep jaqpotpy-inference working independently during migration
- Feature flags to enable/disable shared logic
- Comprehensive testing before production deployment

### Performance Considerations

- Benchmark prediction performance before/after refactoring
- Monitor memory usage with shared dependencies
- Optimize import patterns to avoid circular dependencies

This refactoring will create a single source of truth for prediction logic while maintaining the flexibility of both
local development and production inference workflows.
