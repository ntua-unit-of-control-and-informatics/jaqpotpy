# Integration Testing Strategy: jaqpot-api ↔ jaqpotpy ↔ jaqpotpy-inference

## Overview

This document outlines the comprehensive testing strategy for ensuring seamless integration across the three core Jaqpot
repositories during the prediction logic refactoring.

## Testing Phases

### Phase 1: Current State Validation (Pre-Refactoring)

#### 1.1 API Endpoint Testing

```bash
# Fix jaqpot-api compilation errors first
cd jaqpot-api
./gradlew build
./gradlew bootRun

# Test new presigned URL endpoints
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8080/v1/models/{model-id}/download/urls?expirationMinutes=30"

curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8080/v1/models/{model-id}/preprocessor/download/urls?expirationMinutes=30"
```

#### 1.2 Offline Model Download Testing

```bash
# Test jaqpotpy offline model functionality
cd jaqpotpy
python test_local_model_download.py --model-id <sklearn-model-id> --local
python test_local_model_download.py --model-id <torch-model-id> --local

# Test with production API
python test_local_model_download.py --model-id <sklearn-model-id>
python test_local_model_download.py --model-id <torch-model-id>
```

#### 1.3 Production Inference Testing

```bash
# Test jaqpotpy-inference with existing models
cd jaqpotpy-inference
python main.py

# Send test prediction requests
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d @test_prediction_request.json
```

### Phase 2: Prediction Consistency Validation

#### 2.1 Cross-Platform Prediction Testing

Create test that verifies local and production predictions match:

```python
# test_prediction_consistency.py

import numpy as np
from jaqpotpy import Jaqpot
import requests


def test_prediction_consistency(model_id, test_data):
   """Test that local and production predictions match exactly."""

   # Offline prediction using jaqpotpy
   jaqpot = Jaqpot()
   jaqpot.login()

   # Download model data using the new API
   model_data = jaqpot.download_model(model_id)
   offline_response = jaqpot.predict_offline(model_data, test_data)

   # Production prediction using jaqpotpy-inference
   production_request = create_prediction_request(model_id, test_data)
   production_response = requests.post(
      "http://localhost:8002/predict",
      json=production_request
   )

   # Compare predictions
   offline_predictions = offline_response.predictions
   production_predictions = production_response.json()["predictions"]

   # Allow for small floating point differences
   np.testing.assert_allclose(
      offline_predictions,
      production_predictions,
      rtol=1e-10,
      atol=1e-10,
      err_msg=f"Predictions don't match for model {model_id}"
   )

   return True


# Test different model types
test_prediction_consistency("sklearn-model-id", sklearn_test_data)
test_prediction_consistency("torch-model-id", torch_test_data)
test_prediction_consistency("torch-geometric-model-id", graph_test_data)
```

### Phase 3: Refactoring Integration Testing

#### 3.1 Shared Logic Extraction Validation

After extracting prediction logic to jaqpotpy:

```python
# test_shared_inference.py

from jaqpotpy.inference.service import PredictionService
from jaqpotpy.offline.offline_model_data import OfflineModelData
from jaqpot_api_client.models.dataset import Dataset
import numpy as np

def test_shared_inference_service():
    """Test the new shared inference service."""
    
    service = PredictionService()
    
    # Test with different model types using unified API
    sklearn_response = service.predict(sklearn_model_data, dataset, "SKLEARN_ONNX")
    torch_response = service.predict(torch_model_data, dataset, "TORCH_ONNX")
    
    # Should produce consistent results with raw model data
    assert isinstance(sklearn_response.predictions, list)
    assert isinstance(torch_response.predictions, list)

def test_unified_api():
    """Test the unified API across model types."""
    
    from jaqpotpy.inference.core.predict_methods import (
        predict_sklearn_onnx,
        predict_torch_onnx
    )
    
    # All prediction methods now use (model_data, dataset) signature
    sklearn_predictions, sklearn_probs, sklearn_doa = predict_sklearn_onnx(model_data, dataset)
    torch_predictions = predict_torch_onnx(model_data, tensor_dataset)
    
    # Should work with OfflineModelData directly
    assert isinstance(sklearn_predictions, np.ndarray)
    assert isinstance(torch_predictions, np.ndarray)
```

#### 3.2 jaqpotpy-inference Simplification Testing

After updating jaqpotpy-inference to use jaqpotpy:

```python
# test_simplified_inference.py

def test_simplified_inference_service():
    """Test that simplified jaqpotpy-inference still works correctly."""
    
    # The new simplified predict_service.py should just call jaqpotpy
    from jaqpotpy_inference.src.services.predict_service import run_prediction
    
    response = run_prediction(prediction_request)
    
    # Should produce same results as before refactoring
    assert response.predictions == expected_predictions
```

### Phase 4: End-to-End Integration Testing

#### 4.1 Complete Workflow Testing

```python
# test_end_to_end_workflow.py

def test_complete_model_lifecycle():
   """Test complete model lifecycle across all repositories."""

   # 1. Train and upload model using jaqpotpy
   jaqpot = Jaqpot()
   jaqpot.login()

   model = SklearnModel(...)  # Train sklearn model
   model_id = jaqpot.deploy_model(model, name="e2e-test-model")

   # 2. Download and test offline using jaqpotpy
   model_data = jaqpot.download_model(model_id)
   offline_predictions = jaqpot.predict_offline(model_data, test_data)

   # 3. Test production inference using jaqpotpy-inference
   production_request = create_prediction_request(model_id, test_data)
   production_response = requests.post(
      "http://localhost:8002/predict",
      json=production_request
   )

   # 4. Verify all three paths produce identical results
   upload_predictions = model.predict(test_data)  # Original model predictions

   np.testing.assert_allclose(upload_predictions, offline_predictions.predictions)
   np.testing.assert_allclose(offline_predictions.predictions, production_response.json()["predictions"])

   # 5. Clean up
   jaqpot.delete_model(model_id)
```

#### 4.2 Performance Testing

```python
# test_performance.py

def test_prediction_performance():
   """Test that refactoring doesn't degrade performance."""

   import time

   # Measure offline prediction performance
   start_time = time.time()
   for _ in range(100):
      jaqpot.predict_offline(model_data, test_data)
   offline_time = time.time() - start_time

   # Measure production prediction performance
   start_time = time.time()
   for _ in range(100):
      requests.post("http://localhost:8002/predict", json=prediction_request)
   production_time = time.time() - start_time

   # Performance should be comparable (within 50% difference)
   assert abs(offline_time - production_time) / min(offline_time, production_time) < 0.5
```

## Test Data Requirements

### Model Types to Test

1. **Sklearn Models**:
    - Classification (binary, multiclass)
    - Regression (single output, multi-output)
    - With and without preprocessors

2. **PyTorch Models**:
    - Image input models
    - Tabular data models
    - Sequence models (LSTM, RNN)

3. **PyTorch Geometric Models**:
    - Graph classification
    - Graph regression
    - Node prediction

### Test Data Sets

```python
# Generate comprehensive test data
test_datasets = {
    "sklearn_tabular": generate_sklearn_test_data(),
    "torch_images": generate_image_test_data(),
    "torch_sequences": generate_sequence_test_data(),
    "torch_geometric": generate_graph_test_data(),
}
```

## Continuous Integration Setup

### GitHub Actions Workflow

```yaml
# .github/workflows/integration-test.yml

name: Integration Tests
on: [push, pull_request]

jobs:
  integration-test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
      with:
        # Checkout all three repositories
        repository: jaqpot-api
        path: jaqpot-api
        
    - uses: actions/checkout@v3
      with:
        repository: jaqpotpy
        path: jaqpotpy
        
    - uses: actions/checkout@v3
      with:
        repository: jaqpotpy-inference
        path: jaqpotpy-inference
    
    - name: Setup Java
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Start jaqpot-api
      run: |
        cd jaqpot-api
        ./gradlew bootRun &
        # Wait for API to be ready
        
    - name: Install jaqpotpy
      run: |
        cd jaqpotpy
        pip install -e .
        
    - name: Start jaqpotpy-inference
      run: |
        cd jaqpotpy-inference
        pip install -r requirements.txt
        python main.py &
    
    - name: Run Integration Tests
      run: |
        cd jaqpotpy
        python test_local_model_download.py --model-id ${{ secrets.TEST_MODEL_ID }}
        python test_prediction_consistency.py
        python test_end_to_end_workflow.py
```

## Testing Timeline and Milestones

### Milestone 1: API Fix and Basic Testing (Week 1)

- ✅ Fix jaqpot-api compilation errors
- ✅ Test presigned URL endpoints
- ✅ Test offline model download functionality

### Milestone 2: Prediction Consistency (Week 2)

- Test current offline vs production prediction consistency
- Identify and fix any prediction discrepancies
- Establish baseline performance metrics

### Milestone 3: Shared Logic Integration (Week 3-4)

- Extract prediction logic to jaqpotpy
- Test shared inference service
- Validate handler consistency

### Milestone 4: Production Integration (Week 5)

- Update jaqpotpy-inference to use jaqpotpy
- Test simplified inference service
- End-to-end workflow validation

### Milestone 5: Performance and Stability (Week 6)

- Performance testing and optimization
- Load testing
- Production deployment validation

## Error Scenarios and Edge Cases

### Model Loading Edge Cases

- Models stored only in database (no S3)
- Models stored only in S3 (no database fallback)
- Corrupted model files
- Network failures during download
- Authentication failures

### Prediction Edge Cases

- Invalid input data formats
- Missing features
- Out-of-range values
- Memory constraints with large models
- Timeout scenarios

### Integration Edge Cases

- Version mismatches between repositories
- API schema changes
- Dependency conflicts
- Environment differences (dev vs production)

## Success Criteria

### Functional Requirements

- [ ] All model types (sklearn, torch, torch_geometric) work identically in offline and production
- [ ] Presigned URL download works for both models and preprocessors
- [ ] Error handling is consistent across all platforms
- [ ] Performance is maintained within acceptable bounds

### Quality Requirements

- [ ] 100% test coverage for critical prediction paths
- [ ] Zero prediction discrepancies between local and production
- [ ] <5% performance degradation after refactoring
- [ ] All integration tests pass in CI/CD pipeline

### Operational Requirements

- [ ] Easy rollback strategy if issues arise
- [ ] Clear monitoring and alerting for production
- [ ] Documentation updated for new architecture
- [ ] Team training completed on new shared logic

This comprehensive testing strategy ensures that the prediction logic refactoring maintains quality and consistency
while enabling the desired code sharing between jaqpotpy and jaqpotpy-inference.
