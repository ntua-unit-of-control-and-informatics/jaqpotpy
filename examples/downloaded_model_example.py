"""
Example: Using Jaqpot Downloaded Model Functionality

This example demonstrates how to download ONNX models from the Jaqpot platform
and run predictions in your Jupyter notebook or Python environment.

The implementation now supports:
- Automatic fallback from S3 presigned URLs to database storage
- Large models stored in S3 with efficient presigned URL downloads
- Small models stored directly in the database as base64
- Preprocessor downloads with the same fallback mechanism
- Downloaded model caching for improved performance
"""

import numpy as np
from jaqpotpy import Jaqpot
from jaqpotpy.jaqpot_local import JaqpotLocalhost


def main():
    # Initialize Jaqpot client
    jaqpot = JaqpotLocalhost()

    # Login to Jaqpot platform
    jaqpot.login()

    # Download a model for offline use
    model_id = 3  # Replace with actual model ID

    print(f"Downloading model {model_id}...")
    model_data = jaqpot.download_model(model_id, cache=True)
    print("Model downloaded successfully!")

    # Prepare some sample data for prediction
    # This can be:
    # 1. Numpy array: np.array([[1.0, 2.0, 3.0]])
    # 2. Python list: [[1.0, 2.0, 3.0]]
    # 3. Dictionary: {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}

    sample_data = np.array([[1.0, 2.0, 3.0, 4.0]])  # Example input

    # Make predictions with downloaded model
    print("Making predictions with downloaded model...")
    response = jaqpot.predict_local(model_data, sample_data)

    # The response is a PredictionResponse object with the same format
    # as you would get from the Jaqpot API
    print(f"Predictions: {response.predictions}")

    # Alternative usage: You can also pass just the model_id if it's cached
    response2 = jaqpot.predict_local(model_id, sample_data)
    print(f"Cached model predictions: {response2.predictions}")

    # Check cached models
    cached_models = jaqpot.model_downloader.list_cached_models()
    print(f"Cached models: {cached_models}")

    # Clear cache if needed
    # jaqpot.model_downloader.clear_cache()


def batch_predictions_example():
    """Example of making multiple predictions efficiently with downloaded models"""
    jaqpot = JaqpotLocalhost()
    jaqpot.login()

    model_id = "your-model-id-here"

    # Download model once
    model_data = jaqpot.download_model(model_id)

    # Make multiple predictions with the same downloaded model
    test_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ]

    for i, data in enumerate(test_data):
        response = jaqpot.predict_local(model_data, [data])
        print(f"Sample {i+1} prediction: {response.predictions}")


if __name__ == "__main__":
    # Run the basic example
    main()

    # Uncomment to run batch predictions example
    # batch_predictions_example()
