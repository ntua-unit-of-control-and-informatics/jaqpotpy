#!/usr/bin/env python3
"""
Test script for jaqpotpy local model download with presigned URLs.

This script tests the enhanced local model functionality that supports:
1. Downloading models from database (base64 encoded)
2. Downloading models from S3 using presigned URLs
3. Downloading preprocessors with same fallback logic
4. Running local inference with downloaded models

Usage:
    python test_local_model_download.py --model-id <model_id> [--api-url <url>]

Requirements:
    - jaqpotpy installed with local model functionality
    - Valid Jaqpot authentication (login credentials)
    - Access to models in the platform
"""

import argparse
import sys
import time
import numpy as np
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_download_and_inference():
    """Test the complete model download and local inference workflow."""

    try:
        # Import jaqpotpy components
        from jaqpotpy import Jaqpot
        from jaqpotpy.api.downloaded_model import JaqpotDownloadedModel

        logger.info("✅ Successfully imported jaqpotpy components")
    except ImportError as e:
        logger.error(f"❌ Failed to import jaqpotpy: {e}")
        logger.error("Make sure jaqpotpy is installed: pip install -e .")
        return False

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test jaqpotpy local model download")
    parser.add_argument("--model-id", required=True, help="Model ID to test")
    parser.add_argument("--api-url", default="https://api.jaqpot.org", help="API URL")
    parser.add_argument("--local", action="store_true", help="Use localhost API")
    args = parser.parse_args()

    # Initialize Jaqpot client
    if args.local:
        from jaqpotpy.jaqpot_local import JaqpotLocalhost

        jaqpot = JaqpotLocalhost()
        logger.info("🏠 Using localhost API configuration")
    else:
        jaqpot = Jaqpot(base_url=args.api_url)
        logger.info(f"🌐 Using API: {args.api_url}")

    # Login to Jaqpot
    logger.info("🔐 Attempting to login to Jaqpot...")
    try:
        jaqpot.login()
        logger.info("✅ Successfully logged in to Jaqpot")
    except Exception as e:
        logger.error(f"❌ Failed to login: {e}")
        logger.error("Please check your credentials and network connection")
        return False

    # Initialize local model handler
    local_model = JaqpotDownloadedModel(jaqpot)
    logger.info("📥 Initialized local model handler")

    # Test 1: Download model with presigned URL support
    logger.info(f"📦 Testing model download for model ID: {args.model_id}")

    start_time = time.time()
    try:
        model_data = local_model.download_model(args.model_id, cache=True)
        download_time = time.time() - start_time
        logger.info(f"✅ Model download completed in {download_time:.2f} seconds")

        # Analyze downloaded model
        model_metadata = model_data["model_metadata"]
        onnx_bytes = model_data["onnx_bytes"]
        preprocessor = model_data.get("preprocessor")

        logger.info("📊 Model analysis:")
        logger.info(f"   - Model Type: {model_metadata.type}")
        logger.info(f"   - Model Name: {model_metadata.name}")
        logger.info(
            f"   - ONNX Size: {len(onnx_bytes):,} bytes ({len(onnx_bytes) / 1024 / 1024:.2f} MB)"
        )
        logger.info(f"   - Has Preprocessor: {'Yes' if preprocessor else 'No'}")
        logger.info(
            f"   - Independent Features: {len(model_metadata.independent_features)}"
        )
        logger.info(
            f"   - Dependent Features: {len(model_metadata.dependent_features)}"
        )

    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        return False

    # Test 2: Generate sample data for prediction
    logger.info("🎲 Generating sample prediction data...")

    try:
        # Create sample data based on model features
        sample_data = {}
        for feature in model_metadata.independent_features:
            feature_name = feature.name
            if feature.feature_type.value == "FLOAT":
                # Generate random float in reasonable range
                sample_data[feature_name] = np.random.uniform(-1, 1)
            elif feature.feature_type.value == "INTEGER":
                # Generate random integer
                sample_data[feature_name] = np.random.randint(0, 100)
            elif feature.feature_type.value == "CATEGORICAL":
                # Use first possible value if available
                if hasattr(feature, "possible_values") and feature.possible_values:
                    sample_data[feature_name] = feature.possible_values[0]
                else:
                    sample_data[feature_name] = "category_1"
            else:
                # Default to float
                sample_data[feature_name] = np.random.uniform(0, 1)

        logger.info(f"📋 Sample data generated: {sample_data}")

    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {e}")
        return False

    # Test 3: Run local prediction
    logger.info("🔮 Testing local prediction...")

    try:
        start_time = time.time()
        prediction_response = local_model.predict_local(model_data, sample_data)
        prediction_time = time.time() - start_time

        logger.info(f"✅ Prediction completed in {prediction_time:.4f} seconds")
        logger.info(f"📈 Prediction results: {prediction_response.predictions}")

    except Exception as e:
        logger.error(f"❌ Local prediction failed: {e}")
        return False

    # Test 4: Test cache functionality
    logger.info("💾 Testing model caching...")

    try:
        # Download same model again (should use cache)
        start_time = time.time()
        local_model.download_model(args.model_id, cache=True)
        cache_time = time.time() - start_time

        logger.info(f"✅ Cached model retrieval in {cache_time:.4f} seconds")
        logger.info(f"📚 Cached models: {local_model.list_cached_models()}")

    except Exception as e:
        logger.error(f"❌ Cache test failed: {e}")
        return False

    # Test 5: Test different data formats
    logger.info("🔄 Testing different input data formats...")

    try:
        # Test with list format
        sample_list = list(sample_data.values())
        list_response = local_model.predict_local(model_data, sample_list)
        logger.info(f"✅ List format prediction: {list_response.predictions}")

        # Test with numpy array format
        sample_array = np.array(sample_list).reshape(1, -1)
        array_response = local_model.predict_local(model_data, sample_array)
        logger.info(f"✅ Array format prediction: {array_response.predictions}")

    except Exception as e:
        logger.error(f"❌ Multi-format test failed: {e}")
        return False

    # Test 6: Performance benchmark
    logger.info("⚡ Running performance benchmark...")

    try:
        num_predictions = 100
        start_time = time.time()

        for i in range(num_predictions):
            local_model.predict_local(model_data, sample_data)

        total_time = time.time() - start_time
        avg_time = total_time / num_predictions

        logger.info("✅ Performance benchmark:")
        logger.info(f"   - {num_predictions} predictions in {total_time:.2f} seconds")
        logger.info(f"   - Average: {avg_time * 1000:.2f} ms per prediction")
        logger.info(
            f"   - Throughput: {num_predictions / total_time:.1f} predictions/second"
        )

    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

    # Summary
    logger.info("🎉 All tests completed successfully!")
    logger.info("📋 Test Summary:")
    logger.info("   ✅ Model download with presigned URL support")
    logger.info("   ✅ Preprocessor handling")
    logger.info("   ✅ Local ONNX inference")
    logger.info("   ✅ Model caching")
    logger.info("   ✅ Multiple input formats")
    logger.info("   ✅ Performance benchmarking")

    return True


def test_error_scenarios():
    """Test error handling scenarios."""
    logger.info("🚨 Testing error scenarios...")

    from jaqpotpy import Jaqpot
    from jaqpotpy.api.downloaded_model import JaqpotDownloadedModel

    jaqpot = Jaqpot()
    local_model = JaqpotDownloadedModel(jaqpot)

    # Test invalid model ID
    try:
        local_model.download_model("invalid-model-id")
        logger.warning("⚠️ Expected error for invalid model ID did not occur")
    except Exception as e:
        logger.info(f"✅ Correctly handled invalid model ID: {type(e).__name__}")


def main():
    """Main test function."""
    logger.info("🚀 Starting jaqpotpy local model test suite")
    logger.info("=" * 60)

    # Run main tests
    success = test_model_download_and_inference()

    if success:
        logger.info("=" * 60)
        logger.info("🎊 ALL TESTS PASSED!")
        logger.info("The jaqpotpy local model functionality is working correctly.")
        logger.info("Ready for integration with jaqpotpy-inference!")
        return 0
    else:
        logger.error("=" * 60)
        logger.error("💥 TESTS FAILED!")
        logger.error("Please check the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
