"""
Jaqpotpy Inference Package

This package provides shared prediction logic for both local development
and production inference, eliminating code duplication between jaqpotpy
local models and jaqpotpy-inference service.

Components:
- core: Core prediction algorithms and data processing
- handlers: Model-type-specific prediction handlers
- utils: Utility functions for image processing, tensors, etc.
- service: Unified prediction service for both local and production modes
"""

from .service import PredictionService

__all__ = ["PredictionService"]
