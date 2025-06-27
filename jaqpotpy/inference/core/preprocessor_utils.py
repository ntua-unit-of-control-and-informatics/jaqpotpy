"""
Preprocessor recreation utilities for ONNX models.

This module contains utilities for recreating sklearn preprocessors and
jaqpotpy transformers from configuration data, ensuring consistent
preprocessing between local and production inference.
"""

import numpy as np
from typing import Dict, Any, Union
from jaqpotpy.transformers import LogTransformer


def recreate_preprocessor(
    preprocessor_name: str, preprocessor_config: Dict[str, Any]
) -> Union[object, LogTransformer]:
    """
    Recreate a preprocessor instance from name and configuration.

    Args:
        preprocessor_name (str): Name of the preprocessor class
        preprocessor_config (dict): Configuration dictionary containing
                                   preprocessor parameters and state

    Returns:
        Recreated preprocessor instance

    Raises:
        ImportError: If the preprocessor class cannot be imported
        AttributeError: If the preprocessor class doesn't exist
    """
    if preprocessor_name == "LogTransformer":
        preprocessor = LogTransformer()
    else:
        # Import sklearn preprocessor dynamically
        preprocessor_class = getattr(
            __import__("sklearn.preprocessing", fromlist=[preprocessor_name]),
            preprocessor_name,
        )
        preprocessor = preprocessor_class()

        # Set all attributes from config
        for attr, value in preprocessor_config.items():
            if attr != "class":  # skip the class attribute
                if isinstance(value, list):
                    value = np.array(value)
                setattr(preprocessor, attr, value)

    return preprocessor


def recreate_featurizer(
    featurizer_name: str, featurizer_config: Dict[str, Any]
) -> object:
    """
    Recreate a molecular featurizer instance from name and configuration.

    Args:
        featurizer_name (str): Name of the featurizer class
        featurizer_config (dict): Configuration dictionary containing
                                 featurizer parameters and state

    Returns:
        Recreated featurizer instance

    Raises:
        ImportError: If the featurizer class cannot be imported
        AttributeError: If the featurizer class doesn't exist
    """
    featurizer_class = getattr(
        __import__("jaqpotpy.descriptors.molecular", fromlist=[featurizer_name]),
        featurizer_name,
    )
    featurizer = featurizer_class()

    # Set all attributes from config
    for attr, value in featurizer_config.items():
        if attr != "class":  # skip the class attribute
            setattr(featurizer, attr, value)

    return featurizer
