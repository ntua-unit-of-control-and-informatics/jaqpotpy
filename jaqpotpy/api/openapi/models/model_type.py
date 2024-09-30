# coding: utf-8

"""
    Jaqpot API

    A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

    The version of the OpenAPI document: 1.0.0
    Contact: upci.ntua@gmail.com
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class ModelType(str, Enum):
    """
    ModelType
    """

    """
    allowed enum values
    """
    SKLEARN = 'SKLEARN'
    TORCH_ONNX = 'TORCH_ONNX'
    TORCHSCRIPT = 'TORCHSCRIPT'
    R_BNLEARN_DISCRETE = 'R_BNLEARN_DISCRETE'
    R_CARET = 'R_CARET'
    R_GBM = 'R_GBM'
    R_NAIVE_BAYES = 'R_NAIVE_BAYES'
    R_PBPK = 'R_PBPK'
    R_RF = 'R_RF'
    R_RPART = 'R_RPART'
    R_SVM = 'R_SVM'
    R_TREE_CLASS = 'R_TREE_CLASS'
    R_TREE_REGR = 'R_TREE_REGR'
    QSAR_TOOLBOX = 'QSAR_TOOLBOX'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of ModelType from a JSON string"""
        return cls(json.loads(json_str))


