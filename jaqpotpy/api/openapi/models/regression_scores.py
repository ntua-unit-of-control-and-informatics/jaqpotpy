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
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from typing import Any, ClassVar, Dict, List, Optional, Union
from typing import Optional, Set
from typing_extensions import Self


class RegressionScores(BaseModel):
    """
    RegressionScores
    """  # noqa: E501

    r2: Optional[Union[StrictFloat, StrictInt]] = None
    mae: Optional[Union[StrictFloat, StrictInt]] = None
    rmse: Optional[Union[StrictFloat, StrictInt]] = None
    r_squared_diff_r_zero: Optional[Union[StrictFloat, StrictInt]] = Field(
        default=None, alias="rSquaredDiffRZero"
    )
    r_squared_diff_r_zero_hat: Optional[Union[StrictFloat, StrictInt]] = Field(
        default=None, alias="rSquaredDiffRZeroHat"
    )
    abs_diff_r_zero_hat: Optional[Union[StrictFloat, StrictInt]] = Field(
        default=None, alias="absDiffRZeroHat"
    )
    k: Optional[Union[StrictFloat, StrictInt]] = None
    k_hat: Optional[Union[StrictFloat, StrictInt]] = Field(default=None, alias="kHat")
    __properties: ClassVar[List[str]] = [
        "r2",
        "mae",
        "rmse",
        "rSquaredDiffRZero",
        "rSquaredDiffRZeroHat",
        "absDiffRZeroHat",
        "k",
        "kHat",
    ]

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        # TODO: pydantic v2: use .model_dump_json(by_alias=True, exclude_unset=True) instead
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of RegressionScores from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of RegressionScores from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate(
            {
                "r2": obj.get("r2"),
                "mae": obj.get("mae"),
                "rmse": obj.get("rmse"),
                "rSquaredDiffRZero": obj.get("rSquaredDiffRZero"),
                "rSquaredDiffRZeroHat": obj.get("rSquaredDiffRZeroHat"),
                "absDiffRZeroHat": obj.get("absDiffRZeroHat"),
                "k": obj.get("k"),
                "kHat": obj.get("kHat"),
            }
        )
        return _obj
