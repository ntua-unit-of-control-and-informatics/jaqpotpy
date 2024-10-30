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

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional, Union
from typing_extensions import Annotated
from typing import Optional, Set
from typing_extensions import Self

class MulticlassClassificationScores(BaseModel):
    """
    MulticlassClassificationScores
    """ # noqa: E501
    y_name: StrictStr = Field(alias="yName")
    accuracy: Optional[Union[StrictFloat, StrictInt]] = None
    balanced_accuracy: Optional[Union[StrictFloat, StrictInt]] = Field(default=None, alias="balancedAccuracy")
    precision: Optional[Annotated[List[Union[StrictFloat, StrictInt]], Field(max_length=1000)]] = None
    recall: Optional[Annotated[List[Union[StrictFloat, StrictInt]], Field(max_length=1000)]] = None
    f1_score: Optional[Annotated[List[Union[StrictFloat, StrictInt]], Field(max_length=1000)]] = Field(default=None, alias="f1Score")
    jaccard: Optional[Annotated[List[Union[StrictFloat, StrictInt]], Field(max_length=1000)]] = None
    matthews_corr_coef: Optional[Union[StrictFloat, StrictInt]] = Field(default=None, alias="matthewsCorrCoef")
    confusion_matrix: Optional[Annotated[List[Annotated[List[Annotated[List[Union[StrictFloat, StrictInt]], Field(max_length=2)]], Field(max_length=2)]], Field(max_length=100)]] = Field(default=None, alias="confusionMatrix")
    __properties: ClassVar[List[str]] = ["yName", "accuracy", "balancedAccuracy", "precision", "recall", "f1Score", "jaccard", "matthewsCorrCoef", "confusionMatrix"]

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
        """Create an instance of MulticlassClassificationScores from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of MulticlassClassificationScores from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "yName": obj.get("yName"),
            "accuracy": obj.get("accuracy"),
            "balancedAccuracy": obj.get("balancedAccuracy"),
            "precision": obj.get("precision"),
            "recall": obj.get("recall"),
            "f1Score": obj.get("f1Score"),
            "jaccard": obj.get("jaccard"),
            "matthewsCorrCoef": obj.get("matthewsCorrCoef"),
            "confusionMatrix": obj.get("confusionMatrix")
        })
        return _obj


