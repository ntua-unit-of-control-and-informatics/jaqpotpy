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

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, ClassVar, Dict, List, Optional
from typing_extensions import Annotated
from jaqpotpy.api.openapi.models.transformer import Transformer
from typing import Optional, Set
from typing_extensions import Self

class ModelExtraConfig(BaseModel):
    """
    A JSON object containing extra configuration for the model
    """ # noqa: E501
    torch_config: Optional[Dict[str, Any]] = Field(default=None, alias="torchConfig")
    preprocessors: Optional[Annotated[List[Transformer], Field(max_length=50)]] = None
    featurizers: Optional[Annotated[List[Transformer], Field(max_length=50)]] = None
    __properties: ClassVar[List[str]] = ["torchConfig", "preprocessors", "featurizers"]

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
        """Create an instance of ModelExtraConfig from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in preprocessors (list)
        _items = []
        if self.preprocessors:
            for _item_preprocessors in self.preprocessors:
                if _item_preprocessors:
                    _items.append(_item_preprocessors.to_dict())
            _dict['preprocessors'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in featurizers (list)
        _items = []
        if self.featurizers:
            for _item_featurizers in self.featurizers:
                if _item_featurizers:
                    _items.append(_item_featurizers.to_dict())
            _dict['featurizers'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of ModelExtraConfig from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "torchConfig": obj.get("torchConfig"),
            "preprocessors": [Transformer.from_dict(_item) for _item in obj["preprocessors"]] if obj.get("preprocessors") is not None else None,
            "featurizers": [Transformer.from_dict(_item) for _item in obj["featurizers"]] if obj.get("featurizers") is not None else None
        })
        return _obj


