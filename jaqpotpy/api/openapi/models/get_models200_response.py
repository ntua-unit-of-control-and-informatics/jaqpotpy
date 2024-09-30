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

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Any, ClassVar, Dict, List, Optional
from jaqpotpy.api.openapi.models.model_summary import ModelSummary
from typing import Optional, Set
from typing_extensions import Self

class GetModels200Response(BaseModel):
    """
    GetModels200Response
    """ # noqa: E501
    content: Optional[List[ModelSummary]] = None
    total_elements: Optional[StrictInt] = Field(default=None, alias="totalElements")
    total_pages: Optional[StrictInt] = Field(default=None, alias="totalPages")
    page_size: Optional[StrictInt] = Field(default=None, alias="pageSize")
    page_number: Optional[StrictInt] = Field(default=None, alias="pageNumber")
    __properties: ClassVar[List[str]] = ["content", "totalElements", "totalPages", "pageSize", "pageNumber"]

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
        """Create an instance of GetModels200Response from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in content (list)
        _items = []
        if self.content:
            for _item_content in self.content:
                if _item_content:
                    _items.append(_item_content.to_dict())
            _dict['content'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of GetModels200Response from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "content": [ModelSummary.from_dict(_item) for _item in obj["content"]] if obj.get("content") is not None else None,
            "totalElements": obj.get("totalElements"),
            "totalPages": obj.get("totalPages"),
            "pageSize": obj.get("pageSize"),
            "pageNumber": obj.get("pageNumber")
        })
        return _obj

