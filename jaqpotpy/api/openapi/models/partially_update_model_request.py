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
from typing_extensions import Annotated
from jaqpotpy.api.openapi.models.model_task import ModelTask
from jaqpotpy.api.openapi.models.model_visibility import ModelVisibility
from typing import Optional, Set
from typing_extensions import Self


class PartiallyUpdateModelRequest(BaseModel):
    """
    PartiallyUpdateModelRequest
    """  # noqa: E501

    name: Annotated[str, Field(min_length=3, strict=True, max_length=255)]
    description: Optional[
        Annotated[str, Field(min_length=3, strict=True, max_length=50000)]
    ] = None
    visibility: ModelVisibility
    task: ModelTask
    tags: Optional[Annotated[str, Field(strict=True, max_length=1000)]] = None
    shared_with_organization_ids: Optional[List[StrictInt]] = Field(
        default=None, alias="sharedWithOrganizationIds"
    )
    __properties: ClassVar[List[str]] = [
        "name",
        "description",
        "visibility",
        "task",
        "tags",
        "sharedWithOrganizationIds",
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
        """Create an instance of PartiallyUpdateModelRequest from a JSON string"""
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
        """Create an instance of PartiallyUpdateModelRequest from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate(
            {
                "name": obj.get("name"),
                "description": obj.get("description"),
                "visibility": obj.get("visibility"),
                "task": obj.get("task"),
                "tags": obj.get("tags"),
                "sharedWithOrganizationIds": obj.get("sharedWithOrganizationIds"),
            }
        )
        return _obj
