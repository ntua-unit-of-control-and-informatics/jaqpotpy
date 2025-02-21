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

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List, Optional
from typing_extensions import Annotated
from jaqpotpy.api.openapi.models.dataset_type import DatasetType
from typing import Optional, Set
from typing_extensions import Self

class Dataset(BaseModel):
    """
    Dataset
    """ # noqa: E501
    id: Optional[StrictInt] = None
    name: Optional[StrictStr] = None
    type: DatasetType
    entry_type: StrictStr = Field(alias="entryType")
    input: Annotated[List[Any], Field(max_length=100)]
    result: Optional[List[Any]] = None
    status: Optional[StrictStr] = None
    failure_reason: Optional[StrictStr] = Field(default=None, alias="failureReason")
    user_id: Optional[StrictStr] = Field(default=None, alias="userId")
    model_id: Optional[StrictInt] = Field(default=None, alias="modelId")
    model_name: Optional[StrictStr] = Field(default=None, alias="modelName")
    executed_at: Optional[datetime] = Field(default=None, alias="executedAt")
    execution_finished_at: Optional[datetime] = Field(default=None, alias="executionFinishedAt")
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, alias="updatedAt")
    __properties: ClassVar[List[str]] = ["id", "name", "type", "entryType", "input", "result", "status", "failureReason", "userId", "modelId", "modelName", "executedAt", "executionFinishedAt", "createdAt", "updatedAt"]

    @field_validator('entry_type')
    def entry_type_validate_enum(cls, value):
        """Validates the enum"""
        if value not in set(['ARRAY']):
            raise ValueError("must be one of enum values ('ARRAY')")
        return value

    @field_validator('status')
    def status_validate_enum(cls, value):
        """Validates the enum"""
        if value is None:
            return value

        if value not in set(['CREATED', 'EXECUTING', 'FAILURE', 'SUCCESS']):
            raise ValueError("must be one of enum values ('CREATED', 'EXECUTING', 'FAILURE', 'SUCCESS')")
        return value

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
        """Create an instance of Dataset from a JSON string"""
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
        """Create an instance of Dataset from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "id": obj.get("id"),
            "name": obj.get("name"),
            "type": obj.get("type"),
            "entryType": obj.get("entryType"),
            "input": obj.get("input"),
            "result": obj.get("result"),
            "status": obj.get("status"),
            "failureReason": obj.get("failureReason"),
            "userId": obj.get("userId"),
            "modelId": obj.get("modelId"),
            "modelName": obj.get("modelName"),
            "executedAt": obj.get("executedAt"),
            "executionFinishedAt": obj.get("executionFinishedAt"),
            "createdAt": obj.get("createdAt"),
            "updatedAt": obj.get("updatedAt")
        })
        return _obj


