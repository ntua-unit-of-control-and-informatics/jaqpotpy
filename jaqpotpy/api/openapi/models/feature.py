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
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List, Optional
from typing_extensions import Annotated
from jaqpotpy.api.openapi.models.feature_possible_value import FeaturePossibleValue
from jaqpotpy.api.openapi.models.feature_type import FeatureType
from typing import Optional, Set
from typing_extensions import Self

class Feature(BaseModel):
    """
    Feature
    """ # noqa: E501
    id: Optional[StrictInt] = None
    key: Annotated[str, Field(strict=True)] = Field(description="A key that must start with a letter, followed by any combination of letters, digits, hyphens, or underscores. For example, 'abc123', 'abc-test', or 'Abc_test'. It cannot start with a digit.")
    name: Annotated[str, Field(strict=True, max_length=255)] = Field(description="A name for the feature that will appear on top of the form field")
    units: Optional[Annotated[str, Field(strict=True, max_length=255)]] = Field(default=None, description="A name for the feature that will appear on top of the form field")
    description: Optional[Annotated[str, Field(strict=True, max_length=2000)]] = None
    feature_type: FeatureType = Field(alias="featureType")
    feature_dependency: Optional[StrictStr] = Field(default=None, alias="featureDependency")
    visible: Optional[StrictBool] = None
    possible_values: Optional[List[FeaturePossibleValue]] = Field(default=None, alias="possibleValues")
    created_at: Optional[datetime] = Field(default=None, description="The date and time when the feature was created.", alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, description="The date and time when the feature was last updated.", alias="updatedAt")
    __properties: ClassVar[List[str]] = ["id", "key", "name", "units", "description", "featureType", "featureDependency", "visible", "possibleValues", "createdAt", "updatedAt"]

    @field_validator('key')
    def key_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z][a-zA-Z0-9_-]*$/")
        return value

    @field_validator('feature_dependency')
    def feature_dependency_validate_enum(cls, value):
        """Validates the enum"""
        if value is None:
            return value

        if value not in set(['DEPENDENT', 'INDEPENDENT']):
            raise ValueError("must be one of enum values ('DEPENDENT', 'INDEPENDENT')")
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
        """Create an instance of Feature from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in possible_values (list)
        _items = []
        if self.possible_values:
            for _item_possible_values in self.possible_values:
                if _item_possible_values:
                    _items.append(_item_possible_values.to_dict())
            _dict['possibleValues'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Feature from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "id": obj.get("id"),
            "key": obj.get("key"),
            "name": obj.get("name"),
            "units": obj.get("units"),
            "description": obj.get("description"),
            "featureType": obj.get("featureType"),
            "featureDependency": obj.get("featureDependency"),
            "visible": obj.get("visible"),
            "possibleValues": [FeaturePossibleValue.from_dict(_item) for _item in obj["possibleValues"]] if obj.get("possibleValues") is not None else None,
            "createdAt": obj.get("createdAt"),
            "updatedAt": obj.get("updatedAt")
        })
        return _obj


