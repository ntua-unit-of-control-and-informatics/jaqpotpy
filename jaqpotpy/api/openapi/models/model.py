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
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictBytes, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional, Union
from typing_extensions import Annotated
from jaqpotpy.api.openapi.models.doa import Doa
from jaqpotpy.api.openapi.models.feature import Feature
from jaqpotpy.api.openapi.models.library import Library
from jaqpotpy.api.openapi.models.model_extra_config import ModelExtraConfig
from jaqpotpy.api.openapi.models.model_task import ModelTask
from jaqpotpy.api.openapi.models.model_type import ModelType
from jaqpotpy.api.openapi.models.model_visibility import ModelVisibility
from jaqpotpy.api.openapi.models.organization import Organization
from jaqpotpy.api.openapi.models.user import User
from typing import Optional, Set
from typing_extensions import Self

class Model(BaseModel):
    """
    Model
    """ # noqa: E501
    id: Optional[StrictInt] = None
    name: Annotated[str, Field(min_length=3, strict=True, max_length=255)]
    description: Optional[Annotated[str, Field(min_length=3, strict=True, max_length=50000)]] = None
    type: ModelType
    jaqpotpy_version: StrictStr = Field(alias="jaqpotpyVersion")
    doas: Optional[List[Doa]] = None
    libraries: List[Library]
    dependent_features: List[Feature] = Field(alias="dependentFeatures")
    independent_features: List[Feature] = Field(alias="independentFeatures")
    shared_with_organizations: Optional[List[Organization]] = Field(default=None, alias="sharedWithOrganizations")
    visibility: ModelVisibility
    task: ModelTask
    raw_model: Union[StrictBytes, StrictStr] = Field(description="A base64 representation of the raw model.", alias="rawModel")
    creator: Optional[User] = None
    can_edit: Optional[StrictBool] = Field(default=None, description="If the current user can edit the model", alias="canEdit")
    is_admin: Optional[StrictBool] = Field(default=None, alias="isAdmin")
    tags: Optional[Annotated[str, Field(strict=True, max_length=1000)]] = None
    legacy_prediction_service: Optional[StrictStr] = Field(default=None, alias="legacyPredictionService")
    extra_config: Optional[ModelExtraConfig] = Field(default=None, alias="extraConfig")
    created_at: Optional[datetime] = Field(default=None, description="The date and time when the feature was created.", alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, description="The date and time when the feature was last updated.", alias="updatedAt")
    __properties: ClassVar[List[str]] = ["id", "name", "description", "type", "jaqpotpyVersion", "doas", "libraries", "dependentFeatures", "independentFeatures", "sharedWithOrganizations", "visibility", "task", "rawModel", "creator", "canEdit", "isAdmin", "tags", "legacyPredictionService", "extraConfig", "createdAt", "updatedAt"]

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
        """Create an instance of Model from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in doas (list)
        _items = []
        if self.doas:
            for _item_doas in self.doas:
                if _item_doas:
                    _items.append(_item_doas.to_dict())
            _dict['doas'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in libraries (list)
        _items = []
        if self.libraries:
            for _item_libraries in self.libraries:
                if _item_libraries:
                    _items.append(_item_libraries.to_dict())
            _dict['libraries'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in dependent_features (list)
        _items = []
        if self.dependent_features:
            for _item_dependent_features in self.dependent_features:
                if _item_dependent_features:
                    _items.append(_item_dependent_features.to_dict())
            _dict['dependentFeatures'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in independent_features (list)
        _items = []
        if self.independent_features:
            for _item_independent_features in self.independent_features:
                if _item_independent_features:
                    _items.append(_item_independent_features.to_dict())
            _dict['independentFeatures'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in shared_with_organizations (list)
        _items = []
        if self.shared_with_organizations:
            for _item_shared_with_organizations in self.shared_with_organizations:
                if _item_shared_with_organizations:
                    _items.append(_item_shared_with_organizations.to_dict())
            _dict['sharedWithOrganizations'] = _items
        # override the default output from pydantic by calling `to_dict()` of creator
        if self.creator:
            _dict['creator'] = self.creator.to_dict()
        # override the default output from pydantic by calling `to_dict()` of extra_config
        if self.extra_config:
            _dict['extraConfig'] = self.extra_config.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Model from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "id": obj.get("id"),
            "name": obj.get("name"),
            "description": obj.get("description"),
            "type": obj.get("type"),
            "jaqpotpyVersion": obj.get("jaqpotpyVersion"),
            "doas": [Doa.from_dict(_item) for _item in obj["doas"]] if obj.get("doas") is not None else None,
            "libraries": [Library.from_dict(_item) for _item in obj["libraries"]] if obj.get("libraries") is not None else None,
            "dependentFeatures": [Feature.from_dict(_item) for _item in obj["dependentFeatures"]] if obj.get("dependentFeatures") is not None else None,
            "independentFeatures": [Feature.from_dict(_item) for _item in obj["independentFeatures"]] if obj.get("independentFeatures") is not None else None,
            "sharedWithOrganizations": [Organization.from_dict(_item) for _item in obj["sharedWithOrganizations"]] if obj.get("sharedWithOrganizations") is not None else None,
            "visibility": obj.get("visibility"),
            "task": obj.get("task"),
            "rawModel": obj.get("rawModel"),
            "creator": User.from_dict(obj["creator"]) if obj.get("creator") is not None else None,
            "canEdit": obj.get("canEdit"),
            "isAdmin": obj.get("isAdmin"),
            "tags": obj.get("tags"),
            "legacyPredictionService": obj.get("legacyPredictionService"),
            "extraConfig": ModelExtraConfig.from_dict(obj["extraConfig"]) if obj.get("extraConfig") is not None else None,
            "createdAt": obj.get("createdAt"),
            "updatedAt": obj.get("updatedAt")
        })
        return _obj


