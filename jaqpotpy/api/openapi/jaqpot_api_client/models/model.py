import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.model_task import ModelTask
from ..models.model_type import ModelType
from ..models.model_visibility import ModelVisibility
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.feature import Feature
    from ..models.library import Library
    from ..models.model_extra_config import ModelExtraConfig
    from ..models.model_meta import ModelMeta
    from ..models.organization import Organization
    from ..models.user import User


T = TypeVar("T", bound="Model")


@_attrs_define
class Model:
    """
    Attributes:
        name (str):  Example: My Model.
        type (ModelType):
        jaqpotpy_version (str):  Example: 1.0.0.
        libraries (List['Library']):
        dependent_features (List['Feature']):
        independent_features (List['Feature']):
        visibility (ModelVisibility):
        task (ModelTask):
        actual_model (str): A base64 representation of the actual model.
        id (Union[Unset, int]):
        meta (Union[Unset, ModelMeta]): A JSON object containing meta information.
        description (Union[Unset, str]):  Example: A description of your model.
        shared_with_organizations (Union[Unset, List['Organization']]):
        creator (Union[Unset, User]):
        can_edit (Union[Unset, bool]): If the current user can edit the model
        is_admin (Union[Unset, bool]):
        tags (Union[Unset, str]):
        legacy_prediction_service (Union[Unset, str]):
        extra_config (Union[Unset, ModelExtraConfig]): A JSON object containing extra configuration for the model
        created_at (Union[Unset, datetime.datetime]): The date and time when the feature was created. Example:
            2023-01-01T12:00:00Z.
        updated_at (Union[Unset, str]): The date and time when the feature was last updated. Example:
            2023-01-01T12:00:00Z.
    """

    name: str
    type: ModelType
    jaqpotpy_version: str
    libraries: List["Library"]
    dependent_features: List["Feature"]
    independent_features: List["Feature"]
    visibility: ModelVisibility
    task: ModelTask
    actual_model: str
    id: Union[Unset, int] = UNSET
    meta: Union[Unset, "ModelMeta"] = UNSET
    description: Union[Unset, str] = UNSET
    shared_with_organizations: Union[Unset, List["Organization"]] = UNSET
    creator: Union[Unset, "User"] = UNSET
    can_edit: Union[Unset, bool] = UNSET
    is_admin: Union[Unset, bool] = UNSET
    tags: Union[Unset, str] = UNSET
    legacy_prediction_service: Union[Unset, str] = UNSET
    extra_config: Union[Unset, "ModelExtraConfig"] = UNSET
    created_at: Union[Unset, datetime.datetime] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        type = self.type.value

        jaqpotpy_version = self.jaqpotpy_version

        libraries = []
        for libraries_item_data in self.libraries:
            libraries_item = libraries_item_data.to_dict()
            libraries.append(libraries_item)

        dependent_features = []
        for dependent_features_item_data in self.dependent_features:
            dependent_features_item = dependent_features_item_data.to_dict()
            dependent_features.append(dependent_features_item)

        independent_features = []
        for independent_features_item_data in self.independent_features:
            independent_features_item = independent_features_item_data.to_dict()
            independent_features.append(independent_features_item)

        visibility = self.visibility.value

        task = self.task.value

        actual_model = self.actual_model

        id = self.id

        meta: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.meta, Unset):
            meta = self.meta.to_dict()

        description = self.description

        shared_with_organizations: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.shared_with_organizations, Unset):
            shared_with_organizations = []
            for shared_with_organizations_item_data in self.shared_with_organizations:
                shared_with_organizations_item = shared_with_organizations_item_data.to_dict()
                shared_with_organizations.append(shared_with_organizations_item)

        creator: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.creator, Unset):
            creator = self.creator.to_dict()

        can_edit = self.can_edit

        is_admin = self.is_admin

        tags = self.tags

        legacy_prediction_service = self.legacy_prediction_service

        extra_config: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.extra_config, Unset):
            extra_config = self.extra_config.to_dict()

        created_at: Union[Unset, str] = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "type": type,
                "jaqpotpyVersion": jaqpotpy_version,
                "libraries": libraries,
                "dependentFeatures": dependent_features,
                "independentFeatures": independent_features,
                "visibility": visibility,
                "task": task,
                "actualModel": actual_model,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if meta is not UNSET:
            field_dict["meta"] = meta
        if description is not UNSET:
            field_dict["description"] = description
        if shared_with_organizations is not UNSET:
            field_dict["sharedWithOrganizations"] = shared_with_organizations
        if creator is not UNSET:
            field_dict["creator"] = creator
        if can_edit is not UNSET:
            field_dict["canEdit"] = can_edit
        if is_admin is not UNSET:
            field_dict["isAdmin"] = is_admin
        if tags is not UNSET:
            field_dict["tags"] = tags
        if legacy_prediction_service is not UNSET:
            field_dict["legacyPredictionService"] = legacy_prediction_service
        if extra_config is not UNSET:
            field_dict["extraConfig"] = extra_config
        if created_at is not UNSET:
            field_dict["createdAt"] = created_at
        if updated_at is not UNSET:
            field_dict["updatedAt"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.feature import Feature
        from ..models.library import Library
        from ..models.model_extra_config import ModelExtraConfig
        from ..models.model_meta import ModelMeta
        from ..models.organization import Organization
        from ..models.user import User

        d = src_dict.copy()
        name = d.pop("name")

        type = ModelType(d.pop("type"))

        jaqpotpy_version = d.pop("jaqpotpyVersion")

        libraries = []
        _libraries = d.pop("libraries")
        for libraries_item_data in _libraries:
            libraries_item = Library.from_dict(libraries_item_data)

            libraries.append(libraries_item)

        dependent_features = []
        _dependent_features = d.pop("dependentFeatures")
        for dependent_features_item_data in _dependent_features:
            dependent_features_item = Feature.from_dict(dependent_features_item_data)

            dependent_features.append(dependent_features_item)

        independent_features = []
        _independent_features = d.pop("independentFeatures")
        for independent_features_item_data in _independent_features:
            independent_features_item = Feature.from_dict(independent_features_item_data)

            independent_features.append(independent_features_item)

        visibility = ModelVisibility(d.pop("visibility"))

        task = ModelTask(d.pop("task"))

        actual_model = d.pop("actualModel")

        id = d.pop("id", UNSET)

        _meta = d.pop("meta", UNSET)
        meta: Union[Unset, ModelMeta]
        if isinstance(_meta, Unset):
            meta = UNSET
        else:
            meta = ModelMeta.from_dict(_meta)

        description = d.pop("description", UNSET)

        shared_with_organizations = []
        _shared_with_organizations = d.pop("sharedWithOrganizations", UNSET)
        for shared_with_organizations_item_data in _shared_with_organizations or []:
            shared_with_organizations_item = Organization.from_dict(shared_with_organizations_item_data)

            shared_with_organizations.append(shared_with_organizations_item)

        _creator = d.pop("creator", UNSET)
        creator: Union[Unset, User]
        if isinstance(_creator, Unset):
            creator = UNSET
        else:
            creator = User.from_dict(_creator)

        can_edit = d.pop("canEdit", UNSET)

        is_admin = d.pop("isAdmin", UNSET)

        tags = d.pop("tags", UNSET)

        legacy_prediction_service = d.pop("legacyPredictionService", UNSET)

        _extra_config = d.pop("extraConfig", UNSET)
        extra_config: Union[Unset, ModelExtraConfig]
        if isinstance(_extra_config, Unset):
            extra_config = UNSET
        else:
            extra_config = ModelExtraConfig.from_dict(_extra_config)

        _created_at = d.pop("createdAt", UNSET)
        created_at: Union[Unset, datetime.datetime]
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = isoparse(_created_at)

        updated_at = d.pop("updatedAt", UNSET)

        model = cls(
            name=name,
            type=type,
            jaqpotpy_version=jaqpotpy_version,
            libraries=libraries,
            dependent_features=dependent_features,
            independent_features=independent_features,
            visibility=visibility,
            task=task,
            actual_model=actual_model,
            id=id,
            meta=meta,
            description=description,
            shared_with_organizations=shared_with_organizations,
            creator=creator,
            can_edit=can_edit,
            is_admin=is_admin,
            tags=tags,
            legacy_prediction_service=legacy_prediction_service,
            extra_config=extra_config,
            created_at=created_at,
            updated_at=updated_at,
        )

        model.additional_properties = d
        return model

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
