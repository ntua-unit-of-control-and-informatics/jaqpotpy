import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.model_type import ModelType
from ..models.model_visibility import ModelVisibility
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.organization_summary import OrganizationSummary
    from ..models.user import User


T = TypeVar("T", bound="ModelSummary")


@_attrs_define
class ModelSummary:
    """
    Attributes:
        id (int):
        name (str):  Example: My Model.
        visibility (ModelVisibility):
        type (ModelType):
        shared_with_organizations (List['OrganizationSummary']):
        created_at (datetime.datetime): The date and time when the feature was created. Example: 2023-01-01T12:00:00Z.
        description (Union[Unset, str]):  Example: A description of your model.
        creator (Union[Unset, User]):
        dependent_features_length (Union[Unset, int]):
        independent_features_length (Union[Unset, int]):
        updated_at (Union[Unset, str]): The date and time when the feature was last updated. Example:
            2023-01-01T12:00:00Z.
    """

    id: int
    name: str
    visibility: ModelVisibility
    type: ModelType
    shared_with_organizations: List["OrganizationSummary"]
    created_at: datetime.datetime
    description: Union[Unset, str] = UNSET
    creator: Union[Unset, "User"] = UNSET
    dependent_features_length: Union[Unset, int] = UNSET
    independent_features_length: Union[Unset, int] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id

        name = self.name

        visibility = self.visibility.value

        type = self.type.value

        shared_with_organizations = []
        for shared_with_organizations_item_data in self.shared_with_organizations:
            shared_with_organizations_item = shared_with_organizations_item_data.to_dict()
            shared_with_organizations.append(shared_with_organizations_item)

        created_at = self.created_at.isoformat()

        description = self.description

        creator: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.creator, Unset):
            creator = self.creator.to_dict()

        dependent_features_length = self.dependent_features_length

        independent_features_length = self.independent_features_length

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "visibility": visibility,
                "type": type,
                "sharedWithOrganizations": shared_with_organizations,
                "createdAt": created_at,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if creator is not UNSET:
            field_dict["creator"] = creator
        if dependent_features_length is not UNSET:
            field_dict["dependentFeaturesLength"] = dependent_features_length
        if independent_features_length is not UNSET:
            field_dict["independentFeaturesLength"] = independent_features_length
        if updated_at is not UNSET:
            field_dict["updatedAt"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.organization_summary import OrganizationSummary
        from ..models.user import User

        d = src_dict.copy()
        id = d.pop("id")

        name = d.pop("name")

        visibility = ModelVisibility(d.pop("visibility"))

        type = ModelType(d.pop("type"))

        shared_with_organizations = []
        _shared_with_organizations = d.pop("sharedWithOrganizations")
        for shared_with_organizations_item_data in _shared_with_organizations:
            shared_with_organizations_item = OrganizationSummary.from_dict(shared_with_organizations_item_data)

            shared_with_organizations.append(shared_with_organizations_item)

        created_at = isoparse(d.pop("createdAt"))

        description = d.pop("description", UNSET)

        _creator = d.pop("creator", UNSET)
        creator: Union[Unset, User]
        if isinstance(_creator, Unset):
            creator = UNSET
        else:
            creator = User.from_dict(_creator)

        dependent_features_length = d.pop("dependentFeaturesLength", UNSET)

        independent_features_length = d.pop("independentFeaturesLength", UNSET)

        updated_at = d.pop("updatedAt", UNSET)

        model_summary = cls(
            id=id,
            name=name,
            visibility=visibility,
            type=type,
            shared_with_organizations=shared_with_organizations,
            created_at=created_at,
            description=description,
            creator=creator,
            dependent_features_length=dependent_features_length,
            independent_features_length=independent_features_length,
            updated_at=updated_at,
        )

        model_summary.additional_properties = d
        return model_summary

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
