from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.model_visibility import ModelVisibility
from ..types import UNSET, Unset

T = TypeVar("T", bound="PartiallyUpdateModelBody")


@_attrs_define
class PartiallyUpdateModelBody:
    """
    Attributes:
        name (str):
        visibility (ModelVisibility):
        description (Union[Unset, str]):
        organization_ids (Union[Unset, List[int]]):
    """

    name: str
    visibility: ModelVisibility
    description: Union[Unset, str] = UNSET
    organization_ids: Union[Unset, List[int]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        visibility = self.visibility.value

        description = self.description

        organization_ids: Union[Unset, List[int]] = UNSET
        if not isinstance(self.organization_ids, Unset):
            organization_ids = self.organization_ids

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "visibility": visibility,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if organization_ids is not UNSET:
            field_dict["organizationIds"] = organization_ids

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        visibility = ModelVisibility(d.pop("visibility"))

        description = d.pop("description", UNSET)

        organization_ids = cast(List[int], d.pop("organizationIds", UNSET))

        partially_update_model_body = cls(
            name=name,
            visibility=visibility,
            description=description,
            organization_ids=organization_ids,
        )

        partially_update_model_body.additional_properties = d
        return partially_update_model_body

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
