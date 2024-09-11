from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.model_task import ModelTask
from ..models.model_visibility import ModelVisibility
from ..types import UNSET, Unset

T = TypeVar("T", bound="PartiallyUpdateModelBody")


@_attrs_define
class PartiallyUpdateModelBody:
    """
    Attributes:
        name (str):
        visibility (ModelVisibility):
        task (ModelTask):
        description (Union[Unset, str]):
        tags (Union[Unset, str]):
        shared_with_organization_ids (Union[Unset, List[int]]):
    """

    name: str
    visibility: ModelVisibility
    task: ModelTask
    description: Union[Unset, str] = UNSET
    tags: Union[Unset, str] = UNSET
    shared_with_organization_ids: Union[Unset, List[int]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        visibility = self.visibility.value

        task = self.task.value

        description = self.description

        tags = self.tags

        shared_with_organization_ids: Union[Unset, List[int]] = UNSET
        if not isinstance(self.shared_with_organization_ids, Unset):
            shared_with_organization_ids = self.shared_with_organization_ids

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "visibility": visibility,
                "task": task,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if tags is not UNSET:
            field_dict["tags"] = tags
        if shared_with_organization_ids is not UNSET:
            field_dict["sharedWithOrganizationIds"] = shared_with_organization_ids

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        visibility = ModelVisibility(d.pop("visibility"))

        task = ModelTask(d.pop("task"))

        description = d.pop("description", UNSET)

        tags = d.pop("tags", UNSET)

        shared_with_organization_ids = cast(List[int], d.pop("sharedWithOrganizationIds", UNSET))

        partially_update_model_body = cls(
            name=name,
            visibility=visibility,
            task=task,
            description=description,
            tags=tags,
            shared_with_organization_ids=shared_with_organization_ids,
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
