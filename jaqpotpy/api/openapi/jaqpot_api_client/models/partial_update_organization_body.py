from typing import Any, Dict, Type, TypeVar, Union

from attrs import define as _attrs_define

from ..models.organization_visibility import OrganizationVisibility
from ..types import UNSET, Unset

T = TypeVar("T", bound="PartialUpdateOrganizationBody")


@_attrs_define
class PartialUpdateOrganizationBody:
    """
    Attributes:
        name (str):
        contact_email (str):
        visibility (OrganizationVisibility):
        description (Union[Unset, str]):
    """

    name: str
    contact_email: str
    visibility: OrganizationVisibility
    description: Union[Unset, str] = UNSET

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        contact_email = self.contact_email

        visibility = self.visibility.value

        description = self.description

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {
                "name": name,
                "contactEmail": contact_email,
                "visibility": visibility,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        contact_email = d.pop("contactEmail")

        visibility = OrganizationVisibility(d.pop("visibility"))

        description = d.pop("description", UNSET)

        partial_update_organization_body = cls(
            name=name,
            contact_email=contact_email,
            visibility=visibility,
            description=description,
        )

        return partial_update_organization_body
