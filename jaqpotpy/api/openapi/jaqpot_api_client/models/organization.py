from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.organization_visibility import OrganizationVisibility
from ..types import UNSET, Unset

T = TypeVar("T", bound="Organization")


@_attrs_define
class Organization:
    """
    Attributes:
        name (str):  Example: my-awesome-org.
        visibility (OrganizationVisibility):
        contact_email (str):  Example: contact@my-awesome-org.com.
        id (Union[Unset, int]):
        creator_id (Union[Unset, str]):
        description (Union[Unset, str]):  Example: An awesome organization for managing models..
        user_ids (Union[Unset, List[str]]):
        contact_phone (Union[Unset, str]):  Example: +1234567890.
        website (Union[Unset, str]):  Example: http://www.my-awesome-org.com.
        address (Union[Unset, str]):  Example: 123 Organization St., City, Country.
        can_edit (Union[Unset, bool]): If the current user can edit the organization
        is_creator (Union[Unset, bool]): If the current user is the creator of the organization
        created_at (Union[Unset, str]):
        updated_at (Union[Unset, str]):
    """

    name: str
    visibility: OrganizationVisibility
    contact_email: str
    id: Union[Unset, int] = UNSET
    creator_id: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    user_ids: Union[Unset, List[str]] = UNSET
    contact_phone: Union[Unset, str] = UNSET
    website: Union[Unset, str] = UNSET
    address: Union[Unset, str] = UNSET
    can_edit: Union[Unset, bool] = UNSET
    is_creator: Union[Unset, bool] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        visibility = self.visibility.value

        contact_email = self.contact_email

        id = self.id

        creator_id = self.creator_id

        description = self.description

        user_ids: Union[Unset, List[str]] = UNSET
        if not isinstance(self.user_ids, Unset):
            user_ids = self.user_ids

        contact_phone = self.contact_phone

        website = self.website

        address = self.address

        can_edit = self.can_edit

        is_creator = self.is_creator

        created_at = self.created_at

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "visibility": visibility,
                "contactEmail": contact_email,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if creator_id is not UNSET:
            field_dict["creatorId"] = creator_id
        if description is not UNSET:
            field_dict["description"] = description
        if user_ids is not UNSET:
            field_dict["userIds"] = user_ids
        if contact_phone is not UNSET:
            field_dict["contactPhone"] = contact_phone
        if website is not UNSET:
            field_dict["website"] = website
        if address is not UNSET:
            field_dict["address"] = address
        if can_edit is not UNSET:
            field_dict["canEdit"] = can_edit
        if is_creator is not UNSET:
            field_dict["isCreator"] = is_creator
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        visibility = OrganizationVisibility(d.pop("visibility"))

        contact_email = d.pop("contactEmail")

        id = d.pop("id", UNSET)

        creator_id = d.pop("creatorId", UNSET)

        description = d.pop("description", UNSET)

        user_ids = cast(List[str], d.pop("userIds", UNSET))

        contact_phone = d.pop("contactPhone", UNSET)

        website = d.pop("website", UNSET)

        address = d.pop("address", UNSET)

        can_edit = d.pop("canEdit", UNSET)

        is_creator = d.pop("isCreator", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        organization = cls(
            name=name,
            visibility=visibility,
            contact_email=contact_email,
            id=id,
            creator_id=creator_id,
            description=description,
            user_ids=user_ids,
            contact_phone=contact_phone,
            website=website,
            address=address,
            can_edit=can_edit,
            is_creator=is_creator,
            created_at=created_at,
            updated_at=updated_at,
        )

        organization.additional_properties = d
        return organization

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
