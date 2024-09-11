from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.organization_visibility import OrganizationVisibility
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.organization_user import OrganizationUser
    from ..models.user import User


T = TypeVar("T", bound="Organization")


@_attrs_define
class Organization:
    """
    Attributes:
        name (str):  Example: my-awesome-org.
        visibility (OrganizationVisibility):
        contact_email (str):  Example: contact@my-awesome-org.com.
        id (Union[Unset, int]):
        creator (Union[Unset, User]):
        description (Union[Unset, str]):  Example: An awesome organization for managing models..
        organization_members (Union[Unset, List['OrganizationUser']]):
        contact_phone (Union[Unset, str]):  Example: +1234567890.
        website (Union[Unset, str]):  Example: http://www.my-awesome-org.com.
        address (Union[Unset, str]):  Example: 123 Organization St., City, Country.
        can_edit (Union[Unset, bool]): If the current user can edit the organization
        is_member (Union[Unset, bool]): If the current user is a member of the organization
        created_at (Union[Unset, str]):
        updated_at (Union[Unset, str]):
    """

    name: str
    visibility: OrganizationVisibility
    contact_email: str
    id: Union[Unset, int] = UNSET
    creator: Union[Unset, "User"] = UNSET
    description: Union[Unset, str] = UNSET
    organization_members: Union[Unset, List["OrganizationUser"]] = UNSET
    contact_phone: Union[Unset, str] = UNSET
    website: Union[Unset, str] = UNSET
    address: Union[Unset, str] = UNSET
    can_edit: Union[Unset, bool] = UNSET
    is_member: Union[Unset, bool] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        visibility = self.visibility.value

        contact_email = self.contact_email

        id = self.id

        creator: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.creator, Unset):
            creator = self.creator.to_dict()

        description = self.description

        organization_members: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.organization_members, Unset):
            organization_members = []
            for organization_members_item_data in self.organization_members:
                organization_members_item = organization_members_item_data.to_dict()
                organization_members.append(organization_members_item)

        contact_phone = self.contact_phone

        website = self.website

        address = self.address

        can_edit = self.can_edit

        is_member = self.is_member

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
        if creator is not UNSET:
            field_dict["creator"] = creator
        if description is not UNSET:
            field_dict["description"] = description
        if organization_members is not UNSET:
            field_dict["organizationMembers"] = organization_members
        if contact_phone is not UNSET:
            field_dict["contactPhone"] = contact_phone
        if website is not UNSET:
            field_dict["website"] = website
        if address is not UNSET:
            field_dict["address"] = address
        if can_edit is not UNSET:
            field_dict["canEdit"] = can_edit
        if is_member is not UNSET:
            field_dict["isMember"] = is_member
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.organization_user import OrganizationUser
        from ..models.user import User

        d = src_dict.copy()
        name = d.pop("name")

        visibility = OrganizationVisibility(d.pop("visibility"))

        contact_email = d.pop("contactEmail")

        id = d.pop("id", UNSET)

        _creator = d.pop("creator", UNSET)
        creator: Union[Unset, User]
        if isinstance(_creator, Unset):
            creator = UNSET
        else:
            creator = User.from_dict(_creator)

        description = d.pop("description", UNSET)

        organization_members = []
        _organization_members = d.pop("organizationMembers", UNSET)
        for organization_members_item_data in _organization_members or []:
            organization_members_item = OrganizationUser.from_dict(organization_members_item_data)

            organization_members.append(organization_members_item)

        contact_phone = d.pop("contactPhone", UNSET)

        website = d.pop("website", UNSET)

        address = d.pop("address", UNSET)

        can_edit = d.pop("canEdit", UNSET)

        is_member = d.pop("isMember", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        organization = cls(
            name=name,
            visibility=visibility,
            contact_email=contact_email,
            id=id,
            creator=creator,
            description=description,
            organization_members=organization_members,
            contact_phone=contact_phone,
            website=website,
            address=address,
            can_edit=can_edit,
            is_member=is_member,
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
