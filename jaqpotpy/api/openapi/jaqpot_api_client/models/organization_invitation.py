from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.organization_invitation_status import OrganizationInvitationStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrganizationInvitation")


@_attrs_define
class OrganizationInvitation:
    """
    Attributes:
        user_email (str): Email address of the invited user
        status (OrganizationInvitationStatus): Status of the invitation
        expiration_date (str): Expiration date of the invitation
        id (Union[Unset, str]): ID of the invitation
        user_id (Union[Unset, str]): The user id associated with that invitation
    """

    user_email: str
    status: OrganizationInvitationStatus
    expiration_date: str
    id: Union[Unset, str] = UNSET
    user_id: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_email = self.user_email

        status = self.status.value

        expiration_date = self.expiration_date

        id = self.id

        user_id = self.user_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "userEmail": user_email,
                "status": status,
                "expirationDate": expiration_date,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if user_id is not UNSET:
            field_dict["userId"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_email = d.pop("userEmail")

        status = OrganizationInvitationStatus(d.pop("status"))

        expiration_date = d.pop("expirationDate")

        id = d.pop("id", UNSET)

        user_id = d.pop("userId", UNSET)

        organization_invitation = cls(
            user_email=user_email,
            status=status,
            expiration_date=expiration_date,
            id=id,
            user_id=user_id,
        )

        organization_invitation.additional_properties = d
        return organization_invitation

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
