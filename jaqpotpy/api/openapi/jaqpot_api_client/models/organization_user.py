from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.organization_user_association_type import OrganizationUserAssociationType
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrganizationUser")


@_attrs_define
class OrganizationUser:
    """
    Attributes:
        user_id (str):
        username (str):
        email (str):
        association_type (OrganizationUserAssociationType):
        id (Union[Unset, int]):
    """

    user_id: str
    username: str
    email: str
    association_type: OrganizationUserAssociationType
    id: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_id = self.user_id

        username = self.username

        email = self.email

        association_type = self.association_type.value

        id = self.id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "userId": user_id,
                "username": username,
                "email": email,
                "associationType": association_type,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("userId")

        username = d.pop("username")

        email = d.pop("email")

        association_type = OrganizationUserAssociationType(d.pop("associationType"))

        id = d.pop("id", UNSET)

        organization_user = cls(
            user_id=user_id,
            username=username,
            email=email,
            association_type=association_type,
            id=id,
        )

        organization_user.additional_properties = d
        return organization_user

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
