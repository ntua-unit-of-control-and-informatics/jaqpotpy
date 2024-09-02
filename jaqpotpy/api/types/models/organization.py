from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.organization_visibility import OrganizationVisibility
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model import Model


T = TypeVar("T", bound="Organization")


@_attrs_define
class Organization:
    """Attributes
    name (str):  Example: my-awesome-org.
    contact_email (str):  Example: contact@my-awesome-org.com.
    visibility (OrganizationVisibility):
    id (Union[Unset, int]):
    creator_id (Union[Unset, str]):
    description (Union[Unset, str]):  Example: An awesome organization for managing models..
    user_ids (Union[Unset, List[str]]):
    models (Union[Unset, List['Model']]):
    contact_phone (Union[Unset, str]):  Example: +1234567890.
    website (Union[Unset, str]):  Example: http://www.my-awesome-org.com.
    address (Union[Unset, str]):  Example: 123 Organization St., City, Country.
    can_edit (Union[Unset, bool]): If the current user can edit the organization
    created_at (Union[Unset, str]):
    updated_at (Union[Unset, str]):

    """

    name: str
    contact_email: str
    visibility: OrganizationVisibility
    id: Union[Unset, int] = UNSET
    creator_id: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    user_ids: Union[Unset, List[str]] = UNSET
    models: Union[Unset, List["Model"]] = UNSET
    contact_phone: Union[Unset, str] = UNSET
    website: Union[Unset, str] = UNSET
    address: Union[Unset, str] = UNSET
    can_edit: Union[Unset, bool] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        contact_email = self.contact_email

        visibility = self.visibility.value

        id = self.id

        creator_id = self.creator_id

        description = self.description

        user_ids: Union[Unset, List[str]] = UNSET
        if not isinstance(self.user_ids, Unset):
            user_ids = self.user_ids

        models: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.models, Unset):
            models = []
            for models_item_data in self.models:
                models_item = models_item_data.to_dict()
                models.append(models_item)

        contact_phone = self.contact_phone

        website = self.website

        address = self.address

        can_edit = self.can_edit

        created_at = self.created_at

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "contactEmail": contact_email,
                "visibility": visibility,
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
        if models is not UNSET:
            field_dict["models"] = models
        if contact_phone is not UNSET:
            field_dict["contactPhone"] = contact_phone
        if website is not UNSET:
            field_dict["website"] = website
        if address is not UNSET:
            field_dict["address"] = address
        if can_edit is not UNSET:
            field_dict["canEdit"] = can_edit
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.model import Model

        d = src_dict.copy()
        name = d.pop("name")

        contact_email = d.pop("contactEmail")

        visibility = OrganizationVisibility(d.pop("visibility"))

        id = d.pop("id", UNSET)

        creator_id = d.pop("creatorId", UNSET)

        description = d.pop("description", UNSET)

        user_ids = cast(List[str], d.pop("userIds", UNSET))

        models = []
        _models = d.pop("models", UNSET)
        for models_item_data in _models or []:
            models_item = Model.from_dict(models_item_data)

            models.append(models_item)

        contact_phone = d.pop("contactPhone", UNSET)

        website = d.pop("website", UNSET)

        address = d.pop("address", UNSET)

        can_edit = d.pop("canEdit", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        organization = cls(
            name=name,
            contact_email=contact_email,
            visibility=visibility,
            id=id,
            creator_id=creator_id,
            description=description,
            user_ids=user_ids,
            models=models,
            contact_phone=contact_phone,
            website=website,
            address=address,
            can_edit=can_edit,
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
