from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dataset_entry_type import DatasetEntryType
from ..models.dataset_status import DatasetStatus
from ..models.dataset_type import DatasetType
from ..types import UNSET, Unset

T = TypeVar("T", bound="Dataset")


@_attrs_define
class Dataset:
    """Attributes
    type (DatasetType):  Example: PREDICTION.
    entry_type (DatasetEntryType):  Example: ARRAY.
    input_ (List[Any]):
    id (Union[Unset, int]):  Example: 1.
    result (Union[Unset, List[Any]]):
    status (Union[Unset, DatasetStatus]):
    failure_reason (Union[Unset, str]):
    created_at (Union[Unset, str]):
    updated_at (Union[Unset, str]):

    """

    type: DatasetType
    entry_type: DatasetEntryType
    input_: List[Any]
    id: Union[Unset, int] = UNSET
    result: Union[Unset, List[Any]] = UNSET
    status: Union[Unset, DatasetStatus] = UNSET
    failure_reason: Union[Unset, str] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        type = self.type.value

        entry_type = self.entry_type.value

        input_ = self.input_

        id = self.id

        result: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.result, Unset):
            result = self.result

        status: Union[Unset, str] = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        failure_reason = self.failure_reason

        created_at = self.created_at

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type,
                "entryType": entry_type,
                "input": input_,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if result is not UNSET:
            field_dict["result"] = result
        if status is not UNSET:
            field_dict["status"] = status
        if failure_reason is not UNSET:
            field_dict["failureReason"] = failure_reason
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        type = DatasetType(d.pop("type"))

        entry_type = DatasetEntryType(d.pop("entryType"))

        input_ = cast(List[Any], d.pop("input"))

        id = d.pop("id", UNSET)

        result = cast(List[Any], d.pop("result", UNSET))

        _status = d.pop("status", UNSET)
        status: Union[Unset, DatasetStatus]
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = DatasetStatus(_status)

        failure_reason = d.pop("failureReason", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        dataset = cls(
            type=type,
            entry_type=entry_type,
            input_=input_,
            id=id,
            result=result,
            status=status,
            failure_reason=failure_reason,
            created_at=created_at,
            updated_at=updated_at,
        )

        dataset.additional_properties = d
        return dataset

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
