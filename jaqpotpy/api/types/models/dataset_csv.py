from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dataset_csv_status import DatasetCSVStatus
from ..models.dataset_type import DatasetType
from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetCSV")


@_attrs_define
class DatasetCSV:
    """
    Attributes:
        type (DatasetType):  Example: PREDICTION.
        input_file (str): A base64 representation in CSV format of the input values.
        id (Union[Unset, int]):  Example: 1.
        values (Union[Unset, List[Any]]):
        status (Union[Unset, DatasetCSVStatus]):
        failure_reason (Union[Unset, str]):
        created_at (Union[Unset, str]):
        updated_at (Union[Unset, str]):
    """

    type: DatasetType
    input_file: str
    id: Union[Unset, int] = UNSET
    values: Union[Unset, List[Any]] = UNSET
    status: Union[Unset, DatasetCSVStatus] = UNSET
    failure_reason: Union[Unset, str] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        type = self.type.value

        input_file = self.input_file

        id = self.id

        values: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.values, Unset):
            values = self.values

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
                "inputFile": input_file,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if values is not UNSET:
            field_dict["values"] = values
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

        input_file = d.pop("inputFile")

        id = d.pop("id", UNSET)

        values = cast(List[Any], d.pop("values", UNSET))

        _status = d.pop("status", UNSET)
        status: Union[Unset, DatasetCSVStatus]
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = DatasetCSVStatus(_status)

        failure_reason = d.pop("failureReason", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        dataset_csv = cls(
            type=type,
            input_file=input_file,
            id=id,
            values=values,
            status=status,
            failure_reason=failure_reason,
            created_at=created_at,
            updated_at=updated_at,
        )

        dataset_csv.additional_properties = d
        return dataset_csv

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