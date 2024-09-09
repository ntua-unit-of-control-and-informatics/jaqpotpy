import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.feature_feature_dependency import FeatureFeatureDependency
from ..models.feature_type import FeatureType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.feature_meta import FeatureMeta
    from ..models.feature_possible_value import FeaturePossibleValue


T = TypeVar("T", bound="Feature")


@_attrs_define
class Feature:
    """
    Attributes:
        key (str): A key that must start with a letter, followed by any combination of letters, digits, hyphens, or
            underscores. For example, 'abc123', 'abc-test', or 'Abc_test'. It cannot start with a digit. Example: feature-
            key.
        name (str): A name for the feature that will appear on top of the form field Example: A feature name.
        feature_type (FeatureType):  Example: FLOAT.
        id (Union[Unset, int]):  Example: 1.
        meta (Union[Unset, FeatureMeta]): A JSON object containing meta information.
        units (Union[Unset, str]): A name for the feature that will appear on top of the form field Example: A feature
            unit.
        description (Union[Unset, str]):
        feature_dependency (Union[Unset, FeatureFeatureDependency]):  Example: DEPENDENT.
        visible (Union[Unset, bool]):  Example: True.
        possible_values (Union[Unset, List['FeaturePossibleValue']]):
        created_at (Union[Unset, datetime.datetime]): The date and time when the feature was created. Example:
            2023-01-01T12:00:00Z.
        updated_at (Union[Unset, str]): The date and time when the feature was last updated. Example:
            2023-01-01T12:00:00Z.
    """

    key: str
    name: str
    feature_type: FeatureType
    id: Union[Unset, int] = UNSET
    meta: Union[Unset, "FeatureMeta"] = UNSET
    units: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    feature_dependency: Union[Unset, FeatureFeatureDependency] = UNSET
    visible: Union[Unset, bool] = UNSET
    possible_values: Union[Unset, List["FeaturePossibleValue"]] = UNSET
    created_at: Union[Unset, datetime.datetime] = UNSET
    updated_at: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        key = self.key

        name = self.name

        feature_type = self.feature_type.value

        id = self.id

        meta: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.meta, Unset):
            meta = self.meta.to_dict()

        units = self.units

        description = self.description

        feature_dependency: Union[Unset, str] = UNSET
        if not isinstance(self.feature_dependency, Unset):
            feature_dependency = self.feature_dependency.value

        visible = self.visible

        possible_values: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.possible_values, Unset):
            possible_values = []
            for possible_values_item_data in self.possible_values:
                possible_values_item = possible_values_item_data.to_dict()
                possible_values.append(possible_values_item)

        created_at: Union[Unset, str] = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        updated_at = self.updated_at

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key": key,
                "name": name,
                "featureType": feature_type,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if meta is not UNSET:
            field_dict["meta"] = meta
        if units is not UNSET:
            field_dict["units"] = units
        if description is not UNSET:
            field_dict["description"] = description
        if feature_dependency is not UNSET:
            field_dict["featureDependency"] = feature_dependency
        if visible is not UNSET:
            field_dict["visible"] = visible
        if possible_values is not UNSET:
            field_dict["possibleValues"] = possible_values
        if created_at is not UNSET:
            field_dict["createdAt"] = created_at
        if updated_at is not UNSET:
            field_dict["updatedAt"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.feature_meta import FeatureMeta
        from ..models.feature_possible_value import FeaturePossibleValue

        d = src_dict.copy()
        key = d.pop("key")

        name = d.pop("name")

        feature_type = FeatureType(d.pop("featureType"))

        id = d.pop("id", UNSET)

        _meta = d.pop("meta", UNSET)
        meta: Union[Unset, FeatureMeta]
        if isinstance(_meta, Unset):
            meta = UNSET
        else:
            meta = FeatureMeta.from_dict(_meta)

        units = d.pop("units", UNSET)

        description = d.pop("description", UNSET)

        _feature_dependency = d.pop("featureDependency", UNSET)
        feature_dependency: Union[Unset, FeatureFeatureDependency]
        if isinstance(_feature_dependency, Unset):
            feature_dependency = UNSET
        else:
            feature_dependency = FeatureFeatureDependency(_feature_dependency)

        visible = d.pop("visible", UNSET)

        possible_values = []
        _possible_values = d.pop("possibleValues", UNSET)
        for possible_values_item_data in _possible_values or []:
            possible_values_item = FeaturePossibleValue.from_dict(possible_values_item_data)

            possible_values.append(possible_values_item)

        _created_at = d.pop("createdAt", UNSET)
        created_at: Union[Unset, datetime.datetime]
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = isoparse(_created_at)

        updated_at = d.pop("updatedAt", UNSET)

        feature = cls(
            key=key,
            name=name,
            feature_type=feature_type,
            id=id,
            meta=meta,
            units=units,
            description=description,
            feature_dependency=feature_dependency,
            visible=visible,
            possible_values=possible_values,
            created_at=created_at,
            updated_at=updated_at,
        )

        feature.additional_properties = d
        return feature

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
