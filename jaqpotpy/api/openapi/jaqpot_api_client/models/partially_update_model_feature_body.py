from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.feature_type import FeatureType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.feature_possible_value import FeaturePossibleValue


T = TypeVar("T", bound="PartiallyUpdateModelFeatureBody")


@_attrs_define
class PartiallyUpdateModelFeatureBody:
    """
    Attributes:
        name (str): A name for the feature that will appear on top of the form field Example: Updated Feature Name.
        feature_type (FeatureType):  Example: FLOAT.
        units (Union[Unset, str]): The units that this feature is using Example: mg/L.
        description (Union[Unset, str]):  Example: An updated description for this feature.
        possible_values (Union[Unset, List['FeaturePossibleValue']]):
    """

    name: str
    feature_type: FeatureType
    units: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    possible_values: Union[Unset, List["FeaturePossibleValue"]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        feature_type = self.feature_type.value

        units = self.units

        description = self.description

        possible_values: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.possible_values, Unset):
            possible_values = []
            for possible_values_item_data in self.possible_values:
                possible_values_item = possible_values_item_data.to_dict()
                possible_values.append(possible_values_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "featureType": feature_type,
            }
        )
        if units is not UNSET:
            field_dict["units"] = units
        if description is not UNSET:
            field_dict["description"] = description
        if possible_values is not UNSET:
            field_dict["possibleValues"] = possible_values

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.feature_possible_value import FeaturePossibleValue

        d = src_dict.copy()
        name = d.pop("name")

        feature_type = FeatureType(d.pop("featureType"))

        units = d.pop("units", UNSET)

        description = d.pop("description", UNSET)

        possible_values = []
        _possible_values = d.pop("possibleValues", UNSET)
        for possible_values_item_data in _possible_values or []:
            possible_values_item = FeaturePossibleValue.from_dict(possible_values_item_data)

            possible_values.append(possible_values_item)

        partially_update_model_feature_body = cls(
            name=name,
            feature_type=feature_type,
            units=units,
            description=description,
            possible_values=possible_values,
        )

        partially_update_model_feature_body.additional_properties = d
        return partially_update_model_feature_body

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
