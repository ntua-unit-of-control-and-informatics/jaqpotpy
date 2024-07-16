from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.feature_type import FeatureType
from ..types import UNSET, Unset

T = TypeVar("T", bound="PartiallyUpdateModelFeatureBody")


@_attrs_define
class PartiallyUpdateModelFeatureBody:
    """
    Attributes:
        name (str): A name for the feature that will appear on top of the form field Example: Updated Feature Name.
        feature_type (FeatureType):  Example: FLOAT.
        description (Union[Unset, str]):  Example: An updated description for this feature.
        possible_values (Union[Unset, List[str]]):
    """

    name: str
    feature_type: FeatureType
    description: Union[Unset, str] = UNSET
    possible_values: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        feature_type = self.feature_type.value

        description = self.description

        possible_values: Union[Unset, List[str]] = UNSET
        if not isinstance(self.possible_values, Unset):
            possible_values = self.possible_values

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "featureType": feature_type,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if possible_values is not UNSET:
            field_dict["possibleValues"] = possible_values

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        feature_type = FeatureType(d.pop("featureType"))

        description = d.pop("description", UNSET)

        possible_values = cast(List[str], d.pop("possibleValues", UNSET))

        partially_update_model_feature_body = cls(
            name=name,
            feature_type=feature_type,
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
