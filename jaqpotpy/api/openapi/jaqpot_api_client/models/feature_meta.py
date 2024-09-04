from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.feature_meta_additional_property import FeatureMetaAdditionalProperty


T = TypeVar("T", bound="FeatureMeta")


@_attrs_define
class FeatureMeta:
    """A JSON object containing meta information."""

    additional_properties: Dict[str, "FeatureMetaAdditionalProperty"] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        field_dict: Dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.feature_meta_additional_property import FeatureMetaAdditionalProperty

        d = src_dict.copy()
        feature_meta = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = FeatureMetaAdditionalProperty.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        feature_meta.additional_properties = additional_properties
        return feature_meta

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "FeatureMetaAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "FeatureMetaAdditionalProperty") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
