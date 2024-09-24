from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.transformer_config import TransformerConfig


T = TypeVar("T", bound="Transformer")


@_attrs_define
class Transformer:
    """A preprocessor for the model

    Attributes:
        name (str):  Example: StandardScaler.
        config (TransformerConfig):
    """

    name: str
    config: "TransformerConfig"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        config = self.config.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "config": config,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.transformer_config import TransformerConfig

        d = src_dict.copy()
        name = d.pop("name")

        config = TransformerConfig.from_dict(d.pop("config"))

        transformer = cls(
            name=name,
            config=config,
        )

        transformer.additional_properties = d
        return transformer

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
