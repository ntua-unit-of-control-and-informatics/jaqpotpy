from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.preprocessor_config import PreprocessorConfig


T = TypeVar("T", bound="Preprocessor")


@_attrs_define
class Preprocessor:
    """A preprocessor for the model

    Attributes:
        name (str):  Example: StandardScaler.
        config (PreprocessorConfig):
    """

    name: str
    config: "PreprocessorConfig"
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
        from ..models.preprocessor_config import PreprocessorConfig

        d = src_dict.copy()
        name = d.pop("name")

        config = PreprocessorConfig.from_dict(d.pop("config"))

        preprocessor = cls(
            name=name,
            config=config,
        )

        preprocessor.additional_properties = d
        return preprocessor

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
