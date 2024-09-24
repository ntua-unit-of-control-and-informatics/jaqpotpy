from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_extra_config_torch_config import ModelExtraConfigTorchConfig
    from ..models.transformer import Transformer


T = TypeVar("T", bound="ModelExtraConfig")


@_attrs_define
class ModelExtraConfig:
    """A JSON object containing extra configuration for the model

    Attributes:
        torch_config (Union[Unset, ModelExtraConfigTorchConfig]):
        preprocessors (Union[Unset, List['Transformer']]):
        featurizers (Union[Unset, List['Transformer']]):
        doa (Union[Unset, List['Transformer']]):
    """

    torch_config: Union[Unset, "ModelExtraConfigTorchConfig"] = UNSET
    preprocessors: Union[Unset, List["Transformer"]] = UNSET
    featurizers: Union[Unset, List["Transformer"]] = UNSET
    doa: Union[Unset, List["Transformer"]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        torch_config: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.torch_config, Unset):
            torch_config = self.torch_config.to_dict()

        preprocessors: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.preprocessors, Unset):
            preprocessors = []
            for preprocessors_item_data in self.preprocessors:
                preprocessors_item = preprocessors_item_data.to_dict()
                preprocessors.append(preprocessors_item)

        featurizers: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.featurizers, Unset):
            featurizers = []
            for featurizers_item_data in self.featurizers:
                featurizers_item = featurizers_item_data.to_dict()
                featurizers.append(featurizers_item)

        doa: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.doa, Unset):
            doa = []
            for doa_item_data in self.doa:
                doa_item = doa_item_data.to_dict()
                doa.append(doa_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if torch_config is not UNSET:
            field_dict["torchConfig"] = torch_config
        if preprocessors is not UNSET:
            field_dict["preprocessors"] = preprocessors
        if featurizers is not UNSET:
            field_dict["featurizers"] = featurizers
        if doa is not UNSET:
            field_dict["doa"] = doa

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.model_extra_config_torch_config import ModelExtraConfigTorchConfig
        from ..models.transformer import Transformer

        d = src_dict.copy()
        _torch_config = d.pop("torchConfig", UNSET)
        torch_config: Union[Unset, ModelExtraConfigTorchConfig]
        if isinstance(_torch_config, Unset):
            torch_config = UNSET
        else:
            torch_config = ModelExtraConfigTorchConfig.from_dict(_torch_config)

        preprocessors = []
        _preprocessors = d.pop("preprocessors", UNSET)
        for preprocessors_item_data in _preprocessors or []:
            preprocessors_item = Transformer.from_dict(preprocessors_item_data)

            preprocessors.append(preprocessors_item)

        featurizers = []
        _featurizers = d.pop("featurizers", UNSET)
        for featurizers_item_data in _featurizers or []:
            featurizers_item = Transformer.from_dict(featurizers_item_data)

            featurizers.append(featurizers_item)

        doa = []
        _doa = d.pop("doa", UNSET)
        for doa_item_data in _doa or []:
            doa_item = Transformer.from_dict(doa_item_data)

            doa.append(doa_item)

        model_extra_config = cls(
            torch_config=torch_config,
            preprocessors=preprocessors,
            featurizers=featurizers,
            doa=doa,
        )

        model_extra_config.additional_properties = d
        return model_extra_config

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
