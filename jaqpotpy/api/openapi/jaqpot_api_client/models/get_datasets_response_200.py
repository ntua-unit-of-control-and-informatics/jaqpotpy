from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset import Dataset


T = TypeVar("T", bound="GetDatasetsResponse200")


@_attrs_define
class GetDatasetsResponse200:
    """
    Attributes:
        content (Union[Unset, List['Dataset']]):
        total_elements (Union[Unset, int]):
        total_pages (Union[Unset, int]):
        page_size (Union[Unset, int]):
        page_number (Union[Unset, int]):
    """

    content: Union[Unset, List["Dataset"]] = UNSET
    total_elements: Union[Unset, int] = UNSET
    total_pages: Union[Unset, int] = UNSET
    page_size: Union[Unset, int] = UNSET
    page_number: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        content: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.content, Unset):
            content = []
            for content_item_data in self.content:
                content_item = content_item_data.to_dict()
                content.append(content_item)

        total_elements = self.total_elements

        total_pages = self.total_pages

        page_size = self.page_size

        page_number = self.page_number

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if content is not UNSET:
            field_dict["content"] = content
        if total_elements is not UNSET:
            field_dict["totalElements"] = total_elements
        if total_pages is not UNSET:
            field_dict["totalPages"] = total_pages
        if page_size is not UNSET:
            field_dict["pageSize"] = page_size
        if page_number is not UNSET:
            field_dict["pageNumber"] = page_number

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.dataset import Dataset

        d = src_dict.copy()
        content = []
        _content = d.pop("content", UNSET)
        for content_item_data in _content or []:
            content_item = Dataset.from_dict(content_item_data)

            content.append(content_item)

        total_elements = d.pop("totalElements", UNSET)

        total_pages = d.pop("totalPages", UNSET)

        page_size = d.pop("pageSize", UNSET)

        page_number = d.pop("pageNumber", UNSET)

        get_datasets_response_200 = cls(
            content=content,
            total_elements=total_elements,
            total_pages=total_pages,
            page_size=page_size,
            page_number=page_number,
        )

        get_datasets_response_200.additional_properties = d
        return get_datasets_response_200

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
