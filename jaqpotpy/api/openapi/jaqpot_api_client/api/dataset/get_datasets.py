from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_datasets_response_200 import GetDatasetsResponse200
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
    sort: Union[Unset, List[str]] = UNSET,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params["page"] = page

    params["size"] = size

    json_sort: Union[Unset, List[str]] = UNSET
    if not isinstance(sort, Unset):
        json_sort = sort

    params["sort"] = json_sort

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: Dict[str, Any] = {
        "method": "get",
        "url": "/v1/user/datasets",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, GetDatasetsResponse200]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = GetDatasetsResponse200.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.NOT_FOUND:
        response_404 = cast(Any, None)
        return response_404
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, GetDatasetsResponse200]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
    sort: Union[Unset, List[str]] = UNSET,
) -> Response[Union[Any, GetDatasetsResponse200]]:
    """Get Datasets by User ID

     Retrieve all datasets associated with a specific user ID

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.
        sort (Union[Unset, List[str]]):  Example: ['field1|asc', 'field2|desc'].

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetDatasetsResponse200]]
    """

    kwargs = _get_kwargs(
        page=page,
        size=size,
        sort=sort,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
    sort: Union[Unset, List[str]] = UNSET,
) -> Optional[Union[Any, GetDatasetsResponse200]]:
    """Get Datasets by User ID

     Retrieve all datasets associated with a specific user ID

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.
        sort (Union[Unset, List[str]]):  Example: ['field1|asc', 'field2|desc'].

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetDatasetsResponse200]
    """

    return sync_detailed(
        client=client,
        page=page,
        size=size,
        sort=sort,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
    sort: Union[Unset, List[str]] = UNSET,
) -> Response[Union[Any, GetDatasetsResponse200]]:
    """Get Datasets by User ID

     Retrieve all datasets associated with a specific user ID

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.
        sort (Union[Unset, List[str]]):  Example: ['field1|asc', 'field2|desc'].

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetDatasetsResponse200]]
    """

    kwargs = _get_kwargs(
        page=page,
        size=size,
        sort=sort,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
    sort: Union[Unset, List[str]] = UNSET,
) -> Optional[Union[Any, GetDatasetsResponse200]]:
    """Get Datasets by User ID

     Retrieve all datasets associated with a specific user ID

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.
        sort (Union[Unset, List[str]]):  Example: ['field1|asc', 'field2|desc'].

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetDatasetsResponse200]
    """

    return (
        await asyncio_detailed(
            client=client,
            page=page,
            size=size,
            sort=sort,
        )
    ).parsed