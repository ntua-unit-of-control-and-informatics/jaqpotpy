from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_shared_models_response_200 import GetSharedModelsResponse200
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: Dict[str, Any] = {
        "method": "get",
        "url": "/v1/user/shared-models",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, GetSharedModelsResponse200]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = GetSharedModelsResponse200.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, GetSharedModelsResponse200]]:
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
) -> Response[Union[Any, GetSharedModelsResponse200]]:
    """Get paginated shared models

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetSharedModelsResponse200]]
    """

    kwargs = _get_kwargs(
        page=page,
        size=size,
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
) -> Optional[Union[Any, GetSharedModelsResponse200]]:
    """Get paginated shared models

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetSharedModelsResponse200]
    """

    return sync_detailed(
        client=client,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
) -> Response[Union[Any, GetSharedModelsResponse200]]:
    """Get paginated shared models

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, GetSharedModelsResponse200]]
    """

    kwargs = _get_kwargs(
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    page: Union[Unset, int] = 0,
    size: Union[Unset, int] = 10,
) -> Optional[Union[Any, GetSharedModelsResponse200]]:
    """Get paginated shared models

    Args:
        page (Union[Unset, int]):  Default: 0.
        size (Union[Unset, int]):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, GetSharedModelsResponse200]
    """

    return (
        await asyncio_detailed(
            client=client,
            page=page,
            size=size,
        )
    ).parsed
