from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.organization_invitation import OrganizationInvitation
from ...types import Response


def _get_kwargs(
    name: str,
    uuid: str,
    *,
    body: OrganizationInvitation,
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "put",
        "url": f"/v1/organizations/{name}/invitations/{uuid}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, OrganizationInvitation]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = OrganizationInvitation.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    if response.status_code == HTTPStatus.NOT_FOUND:
        response_404 = cast(Any, None)
        return response_404
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, OrganizationInvitation]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    name: str,
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrganizationInvitation,
) -> Response[Union[Any, OrganizationInvitation]]:
    """Update the status of an invitation

     This endpoint allows a user to update the status of an invitation.

    Args:
        name (str):
        uuid (str):
        body (OrganizationInvitation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, OrganizationInvitation]]
    """

    kwargs = _get_kwargs(
        name=name,
        uuid=uuid,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    name: str,
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrganizationInvitation,
) -> Optional[Union[Any, OrganizationInvitation]]:
    """Update the status of an invitation

     This endpoint allows a user to update the status of an invitation.

    Args:
        name (str):
        uuid (str):
        body (OrganizationInvitation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, OrganizationInvitation]
    """

    return sync_detailed(
        name=name,
        uuid=uuid,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    name: str,
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrganizationInvitation,
) -> Response[Union[Any, OrganizationInvitation]]:
    """Update the status of an invitation

     This endpoint allows a user to update the status of an invitation.

    Args:
        name (str):
        uuid (str):
        body (OrganizationInvitation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, OrganizationInvitation]]
    """

    kwargs = _get_kwargs(
        name=name,
        uuid=uuid,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    name: str,
    uuid: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: OrganizationInvitation,
) -> Optional[Union[Any, OrganizationInvitation]]:
    """Update the status of an invitation

     This endpoint allows a user to update the status of an invitation.

    Args:
        name (str):
        uuid (str):
        body (OrganizationInvitation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, OrganizationInvitation]
    """

    return (
        await asyncio_detailed(
            name=name,
            uuid=uuid,
            client=client,
            body=body,
        )
    ).parsed
