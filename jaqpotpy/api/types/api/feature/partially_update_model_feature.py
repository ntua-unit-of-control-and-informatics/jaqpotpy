from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.feature import Feature
from ...models.partially_update_model_feature_body import PartiallyUpdateModelFeatureBody
from ...types import Response


def _get_kwargs(
    model_id: int,
    feature_id: int,
    *,
    body: PartiallyUpdateModelFeatureBody,
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "patch",
        "url": f"/v1/models/{model_id}/features/{feature_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, Feature]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = Feature.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.BAD_REQUEST:
        response_400 = cast(Any, None)
        return response_400
    if response.status_code == HTTPStatus.NOT_FOUND:
        response_404 = cast(Any, None)
        return response_404
    if response.status_code == HTTPStatus.UNAUTHORIZED:
        response_401 = cast(Any, None)
        return response_401
    if response.status_code == HTTPStatus.FORBIDDEN:
        response_403 = cast(Any, None)
        return response_403
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, Feature]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    model_id: int,
    feature_id: int,
    *,
    client: AuthenticatedClient,
    body: PartiallyUpdateModelFeatureBody,
) -> Response[Union[Any, Feature]]:
    """Update a feature for a specific model

     Update the name, description, and feature type of an existing feature within a specific model

    Args:
        model_id (int):
        feature_id (int):
        body (PartiallyUpdateModelFeatureBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Feature]]
    """

    kwargs = _get_kwargs(
        model_id=model_id,
        feature_id=feature_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    model_id: int,
    feature_id: int,
    *,
    client: AuthenticatedClient,
    body: PartiallyUpdateModelFeatureBody,
) -> Optional[Union[Any, Feature]]:
    """Update a feature for a specific model

     Update the name, description, and feature type of an existing feature within a specific model

    Args:
        model_id (int):
        feature_id (int):
        body (PartiallyUpdateModelFeatureBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Feature]
    """

    return sync_detailed(
        model_id=model_id,
        feature_id=feature_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    model_id: int,
    feature_id: int,
    *,
    client: AuthenticatedClient,
    body: PartiallyUpdateModelFeatureBody,
) -> Response[Union[Any, Feature]]:
    """Update a feature for a specific model

     Update the name, description, and feature type of an existing feature within a specific model

    Args:
        model_id (int):
        feature_id (int):
        body (PartiallyUpdateModelFeatureBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Feature]]
    """

    kwargs = _get_kwargs(
        model_id=model_id,
        feature_id=feature_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    model_id: int,
    feature_id: int,
    *,
    client: AuthenticatedClient,
    body: PartiallyUpdateModelFeatureBody,
) -> Optional[Union[Any, Feature]]:
    """Update a feature for a specific model

     Update the name, description, and feature type of an existing feature within a specific model

    Args:
        model_id (int):
        feature_id (int):
        body (PartiallyUpdateModelFeatureBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Feature]
    """

    return (
        await asyncio_detailed(
            model_id=model_id,
            feature_id=feature_id,
            client=client,
            body=body,
        )
    ).parsed
