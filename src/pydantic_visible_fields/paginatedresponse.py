from __future__ import annotations

import logging
from typing import Any, AsyncIterable, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

from pydantic_visible_fields.core import _DEFAULT_ROLE

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Represents a paginated response for generic data types.

    Attributes:
        data: List of items of generic type T.
        limit: Maximum number of items returned per page.
        offset: Starting position of this page.
        items: Number of items in the current page.
        has_more: Boolean indicating whether more items exist.
        next_offset: Offset value for the next page or 0 if none.
    """

    data: List[T]
    limit: int
    offset: int
    items: int
    has_more: bool
    next_offset: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def from_async_iterable(
    iterator: AsyncIterable[T],
    limit: int,
    offset: int,
    role: Optional[str] = None,
) -> PaginatedResponse[Any]:
    """
    Process an async iterable into a paginated response, converting each item
    using to_response_model if available.

    NOTE: The iterator should already be properly positioned at the starting offset.
    The offset parameter is only used for informational purposes in the response.

    Args:
        iterator: The async iterable containing the source data objects (should
        already be at the correct offset)
        limit: Maximum number of items per page
        offset: Starting position for this page (for information only)
        role: The role to determine field visibility

    Returns:
        A PaginatedResponse containing properly converted model instances
    """
    data: List[Any] = []
    has_more = False

    # Use the provided role or fall back to the default role
    role = role or _DEFAULT_ROLE

    # Handle special case of limit <= 0
    if limit <= 0:
        return PaginatedResponse(
            limit=limit,
            offset=offset,
            data=[],
            items=0,
            has_more=False,
            next_offset=offset + limit,
        )

    async for item in iterator:
        if len(data) >= limit:
            has_more = True
            break

        # Convert using to_response_model if available
        if hasattr(item, "to_response_model"):
            response_item = item.to_response_model(role)
            logger.debug(f"Original item: {item}")
            logger.debug(f"Response model item: {response_item} for role: {role}")
            data.append(response_item)
        else:
            # No conversion method available, use the original item
            logger.debug(f"item: {item}")
            logger.debug("Could not convert item to response model, using as-is")
            data.append(item)

    # Always calculate next_offset, regardless of has_more
    next_offset = offset + limit

    # Create the paginated response
    response = PaginatedResponse(
        limit=limit,
        offset=offset,
        data=data,
        items=len(data),
        has_more=has_more,
        next_offset=next_offset,
    )

    return response


def from_iterable(
    iterator: List[T],
    limit: int,
    offset: int,
    role: Optional[str] = None,
) -> PaginatedResponse[Any]:
    """
    Process a synchronous iterable into a paginated response, converting each item
    using to_response_model if available.

    NOTE: The iterator should already contain only the items for the current page.
    The offset parameter is only used for informational purposes in the response.

    Args:
        iterator: The iterable containing the source data objects for this page
        limit: Maximum number of items per page
        offset: Starting position for this page (for information only)
        role: The role to determine field visibility

    Returns:
        A PaginatedResponse containing properly converted model instances
    """
    role = role or _DEFAULT_ROLE

    data: List[Any] = []
    has_more = False

    # Handle special case of limit <= 0
    if limit <= 0:
        return PaginatedResponse(
            limit=limit,
            offset=offset,
            data=[],
            items=0,
            has_more=False,
            next_offset=offset + limit,
        )

    # Process items up to the limit
    for item in iterator:
        if len(data) >= limit:
            has_more = True
            break

        # Convert using to_response_model if available
        if hasattr(item, "to_response_model"):
            response_item = item.to_response_model(role)
            data.append(response_item)
        else:
            # No conversion method available, use the original item
            data.append(item)

    # Always calculate next_offset, regardless of has_more
    next_offset = offset + limit

    # Create the paginated response
    response = PaginatedResponse(
        limit=limit,
        offset=offset,
        data=data,
        items=len(data),
        has_more=has_more,
        next_offset=next_offset,
    )

    return response
