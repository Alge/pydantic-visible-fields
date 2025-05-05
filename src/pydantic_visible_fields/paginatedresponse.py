# paginatedresponse.py

import asyncio
from typing import Any, AsyncIterable, Generic, List, Optional, TypeVar
from pydantic import BaseModel, ConfigDict
import logging # Ensure logging is imported if used

from .core import _DEFAULT_ROLE, visible_fields_response

T = TypeVar("T")
logger = logging.getLogger(__name__)


class PaginatedResponse(BaseModel, Generic[T]):
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
    Creates a PaginatedResponse from an asynchronous iterable, applying role-based
    conversion and correctly determining if more items exist by fetching limit + 1 items.

    Args:
        iterator: The async iterable containing the source data objects.
        limit: Maximum number of items per page. Treat negative/zero as zero.
        offset: Informational offset value for the response.
        role: The role to use for converting items via visible_fields_response.

    Returns:
        A PaginatedResponse object.
    """
    temp_data: List[Any] = [] # Temporary list to hold fetched items
    processed_count = 0
    has_more = False
    # Ensure limit is non-negative for logic, but preserve original for response
    effective_limit = max(0, limit)
    role = role or _DEFAULT_ROLE

    if effective_limit > 0:
        # --- Start Fix: Fetch limit + 1 ---
        fetch_target = effective_limit + 1
        try:
            async for item in iterator:
                # Apply role-based conversion immediately
                response_item = visible_fields_response(item, role)
                temp_data.append(response_item)
                processed_count += 1
                if processed_count == fetch_target:
                    # Reached fetch target (limit + 1)
                    has_more = True
                    break # Stop fetching
        except Exception as e:
            logger.exception("Error processing async iterable for pagination.")
            raise e

        # Slice the data to the actual limit
        data = temp_data[:effective_limit]
        # Update processed_count to reflect the actual data returned
        processed_count = len(data)
        # --- End Fix ---
    else:
         # Limit is 0 or negative, return empty data
         data = []
         processed_count = 0
         has_more = False


    # Calculate next_offset based on the original provided offset and limit
    next_offset = offset + limit

    return PaginatedResponse(
        limit=limit,  # Return the original limit
        offset=offset,
        data=data,
        items=processed_count,
        has_more=has_more,
        next_offset=next_offset,
    )

# Keep from_iterable as fixed previously (always has_more=False)
def from_iterable(
    items: List[T], # Expects a List containing only items for the current page
    limit: int,
    offset: int,
    role: Optional[str] = None,
) -> PaginatedResponse[Any]:
    """
    Creates a PaginatedResponse from a synchronous list (representing a single page).
    'has_more' is always False as it cannot be determined from the input slice.
    """
    role = role or _DEFAULT_ROLE
    effective_limit = max(0, limit)
    data: List[Any] = []

    if effective_limit <= 0:
        return PaginatedResponse(
            limit=limit, offset=offset, data=[], items=0, has_more=False, next_offset=offset + limit
        )

    items_to_process = items[:effective_limit]
    for item in items_to_process:
        response_item = visible_fields_response(item, role)
        data.append(response_item)

    # Cannot determine has_more from the input slice alone
    has_more = False
    next_offset = offset + limit

    return PaginatedResponse(
        limit=limit, offset=offset, data=data, items=len(data), has_more=has_more, next_offset=next_offset
    )