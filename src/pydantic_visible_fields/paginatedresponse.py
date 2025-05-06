"""
Module defining PaginatedResponse model and utility functions for creating it
from iterables, integrating with role-based visibility.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterable, Generic, Iterable, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

# Use absolute import assuming core.py is in the same package/directory level
from .core import _DEFAULT_ROLE, visible_fields_response

T = TypeVar("T")  # Generic type variable, no longer bound to BaseModel

logger = logging.getLogger(__name__)


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard structure for returning paginated data in API responses.

    Includes metadata about the pagination state along with the data slice.

    Attributes:
        data: A list containing the items for the current page. The type `T`
              indicates the expected type of items within the list.
        limit: The requested maximum number of items per page.
        offset: The starting index requested for this page.
        items: The actual number of items returned in the `data` list for this page.
        has_more: A boolean indicating if there are more items available beyond
                  the current page.
        next_offset: The calculated offset to request the next page of results.
                     Calculated as `offset + limit`.
    """

    data: List[T]
    limit: int
    offset: int
    items: int
    has_more: bool
    next_offset: int

    # Allow arbitrary types in the 'data' list, as they might be
    # processed response models
    model_config = ConfigDict(arbitrary_types_allowed=True)


async def from_async_iterable(
    iterator: AsyncIterable[Any],
    limit: int,
    offset: int,
    role: Optional[str] = None,
) -> PaginatedResponse[Any]:
    """
    Creates a PaginatedResponse from an asynchronous iterable.

    This function iterates through the provided async iterable, fetches items
    up to the specified limit (plus one extra to check for more data), applies
    role-based filtering/conversion using `visible_fields_response`, and constructs
    a `PaginatedResponse` object.

    Args:
        iterator: The asynchronous iterable containing the source data objects.
                  This iterable should yield the full potential dataset; this
                  function handles consuming only the necessary items based on limit.
        limit: Maximum number of items per page. Non-positive values result
               in an empty response.
        offset: Informational offset value included in the response. This function
                does not skip items based on this offset; the iterator should
                ideally start at the correct position if applicable.
        role: The role identifier (string) to use for converting items via
              `visible_fields_response`. Uses the configured default role if None.

    Returns:
        A PaginatedResponse object containing the processed data and
        pagination metadata.

    Raises:
        Exception: Propagates any exception raised during iteration or processing.
    """
    temp_data: List[Any] = []
    processed_count = 0
    has_more = False
    effective_limit = max(0, limit)  # Ensure limit is usable for logic
    actual_role = role or _DEFAULT_ROLE or ""  # Ensure role is a string

    logger.debug(
        f"from_async_iterable called: limit={limit}, offset={offset}, "
        f"role='{actual_role}', effective_limit={effective_limit}"
    )

    if effective_limit > 0:
        fetch_target = effective_limit + 1  # Fetch one extra to check has_more
        items_fetched = 0
        try:
            async for item in iterator:
                items_fetched += 1
                logger.debug(
                    f"Fetched async item {items_fetched}/{fetch_target}. Processing..."
                )
                # Apply role-based conversion before appending to temp list
                response_item = visible_fields_response(item, actual_role)
                temp_data.append(response_item)

                if items_fetched == fetch_target:
                    # Reached fetch target (limit + 1), indicates more items exist
                    has_more = True
                    logger.debug(
                        f"Reached fetch target ({fetch_target}), setting has_more=True."
                    )
                    break  # Stop fetching further items
        except Exception as e:
            logger.exception("Error processing async iterable for pagination.")
            raise e

        # Slice the data to the actual limit
        data = temp_data[:effective_limit]
        processed_count = len(data)
        logger.debug(
            f"Finished fetching. Fetched {items_fetched} items. "
            f"Returning {processed_count} items. has_more={has_more}"
        )
    else:
        # Limit is 0 or negative
        logger.debug("Effective limit is 0, returning empty data.")
        data = []
        processed_count = 0
        has_more = False

    # Calculate next_offset based on the *original* provided offset and limit
    next_offset = offset + limit
    logger.debug(
        f"Calculated next_offset={next_offset} (offset={offset} + limit={limit})"
    )

    return PaginatedResponse(
        limit=limit,  # Return the original requested limit
        offset=offset,
        data=data,
        items=processed_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def from_iterable(
    items: Iterable[Any],  # Changed to Iterable for more flexibility
    limit: int,
    offset: int,
    role: Optional[str] = None,
) -> PaginatedResponse[Any]:
    """
    Creates a PaginatedResponse from a synchronous iterable (e.g., a list).

    Processes items from the iterable up to the specified limit, applies role-based
    filtering/conversion, and determines `has_more` by trying to fetch one extra item.

    Args:
        items: The synchronous iterable containing the source data objects.
               This iterable should yield the full potential dataset; this
               function handles consuming only the necessary items based on limit.
        limit: Maximum number of items per page. Non-positive values result
               in an empty response.
        offset: Informational offset value included in the response. This function
                does not skip items based on this offset.
        role: The role identifier (string) to use for converting items via
              `visible_fields_response`. Uses the configured default role if None.

    Returns:
        A PaginatedResponse containing the processed data and pagination metadata.

    Raises:
        Exception: Propagates any exception raised during iteration or processing.
    """
    data: List[Any] = []
    processed_count = 0
    has_more = False
    effective_limit = max(0, limit)
    actual_role = role or _DEFAULT_ROLE or ""

    logger.debug(
        f"from_iterable called: limit={limit}, offset={offset}, role='{actual_role}', "
        f"effective_limit={effective_limit}"
    )

    if effective_limit > 0:
        iterator = iter(items)  # Ensure we have an iterator
        fetch_count = 0
        try:
            while fetch_count < effective_limit:
                try:
                    item = next(iterator)
                    fetch_count += 1
                    logger.debug(
                        f"Processing sync item {fetch_count}/{effective_limit}..."
                    )
                    response_item = visible_fields_response(item, actual_role)
                    data.append(response_item)
                    processed_count += 1
                except StopIteration:
                    # Iterator ended before reaching the limit
                    logger.debug("Iterator exhausted before reaching limit.")
                    break

            # After the loop, check if there's one more item for has_more
            if processed_count == effective_limit:
                try:
                    next(iterator)  # Try to get one more item
                    has_more = True  # If successful, there are more items
                    logger.debug(
                        "Successfully fetched one extra item, setting has_more=True."
                    )
                except StopIteration:
                    # Iterator was exhausted exactly at the limit
                    has_more = False
                    logger.debug("Iterator exhausted exactly at limit, has_more=False.")

        except Exception as e:
            logger.exception("Error processing sync iterable for pagination.")
            raise e
    else:
        logger.debug("Effective limit is 0, returning empty data.")
        data = []
        processed_count = 0
        has_more = False

    next_offset = offset + limit
    logger.debug(
        f"Finished processing. Returning {processed_count} items. has_more={has_more}. "
        f"next_offset={next_offset}"
    )

    return PaginatedResponse(
        limit=limit,
        offset=offset,
        data=data,
        items=processed_count,
        has_more=has_more,
        next_offset=next_offset,
    )
