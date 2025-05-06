"""Field-level visibility control for Pydantic models."""

import logging  # Import logging

from pydantic_visible_fields.core import (
    VisibleFieldsMixin,
    VisibleFieldsModel,
    configure_roles,
    field,
    visible_fields_response,
)

# Import pagination functions if they should be part of the public API
from pydantic_visible_fields.paginatedresponse import (
    PaginatedResponse,
    from_async_iterable,
    from_iterable,
)

__version__ = "0.2.6"  # Update version if changes warrant it
__all__ = [
    # Core components
    "VisibleFieldsMixin",
    "VisibleFieldsModel",
    "field",
    "configure_roles",
    "visible_fields_response",
    # Pagination components
    "PaginatedResponse",
    "from_iterable",
    "from_async_iterable",
]


# --- Logging Setup ---
# Add a NullHandler to the package's root logger.
# This prevents 'No handler found' warnings if the library user
# doesn't configure logging, while allowing them to enable logging
# messages from this library if they do configure their root logger.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

logger.setLevel(logging.INFO)
# --- End Logging Setup ---
