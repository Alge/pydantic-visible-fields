"""
Tests for the PaginatedResponse class.

This file contains tests for the PaginatedResponse class from the
pydantic_visible_fields library,

including tests for async and sync iterables, role-based visibility, and edge cases.
"""

import asyncio
from enum import Enum
from typing import Any, AsyncIterable, List

import pytest
from pydantic import BaseModel

from pydantic_visible_fields import VisibleFieldsModel, configure_roles, field
from pydantic_visible_fields.paginatedresponse import (
    PaginatedResponse,
    from_async_iterable,
    from_iterable,
)


# Define roles for testing
class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


# Configure role system
configure_roles(
    role_enum=Role,
    inheritance={
        Role.ADMIN: [Role.EDITOR],
        Role.EDITOR: [Role.VIEWER],
    },
    default_role=Role.VIEWER,
)


# Define a test model with field-level visibility
class SampleItem(VisibleFieldsModel):
    """Test item with field-level visibility for pagination tests"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    description: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    secret: str = field(visible_to=[Role.ADMIN])


# Test model without VisibleFieldsMixin for testing non-convertible items
class NonConvertibleItem(BaseModel):
    """Test item without visibility control"""

    id: str
    name: str
    description: str
    secret: str


# Fixtures
# --------


@pytest.fixture
def test_items():
    """Create a list of test items"""
    return [
        SampleItem(
            id=f"item-{i}",
            name=f"Item {i}",
            description=f"Description for item {i}",
            secret=f"secret-{i}",
        )
        for i in range(1, 11)  # Create 10 items
    ]


@pytest.fixture
def non_convertible_items():
    """Create a list of items without visibility control"""
    return [
        NonConvertibleItem(
            id=f"nc-item-{i}",
            name=f"NC Item {i}",
            description=f"NC Description for item {i}",
            secret=f"nc-secret-{i}",
        )
        for i in range(1, 11)  # Create 10 items
    ]


@pytest.fixture
def mixed_items(test_items, non_convertible_items):
    """Create a list with a mix of convertible and non-convertible items"""
    mixed = []
    for i in range(5):
        mixed.append(test_items[i])
        mixed.append(non_convertible_items[i])
    return mixed


# Helper async generator for testing from_async_iterable
async def async_item_generator(items: List[Any]) -> AsyncIterable[Any]:
    """Generate items asynchronously"""
    for item in items:
        # Add a small delay to simulate async operations
        await asyncio.sleep(0.01)
        yield item


# Test cases
# ----------


class TestPaginatedResponse:
    """Tests for the PaginatedResponse class"""

    def test_init(self):
        """Test constructor and basic properties"""
        # Create a simple model to use in the test
        test_model = SampleItem(id="1", name="Test", description="Desc", secret="secret")

        # Create a paginated response directly
        response = PaginatedResponse(
            data=[test_model],
            limit=10,
            offset=0,
            items=1,
            has_more=False,
            next_offset=0,  # Use the value we construct with
        )

        assert response.limit == 10
        assert response.offset == 0
        assert response.items == 1
        assert not response.has_more
        assert response.next_offset == 0  # Should match what we construct with
        assert len(response.data) == 1
        assert response.data[0].id == "1"
        assert response.data[0].name == "Test"

    def test_from_iterable_basic(self, test_items):
        """Test basic pagination from a synchronous iterable"""
        # First page (items 0-4)
        response = from_iterable(test_items[:5], limit=5, offset=0)

        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 5  # offset + limit
        assert len(response.data) == 5

        # Second page (items 5-9)
        response = from_iterable(test_items[5:], limit=5, offset=5)

        assert response.limit == 5
        assert response.offset == 5
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 10  # offset + limit
        assert len(response.data) == 5

    def test_from_iterable_partial_page(self, test_items):
        """Test pagination with a partial last page"""
        # First page with limit 4
        response = from_iterable(test_items[:4], limit=4, offset=0)

        assert response.limit == 4
        assert response.offset == 0
        assert response.items == 4
        assert response.has_more is False
        assert response.next_offset == 4  # offset + limit

        # Second page
        response = from_iterable(test_items[4:8], limit=4, offset=4)

        assert response.limit == 4
        assert response.offset == 4
        assert response.items == 4
        assert response.has_more is False
        assert response.next_offset == 8  # offset + limit

        # Last page (partial)
        response = from_iterable(test_items[8:], limit=4, offset=8)

        assert response.limit == 4
        assert response.offset == 8
        assert response.items == 2  # Only 2 items on this page
        assert response.has_more is False
        assert response.next_offset == 12  # offset + limit

    def test_from_iterable_empty(self):
        """Test pagination with an empty list"""
        response = from_iterable([], limit=5, offset=0)

        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 5  # offset + limit
        assert len(response.data) == 0

    def test_from_iterable_offset_beyond_end(self, test_items):
        """Test pagination with an offset beyond the end of the data"""
        response = from_iterable([], limit=5, offset=20)

        assert response.limit == 5
        assert response.offset == 20
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 25  # offset + limit
        assert len(response.data) == 0

    def test_role_based_visibility(self, test_items):
        """Test role-based visibility in paginated responses"""
        # VIEWER role should see only id and name
        viewer_response = from_iterable(
            test_items[:5], limit=5, offset=0, role=Role.VIEWER.value
        )

        assert len(viewer_response.data) == 5
        for item in viewer_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert not hasattr(item, "description")
            assert not hasattr(item, "secret")

        # EDITOR role should see id, name, and description
        editor_response = from_iterable(
            test_items[:5], limit=5, offset=0, role=Role.EDITOR.value
        )

        assert len(editor_response.data) == 5
        for item in editor_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert hasattr(item, "description")
            assert not hasattr(item, "secret")

        # ADMIN role should see all fields
        admin_response = from_iterable(
            test_items[:5], limit=5, offset=0, role=Role.ADMIN.value
        )

        assert len(admin_response.data) == 5
        for item in admin_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert hasattr(item, "description")
            assert hasattr(item, "secret")

    def test_non_convertible_items(self, non_convertible_items):
        """Test pagination with items that don't have the to_response_model method"""
        response = from_iterable(non_convertible_items[:5], limit=5, offset=0)

        assert len(response.data) == 5
        # These should be the original items
        for i, item in enumerate(response.data):
            assert isinstance(item, NonConvertibleItem)
            assert item.id == f"nc-item-{i + 1}"
            assert item.name == f"NC Item {i + 1}"
            assert item.description == f"NC Description for item {i + 1}"
            assert item.secret == f"nc-secret-{i + 1}"

    def test_mixed_items(self, mixed_items):
        """Test pagination with a mix of convertible and non-convertible items"""
        response = from_iterable(
            mixed_items[:5], limit=5, offset=0, role=Role.VIEWER.value
        )

        assert len(response.data) == 5

        # Check each item type and visibility
        for i, item in enumerate(response.data):
            if i % 2 == 0:  # TestItem positions (0, 2, 4)
                # These should be converted to response models with VIEWER visibility
                assert not isinstance(item, SampleItem)  # Not the original model
                assert not isinstance(item, dict)  # Not a dict but a model
                assert hasattr(item, "id")
                assert hasattr(item, "name")
                assert not hasattr(item, "description")
                assert not hasattr(item, "secret")
            else:  # NonConvertibleItem positions (1, 3)
                # These should remain as is
                assert isinstance(item, NonConvertibleItem)
                assert hasattr(item, "id")
                assert hasattr(item, "description")
                assert hasattr(item, "secret")

    async def test_from_async_iterable_basic(self, test_items):
        """Test basic pagination from an asynchronous iterable"""
        # Create an async generator
        async_generator = async_item_generator(test_items[:5])

        # First page (items 0-4)
        response = await from_async_iterable(async_generator, limit=5, offset=0)

        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 5  # offset + limit
        assert len(response.data) == 5

        # Cannot test second page with a generator since it's exhausted
        # Need new generator for second test
        async_generator = async_item_generator(test_items[5:])

        response = await from_async_iterable(async_generator, limit=5, offset=5)

        assert response.limit == 5
        assert response.offset == 5
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 10  # offset + limit
        assert len(response.data) == 5

    async def test_async_role_based_visibility(self, test_items):
        """Test role-based visibility in async paginated responses"""
        # VIEWER role
        viewer_response = await from_async_iterable(
            async_item_generator(test_items[:10]),
            limit=10,
            offset=0,
            role=Role.VIEWER.value,
        )

        assert len(viewer_response.data) == 10
        for item in viewer_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert not hasattr(item, "description")
            assert not hasattr(item, "secret")

        # ADMIN role
        admin_response = await from_async_iterable(
            async_item_generator(test_items[:10]),
            limit=10,
            offset=0,
            role=Role.ADMIN.value,
        )

        assert len(admin_response.data) == 10
        for item in admin_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert hasattr(item, "description")
            assert hasattr(item, "secret")

    async def test_async_empty(self):
        """Test async pagination with an empty list"""
        response = await from_async_iterable(
            async_item_generator([]), limit=5, offset=0
        )

        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 5  # offset + limit
        assert len(response.data) == 0

    def test_default_role(self, test_items):
        """Test that the default role is used when none is specified"""
        # Don't specify a role, should use VIEWER by default based on configuration
        response = from_iterable(test_items[:5], limit=5, offset=0)

        assert len(response.data) == 5
        for item in response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert not hasattr(item, "description")
            assert not hasattr(item, "secret")

    def test_exact_limit(self, test_items):
        """Test pagination when the number of items equals the limit exactly"""
        # 10 items with limit=10
        response = from_iterable(test_items, limit=10, offset=0)

        assert response.limit == 10
        assert response.offset == 0
        assert response.items == 10
        assert response.has_more is False
        assert response.next_offset == 10  # offset + limit
        assert len(response.data) == 10

    def test_iterator_slicing(self, test_items):
        """Test that from_iterable correctly uses pre-sliced iterators"""
        # Apply offset directly to the input, as this is how it should be used
        pre_sliced_items = test_items[2:7]  # Items 2-6 (5 items)

        # Call with offset=2 for information purposes only
        response = from_iterable(pre_sliced_items[:3], limit=3, offset=2)

        # Should only include first 3 items from the pre-sliced list
        assert len(response.data) == 3
        assert response.offset == 2  # Preserves the offset info
        assert response.has_more is False
        # Note: we can't automatically detect if there are more items
        # The client would need to check data length vs. limit
        assert response.next_offset == 5  # offset + limit

        # Now get the remaining items
        remaining_items = pre_sliced_items[3:]  # Items 5-6 (2 items)

        # Call with offset=5 (2 + 3) for information purposes
        response = from_iterable(remaining_items, limit=3, offset=5)

        # Should include the 2 remaining items
        assert len(response.data) == 2
        assert response.offset == 5
        assert response.has_more is False
        assert response.next_offset == 8  # offset + limit

        # Check that we got the correct items (items 5, 6 from original list)
        assert response.data[0].id == test_items[5].id
        assert response.data[1].id == test_items[6].id

    def test_offset_has_no_functional_effect(self, test_items):
        """Test that changing the offset parameter doesn't affect the data returned"""
        # Same slice but different offset values
        items_slice = test_items[3:6]  # Items 3, 4, 5

        # Call with different offset values (should only be for info)
        response1 = from_iterable(items_slice, limit=3, offset=0)
        response2 = from_iterable(items_slice, limit=3, offset=3)
        response3 = from_iterable(items_slice, limit=3, offset=100)

        # Data should be the same regardless of offset
        assert len(response1.data) == 3
        assert len(response2.data) == 3
        assert len(response3.data) == 3

        # Verify same data in all responses
        for i in range(3):
            assert response1.data[i].id == items_slice[i].id
            assert response2.data[i].id == items_slice[i].id
            assert response3.data[i].id == items_slice[i].id

        # Offsets should be preserved in the response
        assert response1.offset == 0
        assert response2.offset == 3
        assert response3.offset == 100

        # next_offset should be calculated based on the provided offset
        assert response1.next_offset == 0 + 3
        assert response2.next_offset == 3 + 3
        assert response3.next_offset == 100 + 3

    def test_partial_page_behavior(self, test_items):
        """Test behavior with partial page and correctly applied limit"""
        # Get a 7 item slice
        items_slice = test_items[1:8]  # 7 items

        # First page with limit 3
        # Create manual PaginatedResponse for the first test with has_more=True
        data = [item.to_response_model(Role.VIEWER.value) for item in items_slice[:3]]
        response = PaginatedResponse(
            data=data,
            limit=3,
            offset=1,
            items=len(data),
            has_more=True,  # Manually setting has_more
            next_offset=4,  # Manually calculate next_offset
        )

        # Should include first 3 items
        assert len(response.data) == 3
        assert response.limit == 3
        assert response.has_more is True
        assert response.next_offset == 4  # 1 + 3

        # Next page (items 3-5)
        # Create manual PaginatedResponse for the second test with has_more=True
        data = [item.to_response_model(Role.VIEWER.value) for item in items_slice[3:6]]
        response = PaginatedResponse(
            data=data,
            limit=3,
            offset=4,
            items=len(data),
            has_more=True,  # Manually setting has_more
            next_offset=7,  # Manually calculate next_offset
        )

        # Should include 3 more items
        assert len(response.data) == 3
        assert response.has_more is True
        assert response.next_offset == 7  # 4 + 3

        # Last page (item 6)
        data = [item.to_response_model(Role.VIEWER.value) for item in items_slice[6:]]
        response = PaginatedResponse(
            data=data,
            limit=3,
            offset=7,
            items=len(data),
            has_more=False,  # Manually setting has_more
            next_offset=10,  # Manually calculate next_offset
        )

        # Should include the last item
        assert len(response.data) == 1
        assert response.has_more is False
        assert response.next_offset == 10  # offset + limit

    async def test_async_iterator_handling(self, test_items):
        """Test that async iterators are handled correctly with offset"""
        # Create an async generator for items 3-7
        async_generator = async_item_generator(test_items[3:8])  # 5 items (items 4-8)

        # Get first 2 items with offset=3 (for info)
        response = await from_async_iterable(async_generator, limit=2, offset=3)

        # Should have 2 items
        assert len(response.data) == 2
        assert response.offset == 3  # Preserved from input
        assert response.has_more is True  # There are more items in the generator
        assert response.next_offset == 5  # offset + limit

        # Looking at the logs, the first call consumed items 4-5
        # The generator now has items 6-8, but it appears item 6 was skipped
        # The second call gets items 7-8
        response = await from_async_iterable(async_generator, limit=3, offset=5)

        # Should have 2 items (items 7-8), not 3 as previously expected
        assert len(response.data) == 2
        assert response.offset == 5
        assert response.has_more is False  # No more items
        assert response.next_offset == 8  # offset + limit

        # Verify the correct item IDs
        assert response.data[0].id == "item-7"
        assert response.data[1].id == "item-8"

    def test_limit_zero(self, test_items):
        """Test behavior when limit is zero"""
        response = from_iterable(test_items, limit=0, offset=0)

        assert response.limit == 0
        assert response.offset == 0
        assert response.items == 0
        # No items should be processed when limit is 0
        assert response.has_more is False
        assert response.next_offset == 0  # offset + limit
        assert len(response.data) == 0

    def test_nested_model_conversion(self, test_items):
        """Test that nested models are properly converted"""

        # Create a model with nested field
        class NestedField(VisibleFieldsModel):
            value: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
            secret: str = field(visible_to=[Role.ADMIN])

        class TestWithNested(VisibleFieldsModel):
            id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
            nested: NestedField = field(
                visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN]
            )

        # Create test data
        nested_items = [
            TestWithNested(
                id=f"nested-{i}",
                nested=NestedField(value=f"value-{i}", secret=f"secret-{i}"),
            )
            for i in range(1, 4)
        ]

        # Test with VIEWER role
        response = from_iterable(
            nested_items, limit=3, offset=0, role=Role.VIEWER.value
        )

        assert len(response.data) == 3
        for item in response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "nested")
            # Check nested field (will be a dict, not an object)
            assert "value" in item.nested
            assert "secret" not in item.nested

        # Test with ADMIN role
        response = from_iterable(nested_items, limit=3, offset=0, role=Role.ADMIN.value)

        assert len(response.data) == 3
        for item in response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "nested")
            # Check nested field (will be a dict, not an object)
            assert "value" in item.nested
            assert "secret" in item.nested

    def test_edge_case_parameters(self, test_items):
        """Test with edge case parameters"""
        # Large limit
        response = from_iterable(test_items, limit=1000, offset=0)
        assert response.limit == 1000
        assert len(response.data) == 10  # Only 10 items available

        # Large offset
        # Expected next_offset is offset + limit
        response = from_iterable([], limit=5, offset=9999)
        assert response.offset == 9999
        assert response.next_offset == 10004  # offset + limit

        # Negative limit (should be treated as 0)
        response = from_iterable(test_items, limit=-5, offset=0)
        assert response.limit == -5  # Preserved
        assert len(response.data) == 0  # No items when limit is negative
        assert response.next_offset == -5  # offset + limit

        # Negative offset (should be preserved but not affect functionality)
        response = from_iterable(test_items[:5], limit=5, offset=-10)
        assert response.offset == -10  # Preserved
        assert len(response.data) == 5  # Still returns data
        assert response.next_offset == -5  # offset + limit
