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

# Adjust import path if necessary
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
# Ensure this configuration is consistent with other test files if roles are shared
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
        test_model = SampleItem(
            id="1", name="Test", description="Desc", secret="secret"
        )
        response = PaginatedResponse(
            data=[test_model],
            limit=10,
            offset=0,
            items=1,
            has_more=False,
            next_offset=0,
        )
        assert response.limit == 10
        assert response.offset == 0
        assert response.items == 1
        assert not response.has_more
        assert response.next_offset == 0
        assert len(response.data) == 1
        assert response.data[0].id == "1"
        assert response.data[0].name == "Test"

    def test_from_iterable_basic(self, test_items):
        """Test basic pagination from a synchronous iterable"""
        response = from_iterable(test_items[:5], limit=5, offset=0)
        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 5
        assert len(response.data) == 5

        response = from_iterable(test_items[5:], limit=5, offset=5)
        assert response.limit == 5
        assert response.offset == 5
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 10
        assert len(response.data) == 5

    def test_from_iterable_partial_page(self, test_items):
        """Test pagination with a partial last page"""
        response = from_iterable(test_items[:4], limit=4, offset=0)
        assert response.limit == 4
        assert response.offset == 0
        assert response.items == 4
        assert response.has_more is False
        assert response.next_offset == 4

        response = from_iterable(test_items[4:8], limit=4, offset=4)
        assert response.limit == 4
        assert response.offset == 4
        assert response.items == 4
        assert response.has_more is False
        assert response.next_offset == 8

        response = from_iterable(test_items[8:], limit=4, offset=8)
        assert response.limit == 4
        assert response.offset == 8
        assert response.items == 2
        assert response.has_more is False
        assert response.next_offset == 12

    def test_from_iterable_empty(self):
        """Test pagination with an empty list"""
        response = from_iterable([], limit=5, offset=0)
        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 5
        assert len(response.data) == 0

    def test_from_iterable_offset_beyond_end(self, test_items):
        """Test pagination with an offset beyond the end of the data"""
        # Note: from_iterable gets an empty list here, offset is informational
        response = from_iterable([], limit=5, offset=20)
        assert response.limit == 5
        assert response.offset == 20
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 25
        assert len(response.data) == 0

    def test_role_based_visibility(self, test_items):
        """Test role-based visibility in paginated responses"""
        viewer_response = from_iterable(
            test_items[:5], limit=5, offset=0, role=Role.VIEWER.value
        )
        assert len(viewer_response.data) == 5
        for item in viewer_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert not hasattr(item, "description")
            assert not hasattr(item, "secret")

        editor_response = from_iterable(
            test_items[:5], limit=5, offset=0, role=Role.EDITOR.value
        )
        assert len(editor_response.data) == 5
        for item in editor_response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert hasattr(item, "description")
            assert not hasattr(item, "secret")

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
        for i, item in enumerate(response.data):
            assert isinstance(item, NonConvertibleItem)
            assert item.id == f"nc-item-{i + 1}"

    def test_mixed_items(self, mixed_items):
        """Test pagination with a mix of convertible and non-convertible items"""
        response = from_iterable(
            mixed_items[:5], limit=5, offset=0, role=Role.VIEWER.value
        )
        assert len(response.data) == 5
        for i, item in enumerate(response.data):
            if i % 2 == 0:  # SampleItem positions
                assert not isinstance(item, SampleItem)
                assert not isinstance(item, dict)
                assert hasattr(item, "id")
                assert hasattr(item, "name")
                assert not hasattr(item, "description")
            else:  # NonConvertibleItem positions
                assert isinstance(item, NonConvertibleItem)
                assert hasattr(item, "id")
                assert hasattr(item, "description")

    async def test_from_async_iterable_basic(self, test_items):
        """Test basic pagination from an asynchronous iterable"""
        async_generator = async_item_generator(test_items[:5])
        response = await from_async_iterable(async_generator, limit=5, offset=0)
        assert response.limit == 5
        assert response.offset == 0
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 5
        assert len(response.data) == 5

        async_generator = async_item_generator(test_items[5:])
        response = await from_async_iterable(async_generator, limit=5, offset=5)
        assert response.limit == 5
        assert response.offset == 5
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == 10
        assert len(response.data) == 5

    async def test_async_role_based_visibility(self, test_items):
        """Test role-based visibility in async paginated responses"""
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
        assert response.next_offset == 5
        assert len(response.data) == 0

    def test_default_role(self, test_items):
        """Test that the default role is used when none is specified"""
        response = from_iterable(test_items[:5], limit=5, offset=0)
        assert len(response.data) == 5
        for item in response.data:
            assert hasattr(item, "id")
            assert hasattr(item, "name")
            assert not hasattr(item, "description")

    def test_exact_limit(self, test_items):
        """Test pagination when the number of items equals the limit exactly"""
        response = from_iterable(test_items, limit=10, offset=0)
        assert response.limit == 10
        assert response.offset == 0
        assert response.items == 10
        assert response.has_more is False
        assert response.next_offset == 10
        assert len(response.data) == 10

    def test_iterator_slicing(self, test_items):
        """Test that from_iterable correctly handles pre-sliced iterators"""
        pre_sliced_items = test_items[2:7]  # Items 2-6 (5 items)

        # Use from_iterable with the slice and appropriate limit/offset
        # Note: from_iterable doesn't re-apply slicing based on offset
        response = from_iterable(pre_sliced_items[:3], limit=3, offset=2)
        assert len(response.data) == 3
        assert response.offset == 2
        assert response.has_more is False  # Cannot determine from pre-sliced data
        assert response.next_offset == 5  # offset + limit

        remaining_items = pre_sliced_items[3:]  # Items 5-6 (2 items)
        response = from_iterable(remaining_items, limit=3, offset=5)
        assert len(response.data) == 2
        assert response.offset == 5
        assert response.has_more is False  # Cannot determine
        assert response.next_offset == 8
        assert (
            response.data[0].id == test_items[5].id
        )  # Check correct items were processed
        assert response.data[1].id == test_items[6].id

    def test_offset_has_no_functional_effect(self, test_items):
        """
        Test that changing the offset parameter doesn't affect the data returned
        when using from_iterable on a pre-sliced list
        """
        items_slice = test_items[3:6]  # Items 3, 4, 5
        response1 = from_iterable(items_slice, limit=3, offset=0)
        response2 = from_iterable(items_slice, limit=3, offset=3)
        response3 = from_iterable(items_slice, limit=3, offset=100)

        assert len(response1.data) == 3
        assert len(response2.data) == 3
        assert len(response3.data) == 3
        for i in range(3):
            assert response1.data[i].id == items_slice[i].id
            assert response2.data[i].id == items_slice[i].id
            assert response3.data[i].id == items_slice[i].id
        assert response1.offset == 0
        assert response2.offset == 3
        assert response3.offset == 100
        assert response1.next_offset == 3
        assert response2.next_offset == 6
        assert response3.next_offset == 103

    def test_partial_page_behavior(self, test_items):
        """Test behavior with partial page and correctly applied limit"""
        # Simulating manual pagination by slicing input to from_iterable
        items_slice_1 = test_items[1:4]  # Items 1,2,3 (Limit 3, Offset 1)
        response1 = from_iterable(items_slice_1, limit=3, offset=1)
        assert len(response1.data) == 3
        assert response1.has_more is False  # Cannot determine from slice
        assert response1.next_offset == 4

        items_slice_2 = test_items[4:7]  # Items 4,5,6 (Limit 3, Offset 4)
        response2 = from_iterable(items_slice_2, limit=3, offset=4)
        assert len(response2.data) == 3
        assert response2.has_more is False  # Cannot determine
        assert response2.next_offset == 7

        items_slice_3 = test_items[7:10]  # Items 7,8,9 (Limit 3, Offset 7)
        response3 = from_iterable(items_slice_3, limit=3, offset=7)
        assert len(response3.data) == 3
        assert response3.has_more is False  # Cannot determine
        assert response3.next_offset == 10

        items_slice_4 = test_items[10:]  # Empty slice (Limit 3, Offset 10)
        response4 = from_iterable(items_slice_4, limit=3, offset=10)
        assert len(response4.data) == 0
        assert response4.has_more is False
        assert response4.next_offset == 13

    async def test_async_iterator_handling(self, test_items):
        """Test that async iterators are handled correctly, respecting limit"""
        # --- Simulate fetching the first page ---
        # Create an async generator for the relevant slice for page 1
        # (Fetch limit + 1 to correctly test has_more determination)
        # Here, limit=2, so fetch items 4, 5, 6 (indices 3, 4, 5)
        async_generator_page1 = async_item_generator(test_items[3:6])  # Items 4, 5, 6

        # Get first 2 items with offset=3 (for info)
        response_page1 = await from_async_iterable(
            async_generator_page1, limit=2, offset=3
        )

        # Assertions for page 1
        assert len(response_page1.data) == 2
        assert response_page1.offset == 3
        assert (
            response_page1.has_more is True
        )  # Correctly determined because item 6 was fetched
        assert response_page1.next_offset == 5  # 3 + 2
        assert response_page1.data[0].id == "item-4"
        assert response_page1.data[1].id == "item-5"

        # --- Simulate fetching the second page ---
        # Create a *new* async generator for the relevant slice for page 2
        # Starting from next_offset=5. Fetch limit=3 + 1 = 4 items if possible.
        # Items needed: 6, 7, 8 (indices 5, 6, 7)
        async_generator_page2 = async_item_generator(test_items[5:8])  # Items 6, 7, 8

        # Get next 3 items with offset=5 (for info)
        response_page2 = await from_async_iterable(
            async_generator_page2, limit=3, offset=5
        )

        # Assertions for page 2
        assert len(response_page2.data) == 3  # Should get remaining 3 items
        assert response_page2.offset == 5
        assert response_page2.has_more is False  # Generator provided exactly 3 items
        assert response_page2.next_offset == 8  # 5 + 3
        assert response_page2.data[0].id == "item-6"
        assert response_page2.data[1].id == "item-7"
        assert response_page2.data[2].id == "item-8"

    def test_limit_zero(self, test_items):
        """Test behavior when limit is zero"""
        response = from_iterable(test_items, limit=0, offset=0)
        assert response.limit == 0
        assert response.offset == 0
        assert response.items == 0
        assert response.has_more is False  # Cannot be more if limit is 0
        assert response.next_offset == 0  # offset + limit
        assert len(response.data) == 0

    def test_nested_model_conversion(self, test_items):
        """Test that nested models are properly converted"""

        # Define models locally for test isolation
        class NestedField(VisibleFieldsModel):
            value: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
            secret: str = field(visible_to=[Role.ADMIN])

        class TestWithNested(VisibleFieldsModel):
            id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
            nested: NestedField = field(
                visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN]
            )

        nested_items = [
            TestWithNested(
                id=f"nested-{i}",
                nested=NestedField(value=f"value-{i}", secret=f"secret-{i}"),
            )
            for i in range(1, 4)
        ]

        # Test with VIEWER role
        response_viewer = from_iterable(
            nested_items, limit=3, offset=0, role=Role.VIEWER.value
        )

        assert len(response_viewer.data) == 3
        nested_viewer_response_type = NestedField.create_response_model(
            Role.VIEWER.value
        )
        for item in response_viewer.data:
            assert hasattr(item, "id")
            assert hasattr(item, "nested")
            # --- FIX: Check attributes on the nested instance ---
            assert isinstance(item.nested, nested_viewer_response_type)
            assert hasattr(item.nested, "value")
            assert not hasattr(item.nested, "secret")
            # --- End FIX ---

        # Test with ADMIN role
        response_admin = from_iterable(
            nested_items, limit=3, offset=0, role=Role.ADMIN.value
        )

        assert len(response_admin.data) == 3
        nested_admin_response_type = NestedField.create_response_model(Role.ADMIN.value)
        for item in response_admin.data:
            assert hasattr(item, "id")
            assert hasattr(item, "nested")
            # --- FIX: Check attributes on the nested instance ---
            assert isinstance(item.nested, nested_admin_response_type)
            assert hasattr(item.nested, "value")
            assert hasattr(item.nested, "secret")
            # --- End FIX ---

    def test_edge_case_parameters(self, test_items):
        """Test with edge case parameters"""
        # Large limit
        response = from_iterable(test_items, limit=1000, offset=0)
        assert response.limit == 1000
        assert len(response.data) == 10
        assert response.items == 10
        assert response.has_more is False
        assert response.next_offset == 1000

        # Large offset (input list is empty in this simulated case)
        response = from_iterable([], limit=5, offset=9999)
        assert response.offset == 9999
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == 10004

        # Negative limit (should result in empty data)
        response = from_iterable(test_items, limit=-5, offset=0)
        assert response.limit == -5
        assert len(response.data) == 0
        assert response.items == 0
        assert response.has_more is False
        assert response.next_offset == -5

        # Negative offset (informational only)
        response = from_iterable(test_items[:5], limit=5, offset=-10)
        assert response.offset == -10
        assert len(response.data) == 5
        assert response.items == 5
        assert response.has_more is False
        assert response.next_offset == -5
