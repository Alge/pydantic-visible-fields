"""
Tests for the visible_fields library.

This file contains comprehensive tests for all aspects of the visible_fields library,
including field-level visibility, role inheritance, and complex model structures.
"""

from enum import Enum
from typing import ClassVar, Dict, List, Optional, Set, Union

import pytest
from pydantic import BaseModel, field_validator

from pydantic_visible_fields import (
    VisibleFieldsMixin,
    VisibleFieldsModel,
    configure_roles,
    field,
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


# Test Group 1: Basic Models
# --------------------------


# Model with VisibleFieldsMixin and class-level visibility
class SimpleClassModel(BaseModel, VisibleFieldsMixin):
    """Simple model with class-level field visibility rules"""

    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {
        Role.VIEWER: {"id", "name"},
        Role.EDITOR: {"description"},
        Role.ADMIN: {"secret"},
    }

    id: str
    name: str
    description: str
    secret: str


# Model with VisibleFieldsModel and field-level visibility
class SimpleFieldModel(VisibleFieldsModel):
    """Simple model with field-level visibility rules"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    description: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    secret: str = field(visible_to=[Role.ADMIN])


# Test Group 2: Nested Models
# ---------------------------


# Nested model with class-level visibility
class NestedClassModel(BaseModel, VisibleFieldsMixin):
    """Model that contains a nested model using class-level visibility"""

    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {
        Role.VIEWER: {"id", "title", "simple"},
        Role.EDITOR: {"notes"},
        Role.ADMIN: {"internal_id"},
    }

    id: str
    title: str
    notes: str
    internal_id: str
    simple: SimpleClassModel


# Nested model with field-level visibility
class NestedFieldModel(VisibleFieldsModel):
    """Model that contains a nested model using field-level visibility"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    title: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    notes: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    internal_id: str = field(visible_to=[Role.ADMIN])
    simple: SimpleFieldModel = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])


# Test Group 3: Collections
# -------------------------


# Model with a list of other models
class ListModel(VisibleFieldsModel):
    """Model that contains a list of models"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    items: List[SimpleFieldModel] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN]
    )
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Model with a dictionary of other models
class DictModel(VisibleFieldsModel):
    """Model that contains a dictionary of models"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    mapping: Dict[str, SimpleFieldModel] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN]
    )
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Test Group 4: Discriminated Unions
# ----------------------------------


# Define a type enum
class ItemType(str, Enum):
    BASIC = "basic"
    EXTENDED = "extended"


# Base model for union
class BaseItem(VisibleFieldsModel):
    """Base item class for union testing"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    type: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])


# Basic item
class BasicItem(BaseItem):
    """Basic item type"""

    type: str = ItemType.BASIC
    status: str = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default="active"
    )


# Extended item
class ExtendedItem(BaseItem):
    """Extended item with additional fields"""

    type: str = ItemType.EXTENDED
    target: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Union type
ItemUnion = Union[BasicItem, ExtendedItem]


# Container with union
class ContainerWithUnion(VisibleFieldsModel):
    """Model that uses a union type"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    owner: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    item: Optional[ItemUnion] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default=None
    )
    tags: List[str] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default=["default"]
    )
    active: bool = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default=True
    )


# Test Group 5: Deep Nesting
# --------------------------


class DeepChild(VisibleFieldsModel):
    """Deeply nested child model"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    value: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


class DeepParent(VisibleFieldsModel):
    """Parent with deep child"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    child: DeepChild = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    data: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


class DeepContainer(VisibleFieldsModel):
    """Container with deep nesting"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    parent: DeepParent = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    items: List[DeepParent] = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    mapped_items: Dict[str, DeepParent] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN]
    )
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Test Group 6: Tree Structure
# ----------------------------


class TreeNode(VisibleFieldsModel):
    """Tree node with parent/child references by ID"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    data: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    parent_id: Optional[str] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default=None
    )
    children_ids: List[str] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default_factory=list
    )
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Test Group 7: Validation
# -----------------------


class ValidatedModel(VisibleFieldsModel):
    """Model with field validators"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    email: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    count: int = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    internal_code: str = field(visible_to=[Role.EDITOR, Role.ADMIN])

    @field_validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator("count")
    def validate_count(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Count must be between 0 and 100")
        return v


# Test Group 8: Field Aliases
# --------------------------


class ModelWithAliases(VisibleFieldsModel):
    """Model with field aliases"""

    item_id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], alias="id")
    item_name: str = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], alias="name"
    )
    internal_code: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


# Test Group 9: Circular References
# --------------------------------


class NodeWithSelfReference(VisibleFieldsModel):
    """Node that can reference itself"""

    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    self_ref: Optional["NodeWithSelfReference"] = field(
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default=None
    )
    metadata: str = field(visible_to=[Role.EDITOR, Role.ADMIN])


NodeWithSelfReference.model_rebuild()


# Test Fixtures
# ------------


@pytest.fixture
def simple_class_model():
    return SimpleClassModel(
        id="sc1",
        name="Class Model",
        description="Description for class model",
        secret="class-secret",
    )


@pytest.fixture
def simple_field_model():
    return SimpleFieldModel(
        id="sf1",
        name="Field Model",
        description="Description for field model",
        secret="field-secret",
    )


@pytest.fixture
def nested_class_model(simple_class_model):
    return NestedClassModel(
        id="nc1",
        title="Nested Class Title",
        notes="Notes for nested class model",
        internal_id="nc-internal",
        simple=simple_class_model,
    )


@pytest.fixture
def nested_field_model(simple_field_model):
    return NestedFieldModel(
        id="nf1",
        title="Nested Field Title",
        notes="Notes for nested field model",
        internal_id="nf-internal",
        simple=simple_field_model,
    )


@pytest.fixture
def list_model(simple_field_model):
    return ListModel(
        id="l1",
        items=[
            simple_field_model,
            SimpleFieldModel(
                id="sf2",
                name="Another Field Model",
                description="Another description",
                secret="another-secret",
            ),
        ],
        metadata="List metadata",
    )


@pytest.fixture
def dict_model(simple_field_model):
    return DictModel(
        id="d1",
        mapping={
            "first": simple_field_model,
            "second": SimpleFieldModel(
                id="sf3",
                name="Dict Field Model",
                description="Dict description",
                secret="dict-secret",
            ),
        },
        metadata="Dict metadata",
    )


@pytest.fixture
def model_with_basic_item():
    return ContainerWithUnion(
        id="cb1", name="Basic Container", owner="owner1", item=BasicItem(id="bi1")
    )


@pytest.fixture
def model_with_extended_item():
    return ContainerWithUnion(
        id="ce1",
        name="Extended Container",
        owner="owner2",
        item=ExtendedItem(id="ei1", target="target1", metadata="Item metadata"),
    )


@pytest.fixture
def deep_nested_model():
    return DeepContainer(
        id="dc1",
        parent=DeepParent(
            id="dp1",
            child=DeepChild(
                id="dch1", value="Deep child value", metadata="Child metadata"
            ),
            data="Parent data",
            metadata="Parent metadata",
        ),
        items=[
            DeepParent(
                id="dp2",
                child=DeepChild(
                    id="dch2", value="List child value", metadata="List child metadata"
                ),
                data="List parent data",
                metadata="List parent metadata",
            ),
            DeepParent(
                id="dp3",
                child=DeepChild(
                    id="dch3",
                    value="Another list child value",
                    metadata="Another list child metadata",
                ),
                data="Another list parent data",
                metadata="Another list parent metadata",
            ),
        ],
        mapped_items={
            "first": DeepParent(
                id="dp4",
                child=DeepChild(
                    id="dch4", value="Dict child value", metadata="Dict child metadata"
                ),
                data="Dict parent data",
                metadata="Dict parent metadata",
            ),
            "second": DeepParent(
                id="dp5",
                child=DeepChild(
                    id="dch5",
                    value="Another dict child value",
                    metadata="Another dict child metadata",
                ),
                data="Another dict parent data",
                metadata="Another dict parent metadata",
            ),
        },
        metadata="Container metadata",
    )


@pytest.fixture
def tree_structure():
    parent = TreeNode(
        id="tn1",
        data="Parent node data",
        children_ids=["tn2", "tn3"],
        metadata="Parent metadata",
    )

    child1 = TreeNode(
        id="tn2", data="Child 1 data", parent_id="tn1", metadata="Child 1 metadata"
    )

    child2 = TreeNode(
        id="tn3", data="Child 2 data", parent_id="tn1", metadata="Child 2 metadata"
    )

    return {"parent": parent, "child1": child1, "child2": child2}


@pytest.fixture
def validated_model():
    return ValidatedModel(
        id="vm1", email="test@example.com", count=50, internal_code="vm-internal"
    )


@pytest.fixture
def aliased_model():
    return ModelWithAliases(id="am1", name="Aliased Model", internal_code="am-internal")


@pytest.fixture
def circular_reference():
    node1 = NodeWithSelfReference(id="nr1", name="Node 1", metadata="Node 1 metadata")

    node2 = NodeWithSelfReference(
        id="nr2", name="Node 2", self_ref=node1, metadata="Node 2 metadata"
    )

    # Create circular reference
    node1.self_ref = node2

    return node1


@pytest.fixture
def empty_list_model():
    return ListModel(id="el1", items=[], metadata="Empty list metadata")


# Test Cases
# ----------

"""
Additional tests for the get_role_from_request function.

This file contains comprehensive tests for the get_role_from_request function
in the visible_fields library, covering various edge cases and scenarios.
"""


# Define the Role enum for testing (same as in the main test file)
class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class TestVisibleFields:
    """Tests for the VisibleFieldsMixin class"""

    def test_class_model_visibility(self, simple_class_model):
        """Test that class-level visibility works correctly"""
        # VIEWER should see id, name
        viewer_dict = simple_class_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}
        assert "description" not in viewer_dict
        assert "secret" not in viewer_dict

        # EDITOR should see id, name, description
        editor_dict = simple_class_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}
        assert "secret" not in editor_dict

        # ADMIN should see all fields
        admin_dict = simple_class_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}

    def test_field_model_visibility(self, simple_field_model):
        """Test that field-level visibility works correctly"""
        # VIEWER should see id, name
        viewer_dict = simple_field_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}
        assert "description" not in viewer_dict
        assert "secret" not in viewer_dict

        # EDITOR should see id, name, description
        editor_dict = simple_field_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}
        assert "secret" not in editor_dict

        # ADMIN should see all fields
        admin_dict = simple_field_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}

    def test_nested_class_model(self, nested_class_model):
        """Test nested models with class-level visibility"""
        # Get VIEWER view
        viewer_dict = nested_class_model.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "title", "simple"}
        assert "notes" not in viewer_dict
        assert "internal_id" not in viewer_dict

        # Check nested model fields
        simple_dict = viewer_dict["simple"]
        assert set(simple_dict.keys()) == {"id", "name"}
        assert "description" not in simple_dict
        assert "secret" not in simple_dict

    def test_nested_field_model(self, nested_field_model):
        """Test nested models with field-level visibility"""
        # Get VIEWER view
        viewer_dict = nested_field_model.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "title", "simple"}
        assert "notes" not in viewer_dict
        assert "internal_id" not in viewer_dict

        # Check nested model fields
        simple_dict = viewer_dict["simple"]
        assert set(simple_dict.keys()) == {"id", "name"}
        assert "description" not in simple_dict
        assert "secret" not in simple_dict

    def test_list_model(self, list_model):
        """Test models with list of other models"""
        # Get VIEWER view
        viewer_dict = list_model.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "items"}
        assert "metadata" not in viewer_dict

        # Check list items
        items = viewer_dict["items"]
        assert len(items) == 2
        assert all(isinstance(item, dict) for item in items)
        assert all(set(item.keys()) == {"id", "name"} for item in items)
        assert all("description" not in item for item in items)
        assert all("secret" not in item for item in items)

        # Now test with to_response_model
        response = list_model.to_response_model(Role.VIEWER.value)
        assert response.id == list_model.id
        assert not hasattr(response, "metadata")

        # The items should be a list of dicts
        assert isinstance(response.items, list)
        assert len(response.items) == 2
        assert all(isinstance(item, dict) for item in response.items)
        assert all("id" in item for item in response.items)
        assert all("name" in item for item in response.items)
        assert all("description" not in item for item in response.items)
        assert all("secret" not in item for item in response.items)

    def test_dict_model(self, dict_model):
        """Test models with dictionary of other models"""
        # Get VIEWER view
        viewer_dict = dict_model.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "mapping"}
        assert "metadata" not in viewer_dict

        # Check dict values
        mapping = viewer_dict["mapping"]
        assert len(mapping) == 2
        assert all(isinstance(v, dict) for v in mapping.values())
        assert all(set(v.keys()) == {"id", "name"} for v in mapping.values())
        assert all("description" not in v for v in mapping.values())
        assert all("secret" not in v for v in mapping.values())

        # Now test with to_response_model
        response = dict_model.to_response_model(Role.VIEWER.value)
        assert response.id == dict_model.id
        assert not hasattr(response, "metadata")

        # The mapping should be a dict of dicts
        assert isinstance(response.mapping, dict)
        assert len(response.mapping) == 2
        assert "first" in response.mapping
        assert "second" in response.mapping
        assert all(isinstance(v, dict) for v in response.mapping.values())
        assert all("id" in v for v in response.mapping.values())
        assert all("name" in v for v in response.mapping.values())
        assert all("description" not in v for v in response.mapping.values())
        assert all("secret" not in v for v in response.mapping.values())

    def test_union_basic_item(self, model_with_basic_item):
        """Test with a discriminated union (basic item)"""
        # Get VIEWER view
        viewer_dict = model_with_basic_item.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "name", "item", "tags", "active"}
        assert "owner" not in viewer_dict

        # Check union field
        item = viewer_dict["item"]
        assert isinstance(item, dict)
        assert item["type"] == ItemType.BASIC.value
        assert item["status"] == "active"
        assert set(item.keys()) == {"id", "type", "status"}

        # Now test with to_response_model
        response = model_with_basic_item.to_response_model(Role.VIEWER.value)
        assert response.id == model_with_basic_item.id
        assert response.name == model_with_basic_item.name
        assert not hasattr(response, "owner")

        # The item should be a dict
        assert isinstance(response.item, dict)
        assert response.item["type"] == ItemType.BASIC.value
        assert response.item["status"] == "active"

    def test_union_extended_item(self, model_with_extended_item):
        """Test with a discriminated union (extended item)"""
        # Get VIEWER view
        viewer_dict = model_with_extended_item.visible_dict(Role.VIEWER.value)

        # Check top level fields
        assert set(viewer_dict.keys()) == {"id", "name", "item", "tags", "active"}
        assert "owner" not in viewer_dict

        # Check union field
        item = viewer_dict["item"]
        assert isinstance(item, dict)
        assert item["type"] == ItemType.EXTENDED.value
        assert item["target"] == "target1"
        assert "metadata" not in item
        assert set(item.keys()) == {"id", "type", "target"}

        # EDITOR should see additional fields
        editor_dict = model_with_extended_item.visible_dict(Role.EDITOR.value)
        item = editor_dict["item"]
        assert item["metadata"] == "Item metadata"
        assert set(item.keys()) == {"id", "type", "target", "metadata"}

    def test_deep_nesting(self, deep_nested_model):
        """Test deeply nested models"""
        # Get VIEWER view
        viewer_dict = deep_nested_model.visible_dict(Role.VIEWER.value)

        # Check top level
        assert set(viewer_dict.keys()) == {"id", "parent", "items", "mapped_items"}
        assert "metadata" not in viewer_dict

        # Check parent level
        parent = viewer_dict["parent"]
        assert set(parent.keys()) == {"id", "child", "data"}
        assert "metadata" not in parent

        # Check child level
        child = parent["child"]
        assert set(child.keys()) == {"id", "value"}
        assert "metadata" not in child

        # Check list items
        assert len(viewer_dict["items"]) == 2
        list_item = viewer_dict["items"][0]
        assert set(list_item.keys()) == {"id", "child", "data"}
        assert "metadata" not in list_item
        assert set(list_item["child"].keys()) == {"id", "value"}
        assert "metadata" not in list_item["child"]

        # Check dict items
        assert len(viewer_dict["mapped_items"]) == 2
        dict_item = viewer_dict["mapped_items"]["first"]
        assert set(dict_item.keys()) == {"id", "child", "data"}
        assert "metadata" not in dict_item
        assert set(dict_item["child"].keys()) == {"id", "value"}
        assert "metadata" not in dict_item["child"]

    def test_tree_structure(self, tree_structure):
        """Test tree structure with parent/child references"""
        parent = tree_structure["parent"]
        child1 = tree_structure["child1"]

        # Get VIEWER view for parent
        parent_dict = parent.visible_dict(Role.VIEWER.value)

        # Check parent fields
        assert set(parent_dict.keys()) == {"id", "data", "parent_id", "children_ids"}
        assert "metadata" not in parent_dict
        assert parent_dict["parent_id"] is None
        assert set(parent_dict["children_ids"]) == {"tn2", "tn3"}

        # Get VIEWER view for child
        child_dict = child1.visible_dict(Role.VIEWER.value)

        # Check child fields
        assert set(child_dict.keys()) == {"id", "data", "parent_id", "children_ids"}
        assert "metadata" not in child_dict
        assert child_dict["parent_id"] == "tn1"
        assert child_dict["children_ids"] == []

    def test_validated_model(self, validated_model):
        """Test models with field validators"""
        # Get VIEWER view
        viewer_dict = validated_model.visible_dict(Role.VIEWER.value)

        # Check fields
        assert set(viewer_dict.keys()) == {"id", "email", "count"}
        assert "internal_code" not in viewer_dict

        # Now test with to_response_model
        response = validated_model.to_response_model(Role.VIEWER.value)
        assert response.id == validated_model.id
        assert response.email == validated_model.email
        assert response.count == validated_model.count
        assert not hasattr(response, "internal_code")

    def test_field_aliases(self, aliased_model):
        """Test models with field aliases"""
        # Get VIEWER view - should use actual field names, not aliases
        viewer_dict = aliased_model.visible_dict(Role.VIEWER.value)

        # Check fields
        assert set(viewer_dict.keys()) == {"item_id", "item_name"}
        assert "internal_code" not in viewer_dict

        # Now test with to_response_model
        response = aliased_model.to_response_model(Role.VIEWER.value)
        assert response.item_id == aliased_model.item_id
        assert response.item_name == aliased_model.item_name
        assert not hasattr(response, "internal_code")

    def test_circular_reference(self, circular_reference):
        """Test models with circular references"""
        # Get VIEWER view
        viewer_dict = circular_reference.visible_dict(Role.VIEWER.value)

        # Check fields
        assert set(viewer_dict.keys()) == {"id", "name", "self_ref"}
        assert "metadata" not in viewer_dict

        # Check self reference
        self_ref = viewer_dict["self_ref"]
        assert isinstance(self_ref, dict)
        assert self_ref["id"] == "nr2"
        assert self_ref["name"] == "Node 2"

        # Check for cycle detection
        nested_ref = self_ref["self_ref"]
        assert isinstance(nested_ref, dict)
        assert nested_ref["id"] == "nr1"

        # In a cycle, we should detect and not recurse infinitely
        assert "__cycle_reference__" in nested_ref or len(nested_ref) <= 2

    def test_empty_collections(self, empty_list_model):
        """Test with empty collections"""
        # Get VIEWER view
        viewer_dict = empty_list_model.visible_dict(Role.VIEWER.value)
        assert viewer_dict["items"] == []

        # Now test with to_response_model
        response = empty_list_model.to_response_model(Role.VIEWER.value)
        assert response.id == empty_list_model.id
        assert response.items == []

    def test_role_inheritance(self, simple_field_model):
        """Test that role inheritance works correctly"""
        # ADMIN inherits from EDITOR, which inherits from VIEWER
        # So ADMIN should see all fields visible to any role
        admin_dict = simple_field_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}

        # EDITOR inherits from VIEWER, so should see all VIEWER fields
        editor_dict = simple_field_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}

        # VIEWER sees only its own fields
        viewer_dict = simple_field_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}

    def test_response_model_creation(self, simple_field_model):
        """Test response model creation"""
        # Create a VIEWER response model
        viewer_response = simple_field_model.to_response_model(Role.VIEWER.value)
        assert hasattr(viewer_response, "id")
        assert hasattr(viewer_response, "name")
        assert not hasattr(viewer_response, "description")
        assert not hasattr(viewer_response, "secret")

        # Create an EDITOR response model
        editor_response = simple_field_model.to_response_model(Role.EDITOR.value)
        assert hasattr(editor_response, "id")
        assert hasattr(editor_response, "name")
        assert hasattr(editor_response, "description")
        assert not hasattr(editor_response, "secret")

    def test_response_model_caching(self):
        """Test that response models are cached for performance"""
        # Create response models
        model1 = SimpleFieldModel.create_response_model(Role.VIEWER.value)
        model2 = SimpleFieldModel.create_response_model(Role.VIEWER.value)

        # They should be the same object (cached)
        assert model1 is model2

    def test_configure_visibility(self, simple_class_model):
        """Test dynamic configuration of visibility"""
        # First verify the initial visible fields
        assert SimpleClassModel._get_all_visible_fields(Role.VIEWER.value) == {
            "id",
            "name",
        }

        # Configure visibility for VIEWER role
        SimpleClassModel.configure_visibility(
            Role.VIEWER.value, {"id", "name", "new_field"}
        )
        assert SimpleClassModel._get_all_visible_fields(Role.VIEWER.value) == {
            "id",
            "name",
            "new_field",
        }

        # Configure visibility back to original
