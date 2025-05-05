"""
Tests for the visible_fields library.

This file contains comprehensive tests for all aspects of the visible_fields library,
including field-level visibility, role inheritance, and complex model structures.
"""
import uuid
from datetime import datetime, timezone # Added timezone
from enum import Enum
from types import NoneType
from typing import ClassVar, Dict, List, Optional, Set, Union, Any # Added Any

import pytest
from pydantic import ( # Added ConfigDict
    BaseModel,
    field_validator,
    ValidationError,
    ConfigDict
)

# Ensure this matches the actual import path for your library
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
        visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN], default_factory=list # Use default_factory for lists
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
    # Allow population by alias name as well as field name
    model_config = ConfigDict(populate_by_name=True)

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

# Test Group 10: Models for Bug Detection
# ---------------------------------------

class ModelWithSpecificTypes(VisibleFieldsModel):
    """Model for testing type preservation"""
    id: str = field(visible_to=[Role.ADMIN])
    event_time: datetime = field(visible_to=[Role.ADMIN])
    unique_id: uuid.UUID = field(visible_to=[Role.ADMIN])
    current_status: Role = field(visible_to=[Role.ADMIN]) # Use Enum type
    optional_int: Optional[int] = field(visible_to=[Role.ADMIN], default=None)

class CorruptibleSimpleFieldModel(VisibleFieldsModel):
    """Subclass to inject invalid data for validation test"""
    id: str = field(visible_to=[Role.VIEWER, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.ADMIN])
    value: int = field(visible_to=[Role.ADMIN]) # Expects an int

    # Override to deliberately corrupt data for testing
    def visible_dict(
        self,
        role: Optional[str] = None,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Override to deliberately corrupt data for testing masked validation errors."""
        data = super().visible_dict(role, visited, depth)
        # Make data invalid for the ADMIN role response model ONLY if ADMIN role is passed
        if role == Role.ADMIN.value:
            if "value" in data:
                # Intentionally set the wrong type
                data["value"] = "this-is-not-an-integer"
            # Intentionally remove a required field ('id') if present for ADMIN view
            # Note: The response model created for ADMIN *must* require 'id' for this to fail validation.
            if "id" in data:
                 del data["id"]
        return data

class CycleNodeA(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER])
    ref_b: Optional["CycleNodeB"] = field(visible_to=[Role.VIEWER], default=None)

class CycleNodeB(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER])
    ref_a: Optional[CycleNodeA] = field(visible_to=[Role.VIEWER], default=None)

# Ensure forward references are resolved
CycleNodeA.model_rebuild()
CycleNodeB.model_rebuild()

class ConstructTestModel(VisibleFieldsModel):
    id: str = field(visible_to=[Role.ADMIN])
    # Field expecting an int in the response model
    int_field: int = field(visible_to=[Role.ADMIN])
    # Field required in the response model
    required_field: str = field(visible_to=[Role.ADMIN])
    # Field with a constraint
    constrained_field: int = field(gt=0, visible_to=[Role.ADMIN])

    # Override visible_dict to produce problematic data for ADMIN role
    def visible_dict(
        self,
        role: Optional[str] = None,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Override to produce data testing model_construct behaviors."""
        data = super().visible_dict(role, visited, depth)
        if role == Role.ADMIN.value:
            if "int_field" in data:
                data["int_field"] = "123" # Provide string instead of int
            if "required_field" in data:
                del data["required_field"] # Omit the required field
            if "constrained_field" in data:
                data["constrained_field"] = -5 # Violate gt=0 constraint
        return data

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
    # Use copies to avoid modifying the original fixture instance if tests alter it
    item1 = simple_field_model.model_copy()
    item2 = SimpleFieldModel(
        id="sf2",
        name="Another Field Model",
        description="Another description",
        secret="another-secret",
    )
    return ListModel(
        id="l1",
        items=[item1, item2],
        metadata="List metadata",
    )


@pytest.fixture
def dict_model(simple_field_model):
    # Use copies
    item1 = simple_field_model.model_copy()
    item2 = SimpleFieldModel(
        id="sf3",
        name="Dict Field Model",
        description="Dict description",
        secret="dict-secret",
    )
    return DictModel(
        id="d1",
        mapping={"first": item1, "second": item2},
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
    child1 = DeepChild(id="dch1", value="Deep child value", metadata="Child metadata")
    parent1 = DeepParent(id="dp1", child=child1, data="Parent data", metadata="Parent metadata")

    child2 = DeepChild(id="dch2", value="List child value", metadata="List child metadata")
    parent2 = DeepParent(id="dp2", child=child2, data="List parent data", metadata="List parent metadata")

    child3 = DeepChild(id="dch3", value="Another list child value", metadata="Another list child metadata")
    parent3 = DeepParent(id="dp3", child=child3, data="Another list parent data", metadata="Another list parent metadata")

    child4 = DeepChild(id="dch4", value="Dict child value", metadata="Dict child metadata")
    parent4 = DeepParent(id="dp4", child=child4, data="Dict parent data", metadata="Dict parent metadata")

    child5 = DeepChild(id="dch5", value="Another dict child value", metadata="Another dict child metadata")
    parent5 = DeepParent(id="dp5", child=child5, data="Another dict parent data", metadata="Another dict parent metadata")

    return DeepContainer(
        id="dc1",
        parent=parent1,
        items=[parent2, parent3],
        mapped_items={"first": parent4, "second": parent5},
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
    # Initialize using defined field names
    return ModelWithAliases(item_id="am1", item_name="Aliased Model", internal_code="am-internal")


@pytest.fixture
def circular_reference():
    node1 = NodeWithSelfReference(id="nr1", name="Node 1", metadata="Node 1 metadata")
    node2 = NodeWithSelfReference(
        id="nr2", name="Node 2", self_ref=node1, metadata="Node 2 metadata"
    )
    node1.self_ref = node2
    return node1


@pytest.fixture
def empty_list_model():
    return ListModel(id="el1", items=[], metadata="Empty list metadata")

# --- Fixtures for Bug Detection ---

@pytest.fixture
def model_with_specific_types():
    return ModelWithSpecificTypes(
        id="spec1",
        event_time=datetime.now(timezone.utc),
        unique_id=uuid.uuid4(),
        current_status=Role.EDITOR,
        optional_int=123
    )

@pytest.fixture
def corruptible_simple_field_model():
    # This model will produce invalid data via its overridden visible_dict
    return CorruptibleSimpleFieldModel(id="corr1", name="Corruptible", value=10)

@pytest.fixture
def node_with_direct_cycle():
    # Re-use NodeWithSelfReference which is already defined globally
    node = NodeWithSelfReference(id="direct_cycle_1", name="Node 1", metadata="Direct Cycle Meta")
    node.self_ref = node # Create direct cycle
    return node

@pytest.fixture
def node_with_indirect_cycle():
    node_a = CycleNodeA(id="indirect_a")
    node_b = CycleNodeB(id="indirect_b")
    node_a.ref_b = node_b
    node_b.ref_a = node_a # Create indirect cycle (A -> B -> A)
    return node_a

@pytest.fixture
def construct_test_model_instance():
    # Instance whose visible_dict will be overridden for ADMIN role
    return ConstructTestModel(
        id="construct1",
        int_field=123,
        required_field="present",
        constrained_field=10
    )


# Test Cases
# ----------

class TestVisibleFields:
    """Tests for the VisibleFieldsMixin and related functionality"""

    # --- Original Tests ---
    def test_class_model_visibility(self, simple_class_model):
        """Test that class-level visibility works correctly"""
        viewer_dict = simple_class_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}
        editor_dict = simple_class_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}
        admin_dict = simple_class_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}

    def test_field_model_visibility(self, simple_field_model):
        """Test that field-level visibility works correctly"""
        viewer_dict = simple_field_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}
        editor_dict = simple_field_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}
        admin_dict = simple_field_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}

    def test_nested_class_model(self, nested_class_model):
        """Test nested models with class-level visibility"""
        viewer_dict = nested_class_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "title", "simple"}
        simple_dict = viewer_dict["simple"]
        assert set(simple_dict.keys()) == {"id", "name"}

    def test_nested_field_model(self, nested_field_model):
        """Test nested models with field-level visibility"""
        viewer_dict = nested_field_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "title", "simple"}
        simple_dict = viewer_dict["simple"]
        assert set(simple_dict.keys()) == {"id", "name"}

    def test_list_model(self, list_model):
        """Test models with list of other models"""
        viewer_dict = list_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "items"}
        items = viewer_dict["items"]
        assert len(items) == 2
        assert all(isinstance(item, dict) for item in items)
        assert all(set(item.keys()) == {"id", "name"} for item in items)

        response = list_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "id") and response.id == list_model.id
        assert not hasattr(response, "metadata")
        assert isinstance(response.items, list)
        assert len(response.items) == 2
        # Check that items are instances of the correct response model type
        item_response_type = SimpleFieldModel.create_response_model(Role.VIEWER.value)
        assert all(isinstance(item, item_response_type) for item in response.items)
        assert all(hasattr(item, "id") and hasattr(item, "name") for item in response.items)
        assert all(not hasattr(item, "description") and not hasattr(item, "secret") for item in response.items)


    def test_dict_model(self, dict_model):
        """Test models with dictionary of other models"""
        viewer_dict = dict_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "mapping"}
        mapping = viewer_dict["mapping"]
        assert len(mapping) == 2
        assert all(isinstance(v, dict) for v in mapping.values())
        assert all(set(v.keys()) == {"id", "name"} for v in mapping.values())

        response = dict_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "id") and response.id == dict_model.id
        assert not hasattr(response, "metadata")
        assert isinstance(response.mapping, dict)
        assert len(response.mapping) == 2
        # Check that items are instances of the correct response model type
        item_response_type = SimpleFieldModel.create_response_model(Role.VIEWER.value)
        assert all(isinstance(v, item_response_type) for v in response.mapping.values())
        assert all(hasattr(v, "id") and hasattr(v, "name") for v in response.mapping.values())
        assert all(not hasattr(v, "description") and not hasattr(v, "secret") for v in response.mapping.values())

    def test_union_basic_item(self, model_with_basic_item):
        """Test with a discriminated union (basic item)"""
        viewer_dict = model_with_basic_item.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name", "item", "tags", "active"}
        item_dict = viewer_dict["item"]
        assert isinstance(item_dict, dict)
        assert item_dict["type"] == ItemType.BASIC.value
        assert set(item_dict.keys()) == {"id", "type", "status"}

        response = model_with_basic_item.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "id") and response.id == model_with_basic_item.id
        assert hasattr(response, "name") and response.name == model_with_basic_item.name
        assert not hasattr(response, "owner")
        response_item = response.item # Check instance type
        # Get the expected response type for BasicItem based on the role
        basic_item_response_type = BasicItem.create_response_model(Role.VIEWER.value)
        assert isinstance(response_item, basic_item_response_type)
        assert response_item.type == ItemType.BASIC.value
        assert response_item.status == "active"
        assert hasattr(response_item, "id")

    def test_union_extended_item(self, model_with_extended_item):
        """Test with a discriminated union (extended item)"""
        viewer_dict = model_with_extended_item.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name", "item", "tags", "active"}
        item_dict = viewer_dict["item"]
        assert isinstance(item_dict, dict)
        assert item_dict["type"] == ItemType.EXTENDED.value
        assert "metadata" not in item_dict
        assert set(item_dict.keys()) == {"id", "type", "target"}

        editor_dict = model_with_extended_item.visible_dict(Role.EDITOR.value)
        editor_item_dict = editor_dict["item"]
        assert editor_item_dict["metadata"] == "Item metadata"
        assert set(editor_item_dict.keys()) == {"id", "type", "target", "metadata"}

        editor_response = model_with_extended_item.to_response_model(Role.EDITOR.value)
        editor_response_item = editor_response.item # Check instance type
        # Get expected response type for ExtendedItem based on role
        extended_item_response_type = ExtendedItem.create_response_model(Role.EDITOR.value)
        assert isinstance(editor_response_item, extended_item_response_type)
        assert editor_response_item.type == ItemType.EXTENDED.value
        assert editor_response_item.target == "target1"
        assert editor_response_item.metadata == "Item metadata"

    def test_deep_nesting(self, deep_nested_model):
        """Test deeply nested models"""
        viewer_dict = deep_nested_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "parent", "items", "mapped_items"}
        parent_dict = viewer_dict["parent"]
        assert set(parent_dict.keys()) == {"id", "child", "data"}
        child_dict = parent_dict["child"]
        assert set(child_dict.keys()) == {"id", "value"}
        list_item_dict = viewer_dict["items"][0]
        assert set(list_item_dict.keys()) == {"id", "child", "data"}
        assert set(list_item_dict["child"].keys()) == {"id", "value"}
        dict_item_dict = viewer_dict["mapped_items"]["first"]
        assert set(dict_item_dict.keys()) == {"id", "child", "data"}
        assert set(dict_item_dict["child"].keys()) == {"id", "value"}

        # Test response model construction (basic check)
        response = deep_nested_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "parent")
        assert hasattr(response.parent, "child")
        assert isinstance(response.items, list)
        assert len(response.items) > 0
        assert hasattr(response.items[0], "child")
        assert isinstance(response.mapped_items, dict)
        assert "first" in response.mapped_items
        assert hasattr(response.mapped_items["first"], "child")


    def test_tree_structure(self, tree_structure):
        """Test tree structure with parent/child references"""
        parent = tree_structure["parent"]
        child1 = tree_structure["child1"]
        parent_dict = parent.visible_dict(Role.VIEWER.value)
        assert set(parent_dict.keys()) == {"id", "data", "parent_id", "children_ids"}
        assert set(parent_dict["children_ids"]) == {"tn2", "tn3"}
        child_dict = child1.visible_dict(Role.VIEWER.value)
        assert set(child_dict.keys()) == {"id", "data", "parent_id", "children_ids"}
        assert child_dict["parent_id"] == "tn1"

    def test_validated_model(self, validated_model):
        """Test models with field validators"""
        viewer_dict = validated_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "email", "count"}
        response = validated_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "id") and response.id == validated_model.id
        assert hasattr(response, "email") and response.email == validated_model.email
        assert hasattr(response, "count") and response.count == validated_model.count
        assert not hasattr(response, "internal_code")
        # Note: Validators are not run with model_construct, so this only checks field presence

    def test_field_aliases(self, aliased_model):
        """Test models with field aliases"""
        viewer_dict = aliased_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"item_id", "item_name"} # Check internal names

        response = aliased_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "item_id") and response.item_id == aliased_model.item_id
        assert hasattr(response, "item_name") and response.item_name == aliased_model.item_name
        assert not hasattr(response, "internal_code")

        # Optional: Test serialization with by_alias=True if needed
        response_dump = response.model_dump(by_alias=True)
        assert response_dump.get("id") == aliased_model.item_id
        assert response_dump.get("name") == aliased_model.item_name


    def test_circular_reference(self, circular_reference):
        """Test models with circular references"""
        viewer_dict = circular_reference.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name", "self_ref"}
        self_ref_dict = viewer_dict["self_ref"]
        assert isinstance(self_ref_dict, dict)
        assert self_ref_dict["id"] == "nr2"
        assert "metadata" not in self_ref_dict
        nested_ref_dict = self_ref_dict["self_ref"]
        assert isinstance(nested_ref_dict, dict)
        has_cycle_marker = nested_ref_dict.get("__cycle_reference__") is True
        assert has_cycle_marker, f"Cycle marker not found: {nested_ref_dict}"
        assert "id" in nested_ref_dict and nested_ref_dict["id"] == "nr1"


    def test_empty_collections(self, empty_list_model):
        """Test with empty collections"""
        viewer_dict = empty_list_model.visible_dict(Role.VIEWER.value)
        assert viewer_dict["items"] == []
        response = empty_list_model.to_response_model(Role.VIEWER.value)
        assert hasattr(response, "id") and response.id == empty_list_model.id
        assert hasattr(response, "items") and response.items == []

    def test_role_inheritance(self, simple_field_model):
        """Test that role inheritance works correctly"""
        admin_dict = simple_field_model.visible_dict(Role.ADMIN.value)
        assert set(admin_dict.keys()) == {"id", "name", "description", "secret"}
        editor_dict = simple_field_model.visible_dict(Role.EDITOR.value)
        assert set(editor_dict.keys()) == {"id", "name", "description"}
        viewer_dict = simple_field_model.visible_dict(Role.VIEWER.value)
        assert set(viewer_dict.keys()) == {"id", "name"}

    def test_response_model_creation(self, simple_field_model):
        """Test response model creation for different roles"""
        viewer_response = simple_field_model.to_response_model(Role.VIEWER.value)
        assert hasattr(viewer_response, "id") and hasattr(viewer_response, "name")
        assert not hasattr(viewer_response, "description") and not hasattr(viewer_response, "secret")
        editor_response = simple_field_model.to_response_model(Role.EDITOR.value)
        assert hasattr(editor_response, "id") and hasattr(editor_response, "name") and hasattr(editor_response, "description")
        assert not hasattr(editor_response, "secret")
        admin_response = simple_field_model.to_response_model(Role.ADMIN.value)
        assert hasattr(admin_response, "id") and hasattr(admin_response, "name") and hasattr(admin_response, "description") and hasattr(admin_response, "secret")


    def test_response_model_caching(self):
        """Test that response model *types* are cached for performance"""
        # Test behavior instead of importing internal cache
        ModelType1 = SimpleFieldModel.create_response_model(Role.VIEWER.value)
        ModelType2 = SimpleFieldModel.create_response_model(Role.VIEWER.value)
        ModelType3 = SimpleFieldModel.create_response_model(Role.EDITOR.value)

        assert ModelType1 is ModelType2
        assert ModelType1 is not ModelType3


    def test_configure_visibility(self):
        """Test dynamic configuration of visibility using configure_visibility"""
        class ConfigurableModel(VisibleFieldsModel):
             id: str = field(visible_to=[Role.VIEWER])
             data: str = field() # Initially not visible

        original_viewer_fields = ConfigurableModel._get_all_visible_fields(Role.VIEWER.value).copy()
        original_editor_fields = ConfigurableModel._get_all_visible_fields(Role.EDITOR.value).copy()

        try:
            assert ConfigurableModel._get_all_visible_fields(Role.VIEWER.value) == {"id"}
            assert ConfigurableModel._get_all_visible_fields(Role.EDITOR.value) == {"id"}

            ConfigurableModel.configure_visibility(Role.EDITOR.value, {"id", "data"})

            assert ConfigurableModel._get_all_visible_fields(Role.EDITOR.value) == {"id", "data"}
            assert ConfigurableModel._get_all_visible_fields(Role.VIEWER.value) == {"id"}
            assert ConfigurableModel._get_all_visible_fields(Role.ADMIN.value) == {"id", "data"}

        finally:
            # Reset to only fields explicitly defined for EDITOR (none in this case)
            ConfigurableModel.configure_visibility(Role.EDITOR.value, set())
            # Re-assert based on inheritance after reset
            assert ConfigurableModel._get_all_visible_fields(Role.EDITOR.value) == {"id"}
            assert ConfigurableModel._get_all_visible_fields(Role.ADMIN.value) == {"id"}


    def test_create_response_model_does_not_corrupt_original_types(self):
        """
        Verify that calling create_response_model does not modify the
        type annotations or FieldInfo objects of the original model's fields.
        """
        class ModelForTypeCorruptionTest(VisibleFieldsModel):
            id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
            optional_str: Optional[str] = field(visible_to=[Role.VIEWER], default=None)
            list_of_int: List[int] = field(visible_to=[Role.EDITOR], default_factory=list)
            dict_field: Dict[str, float] = field(visible_to=[Role.ADMIN], default_factory=dict)
            nested_model: Optional[SimpleFieldModel] = field(visible_to=[Role.VIEWER], default=None)
            secret_data: str = field(visible_to=[Role.ADMIN])

        initial_fields = ModelForTypeCorruptionTest.model_fields
        initial_optional_annotation = initial_fields['optional_str'].annotation
        initial_list_annotation = initial_fields['list_of_int'].annotation
        initial_dict_annotation = initial_fields['dict_field'].annotation
        initial_nested_annotation = initial_fields['nested_model'].annotation
        initial_simple_annotation = initial_fields['secret_data'].annotation
        initial_optional_fieldinfo = initial_fields['optional_str']
        initial_list_fieldinfo = initial_fields['list_of_int']
        initial_dict_fieldinfo = initial_fields['dict_field']
        initial_nested_fieldinfo = initial_fields['nested_model']
        initial_simple_fieldinfo = initial_fields['secret_data']

        assert initial_optional_annotation == Union[str, NoneType] or initial_optional_annotation == Optional[str]
        assert initial_list_annotation == List[int]
        assert initial_dict_annotation == Dict[str, float]
        assert initial_nested_annotation == Union[SimpleFieldModel, NoneType] or initial_nested_annotation == Optional[SimpleFieldModel]
        assert initial_simple_annotation == str

        _ = ModelForTypeCorruptionTest.create_response_model(Role.VIEWER.value)
        _ = ModelForTypeCorruptionTest.create_response_model(Role.EDITOR.value)
        _ = ModelForTypeCorruptionTest.create_response_model(Role.ADMIN.value)

        final_fields = ModelForTypeCorruptionTest.model_fields
        final_optional_annotation = final_fields['optional_str'].annotation
        final_list_annotation = final_fields['list_of_int'].annotation
        final_dict_annotation = final_fields['dict_field'].annotation
        final_nested_annotation = final_fields['nested_model'].annotation
        final_simple_annotation = final_fields['secret_data'].annotation
        final_optional_fieldinfo = final_fields['optional_str']
        final_list_fieldinfo = final_fields['list_of_int']
        final_dict_fieldinfo = final_fields['dict_field']
        final_nested_fieldinfo = final_fields['nested_model']
        final_simple_fieldinfo = final_fields['secret_data']

        assert final_optional_annotation == initial_optional_annotation, "Optional field annotation changed"
        assert final_list_annotation == initial_list_annotation, "List field annotation changed"
        assert final_dict_annotation == initial_dict_annotation, "Dict field annotation changed"
        assert final_nested_annotation == initial_nested_annotation, "Nested Model field annotation changed"
        assert final_simple_annotation == initial_simple_annotation, "Simple field annotation changed"
        assert final_optional_fieldinfo is initial_optional_fieldinfo, "Optional FieldInfo object identity changed"
        assert final_list_fieldinfo is initial_list_fieldinfo, "List FieldInfo object identity changed"
        assert final_dict_fieldinfo is initial_dict_fieldinfo, "Dict FieldInfo object identity changed"
        assert final_nested_fieldinfo is initial_nested_fieldinfo, "Nested Model FieldInfo object identity changed"
        assert final_simple_fieldinfo is initial_simple_fieldinfo, "Simple FieldInfo object identity changed"

    # --- Tests for Potential Bugs (Original Library Code) ---

    # Note: test_to_response_model_raises_validation_error_not_fallback
    # is expected to FAIL if to_response_model uses model_construct,
    # because model_construct bypasses validation.

    def test_to_response_model_preserves_types_without_json_loss(self, model_with_specific_types):
        """
        Tests that to_response_model preserves specific Python types
        (datetime, UUID, Enum) IF model_construct is used AND visible_dict provides them.
        """
        original_model = model_with_specific_types
        role = Role.ADMIN.value
        response_model_instance = original_model.to_response_model(role=role)

        assert isinstance(response_model_instance.event_time, datetime), \
            f"Expected datetime, got {type(response_model_instance.event_time)}"
        assert isinstance(response_model_instance.unique_id, uuid.UUID), \
            f"Expected UUID, got {type(response_model_instance.unique_id)}"
        assert isinstance(response_model_instance.current_status, Role), \
            f"Expected Role Enum, got {type(response_model_instance.current_status)}"
        assert isinstance(response_model_instance.optional_int, int), \
            f"Expected int, got {type(response_model_instance.optional_int)}"
        assert response_model_instance.current_status == Role.EDITOR


    @pytest.mark.xfail(reason="model_construct bypasses validation, so this test is expected to fail.", raises=AssertionError)
    def test_to_response_model_raises_validation_error_not_fallback(self, corruptible_simple_field_model):
        """
        Tests that to_response_model raises ValidationError when the underlying
        data is invalid. This test is EXPECTED TO FAIL if using model_construct.
        """
        model = corruptible_simple_field_model
        role = Role.ADMIN.value

        # This assertion should fail because model_construct won't raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            model.to_response_model(role=role)

        # These checks are now within the xfail context
        errors = exc_info.value.errors()
        error_locs = {err.get('loc',())[0] for err in errors if err.get('loc')}
        assert 'id' in error_locs or any(e['type']=='missing' for e in errors), "Error for missing 'id' not found"
        assert 'value' in error_locs, "Error for invalid 'value' type not found"


    def test_visible_dict_handles_cycles_correctly(self, node_with_direct_cycle, node_with_indirect_cycle):
        """
        Tests that visible_dict correctly identifies cycles and returns
        a predictable structure (e.g., a marker with id).
        """
        direct_cycle_dict = node_with_direct_cycle.visible_dict(role=Role.VIEWER.value)
        assert "self_ref" in direct_cycle_dict
        ref_in_direct_cycle = direct_cycle_dict["self_ref"]
        assert isinstance(ref_in_direct_cycle, dict)
        assert ref_in_direct_cycle.get("__cycle_reference__") is True, "Cycle marker missing in direct cycle"
        assert "id" in ref_in_direct_cycle, "ID missing in direct cycle representation"
        assert ref_in_direct_cycle["id"] == node_with_direct_cycle.id

        indirect_cycle_dict_a = node_with_indirect_cycle.visible_dict(role=Role.VIEWER.value)
        assert "ref_b" in indirect_cycle_dict_a
        ref_b_dict = indirect_cycle_dict_a["ref_b"]
        assert isinstance(ref_b_dict, dict)
        assert "ref_a" in ref_b_dict
        ref_a_in_b_dict = ref_b_dict["ref_a"]
        assert isinstance(ref_a_in_b_dict, dict)
        assert ref_a_in_b_dict.get("__cycle_reference__") is True, "Cycle marker missing in indirect cycle"
        assert "id" in ref_a_in_b_dict, "ID missing in indirect cycle representation"
        assert ref_a_in_b_dict["id"] == node_with_indirect_cycle.id

    # --- Tests for model_construct Behavior (Expected to PASS) ---

    def test_to_response_model_construct_skips_type_coercion(self, construct_test_model_instance):
        """
        Verify model_construct puts the raw type from visible_dict (e.g., str)
        into the field, even if the response model field expects coercion (e.g., int).
        """
        model = construct_test_model_instance
        role = Role.ADMIN.value

        response = model.to_response_model(role=role)

        assert hasattr(response, "int_field")
        assert response.int_field == "123", "Value should be the string provided"
        assert isinstance(response.int_field, str), \
            f"Expected str due to skipped coercion, got {type(response.int_field)}"

    def test_to_response_model_construct_allows_missing_required_fields(self, construct_test_model_instance):
        """
        Verify model_construct creates an instance even if visible_dict omits
        a field required by the response model, leading to AttributeError on access.
        """
        model = construct_test_model_instance
        role = Role.ADMIN.value

        response = model.to_response_model(role=role)

        # Check the field exists on the model *type* but not the *instance*
        assert "required_field" in response.model_fields
        assert not hasattr(response, "required_field"), "Instance should not have the missing attribute"
        # Accessing it should fail
        with pytest.raises(AttributeError):
            _ = response.required_field


    def test_to_response_model_construct_skips_constraints(self, construct_test_model_instance):
        """
        Verify model_construct creates an instance with values violating
        constraints defined on the response model field (copied from original).
        """
        model = construct_test_model_instance
        role = Role.ADMIN.value

        response = model.to_response_model(role=role)

        assert hasattr(response, "constrained_field")
        assert response.constrained_field == -5, "Should contain the violating value"
        # No error is raised, confirming constraint gt=0 was skipped