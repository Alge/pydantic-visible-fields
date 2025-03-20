from __future__ import annotations

import inspect
import json
import sys
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    ForwardRef,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic
from pydantic import BaseModel, Field, create_model

__version__ = "0.1.0"
__all__ = [
    "VisibleFieldsMixin",
    "field",
    "configure_roles",
    "get_role_from_request",
    "VisibleFieldsResponse",
    "VisibleFieldsModel",
]

# Global role configuration
_ROLE_ENUM = None
_ROLE_INHERITANCE = {}
_DEFAULT_ROLE = None
_RESPONSE_MODEL_CACHE = {}

T = TypeVar("T", bound=BaseModel)
ModelT = TypeVar("ModelT", bound="VisibleFieldsModel")


def field(*, visible_to: Optional[List] = None, **kwargs) -> Any:
    """
    Field decorator that adds role visibility metadata to a Pydantic field.

    Args:
        visible_to: List of roles that can see this field
        **kwargs: Additional arguments to pass to pydantic.Field

    Returns:
        Pydantic Field with visibility metadata
    """
    field_kwargs = kwargs.copy()

    if visible_to is not None:
        # Convert role enums to strings for serialization
        visible_to_str = [r.value if isinstance(r, Enum) else r for r in visible_to]

        # Ensure json_schema_extra exists
        if "json_schema_extra" not in field_kwargs:
            field_kwargs["json_schema_extra"] = {}
        elif field_kwargs["json_schema_extra"] is None:
            field_kwargs["json_schema_extra"] = {}

        # Add visibility metadata
        field_kwargs["json_schema_extra"]["visible_to"] = visible_to_str

    return Field(**field_kwargs)


def configure_roles(
    *, role_enum: Type[Enum], inheritance: Dict = None, default_role=None
):
    """
    Configure the role system for visible_fields.

    Args:
        role_enum: Enum class defining the available roles
        inheritance: Dictionary mapping roles to the roles they inherit from
        default_role: Default role to use when none is specified
    """
    global _ROLE_ENUM, _ROLE_INHERITANCE, _DEFAULT_ROLE

    _ROLE_ENUM = role_enum

    if inheritance:
        # Convert enum values to strings in the inheritance dictionary
        _ROLE_INHERITANCE = {
            r.value
            if isinstance(r, Enum)
            else r: [ir.value if isinstance(ir, Enum) else ir for ir in inherited_roles]
            for r, inherited_roles in inheritance.items()
        }

    if default_role is not None:
        _DEFAULT_ROLE = (
            default_role.value if isinstance(default_role, Enum) else default_role
        )


def get_role_from_request(request):
    """
    Extract the user's role from a request object.

    This is a placeholder function that should be customized based on your
    authentication system.

    Args:
        request: The request object

    Returns:
        The user's role
    """
    # This is just a placeholder - implement based on your auth system
    return getattr(request, "user_role", _DEFAULT_ROLE)


def VisibleFieldsResponse(model: Any, role: Any = None):
    """
    Create a response that includes only the fields visible to the specified role.

    Args:
        model: The model to convert
        role: The role to determine field visibility

    Returns:
        Model with only the fields visible to the role
    """
    if hasattr(model, "to_response_model"):
        return model.to_response_model(role=role)
    return model


class VisibleFieldsMixin:
    """
    Mixin class that adds role-based field visibility.
    This can be added to any Pydantic model.
    """

    # Define field visibility by role - can be auto-populated by field decorators
    # This will be properly initialized in __init_subclass__
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {}

    # Role inheritance from the global configuration
    @property
    def _role_inheritance(self) -> Dict[str, List[str]]:
        return _ROLE_INHERITANCE

    # Default role if none specified
    @property
    def _default_role(self) -> str:
        return _DEFAULT_ROLE

    def visible_dict(
        self,
        role: Optional[str] = None,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Convert the model to a dictionary with only the fields
        that should be visible to the specified role.

        Args:
            role: User role to determine field visibility.
                 Defaults to the class's _default_role.
            visited: Dict of already visited object IDs (for cycle detection)
            depth: Current recursion depth

        Returns:
            Dictionary containing only the visible fields for the role.
        """
        # Use default role if none specified
        role = role or self._default_role

        # Initialize visited dict if not provided
        if visited is None:
            visited = {}

        # Get object ID for cycle detection
        obj_id = id(self)

        # Check if we've seen this exact object before
        if obj_id in visited:
            # For circular references more than one level deep, add a marker
            if depth > 1:
                result = visited[obj_id].copy()
                result["__cycle_reference__"] = True
                return result
            return visited[obj_id]

        # Create a placeholder for this object to avoid infinite recursion
        # Use just the ID initially, will be updated with full result at the end
        result = {}
        if hasattr(self, "id"):
            result["id"] = getattr(self, "id")

        # Register this object in the visited dict to break cycles
        visited[obj_id] = result

        # Get all visible fields for this role (including inherited)
        visible_fields = self.__class__._get_all_visible_fields(role)

        # Process visible fields recursively
        for field_name in visible_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                result[field_name] = self._convert_field_to_dict(
                    value, role, visited, depth + 1
                )

        # Update the entry in visited with the complete result
        visited[obj_id] = result

        return result

    def _convert_field_to_dict(
        self,
        value: Any,
        role: str,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Any:
        """
        Convert a field value to a dictionary, recursively handling nested models.

        Args:
            value: The field value to convert
            role: The role to determine field visibility
            visited: Dict of already visited object IDs (for cycle detection)
            depth: Current recursion depth

        Returns:
            The converted value
        """
        if visited is None:
            visited = {}

        # Handle None
        if value is None:
            return None

        # Handle models with visible_dict method
        if (
            isinstance(value, BaseModel)
            and hasattr(value, "visible_dict")
            and callable(getattr(value, "visible_dict"))
        ):
            # Check if this exact object has been visited before
            obj_id = id(value)
            if obj_id in visited and depth > 1:
                result = visited[obj_id].copy()
                result["__cycle_reference__"] = True
                return result

            # If not, process it recursively
            return value.visible_dict(role, visited, depth + 1)

        # Handle Pydantic models without visible_dict
        if isinstance(value, BaseModel):
            # For models without visible_dict, just convert to dict
            return value.model_dump()

        # Handle lists (including lists of models)
        if isinstance(value, list):
            return [
                self._convert_field_to_dict(item, role, visited, depth + 1)
                for item in value
            ]

        # Handle dictionaries (including dictionaries of models)
        if isinstance(value, dict):
            return {
                k: self._convert_field_to_dict(v, role, visited, depth + 1)
                for k, v in value.items()
            }

        # For primitive types, return as is
        return value

    @classmethod
    def _get_all_visible_fields(cls, role: str) -> Set[str]:
        """
        Get all fields visible to a role, including inherited fields.

        Args:
            role: The role to get visible fields for

        Returns:
            Set of all field names visible to the role
        """
        # For non-VisibleFieldsModel (SimpleClassModel in test), use the class attribute directly
        if not issubclass(cls, VisibleFieldsModel) and hasattr(
            cls, "_role_visible_fields"
        ):
            # Direct class-level visibility (SimpleClassModel case)
            visible_fields = set(cls._role_visible_fields.get(role, set()))
        else:
            # Use class attribute
            visible_fields = set(
                getattr(cls, "_role_visible_fields", {}).get(role, set())
            )

        # Add fields from inherited roles
        inherited_roles = _ROLE_INHERITANCE.get(role, [])
        for inherited_role in inherited_roles:
            visible_fields.update(cls._get_all_visible_fields(inherited_role))

        # Get fields from parent classes
        for base in cls.__bases__:
            if hasattr(base, "_get_all_visible_fields") and base != VisibleFieldsMixin:
                visible_fields.update(base._get_all_visible_fields(role))

        return visible_fields

    def to_response_model(self, role: Optional[str] = None):
        """
        Converts this model to a response model for the specified role.
        """
        role = role or self._default_role

        # Get or create the response model class
        if role == self._default_role:
            model_name = f"{self.__class__.__name__}Response"
        else:
            model_name = f"{self.__class__.__name__}{role.capitalize()}Response"

        module = sys.modules[self.__class__.__module__]
        model_cls = getattr(module, model_name, None)

        if model_cls is None:
            model_cls = self.__class__.create_response_model(role)

        # Get visible fields as a dictionary
        visible_data = self.visible_dict(role)

        # Force dictionary conversion for nested structures
        # Use a specialized serializer to handle circular references
        try:
            serialized_data = json.dumps(visible_data)
            processed_data = json.loads(serialized_data)
        except (TypeError, ValueError, OverflowError):
            # If JSON serialization fails (e.g., circular references),
            # handle it manually by creating a new dict without problematic refs
            processed_data = self._sanitize_dict_for_json(visible_data)

        # Create model instance with multiple fallback strategies
        try:
            # First try with model_construct to bypass validation
            return model_cls.model_construct(**processed_data)
        except Exception as e1:
            try:
                # If that fails, try normal validation
                return model_cls.model_validate(processed_data)
            except Exception as e2:
                # If both attempts fail, create a new adjusted model
                try:
                    # Generate a dynamic model based on the actual data structure
                    dynamic_model = self._create_dynamic_model(
                        model_name, processed_data
                    )
                    return dynamic_model.model_validate(processed_data)
                except Exception as e3:
                    # Ultimate fallback - create a response model with generic types
                    ResponseModel = create_model(
                        model_name + "Fallback",
                        **{
                            k: (Any, Field(default=None)) for k in processed_data.keys()
                        },
                    )
                    return ResponseModel.model_validate(processed_data)

    def _sanitize_dict_for_json(self, data):
        """
        Sanitize dictionary to make it JSON serializable by removing circular references.
        """
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                try:
                    # Test if this value is JSON serializable
                    json.dumps(v)
                    result[k] = v
                except (TypeError, ValueError, OverflowError):
                    # If not, recursively sanitize it
                    result[k] = self._sanitize_dict_for_json(v)
            return result
        elif isinstance(data, list):
            result = []
            for item in data:
                try:
                    # Test if this item is JSON serializable
                    json.dumps(item)
                    result.append(item)
                except (TypeError, ValueError, OverflowError):
                    # If not, recursively sanitize it
                    result.append(self._sanitize_dict_for_json(item))
            return result
        else:
            # For non-container types, return a string representation
            return str(data)

    def _create_dynamic_model(
        self, base_name: str, data: Dict[str, Any]
    ) -> Type[BaseModel]:
        """
        Create a dynamic model that exactly matches the structure of the data.
        """
        model_name = f"{base_name}Dynamic"

        # Define fields based on the data
        fields = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # For nested dict, use Dict[str, Any]
                fields[key] = (Dict[str, Any], Field(default_factory=dict))
            elif isinstance(value, list):
                # For lists, use List[Any]
                fields[key] = (List[Any], Field(default_factory=list))
            elif value is None:
                # For None values, use Optional[Any]
                fields[key] = (Optional[Any], None)
            else:
                # For other types, use their actual type with default
                fields[key] = (type(value), Field(default=None))

        # Create the model
        return create_model(model_name, **fields)

    @classmethod
    def create_response_model(
        cls, role: str, model_name_suffix: str = "Response"
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model with only the fields visible to the specified role.
        """
        # Check cache first to avoid recreating models
        cache_key = (cls.__name__, role, model_name_suffix)
        if cache_key in _RESPONSE_MODEL_CACHE:
            return _RESPONSE_MODEL_CACHE[cache_key]

        # Get visible fields
        visible_fields = cls._get_all_visible_fields(role)

        # Create field definitions
        fields = {}
        for field_name in visible_fields:
            if field_name not in cls.model_fields:
                continue

            field_info = cls.model_fields[field_name]
            annotation = field_info.annotation

            # For complex types like unions, nested models, etc., use Dict or List
            origin = get_origin(annotation)

            if origin is Union:
                # For Union types, use Dict[str, Any]
                fields[field_name] = (Dict[str, Any], field_info)
            elif origin is list or origin is List:
                # For lists, use List[Any]
                fields[field_name] = (List[Any], field_info)
            elif origin is dict or origin is Dict:
                # For dictionaries, use Dict[str, Any]
                fields[field_name] = (Dict[str, Any], field_info)
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                # For model fields, use Dict[str, Any]
                fields[field_name] = (Dict[str, Any], field_info)
            else:
                # For primitive types, use the original annotation
                fields[field_name] = (annotation, field_info)

        # Create model name
        if role == cls._default_role:
            model_name = f"{cls.__name__}{model_name_suffix}"
        else:
            model_name = f"{cls.__name__}{role.capitalize()}{model_name_suffix}"

        # Create the model
        response_model = create_model(model_name, **fields)

        # Add a Config class to make validation more permissive
        setattr(response_model, "model_config", {"extra": "ignore"})

        # Cache and return
        _RESPONSE_MODEL_CACHE[cache_key] = response_model
        return response_model

    @classmethod
    def configure_visibility(cls, role: str, visible_fields: Set[str]):
        """
        Configure the visibility of fields for a specific role.

        Args:
            role: Role to configure visibility for
            visible_fields: Set of field names that should be visible to the role
        """
        # Initialize _role_visible_fields if not already done
        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
            cls._role_visible_fields = {}

        # Update visibility for the specified role
        cls._role_visible_fields[role] = set(visible_fields)


class VisibleFieldsModel(BaseModel, VisibleFieldsMixin):
    """
    Base class for models with field-level visibility control.

    Use this instead of BaseModel when you want field-level visibility control.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Initialize the _role_visible_fields for each subclass."""
        # Initialize role visible fields dictionary
        cls._role_visible_fields = {}

        # Fill with roles from enum if available
        if _ROLE_ENUM:
            for role in _ROLE_ENUM.__members__.values():
                role_value = role.value if isinstance(role, Enum) else role
                cls._role_visible_fields[role_value] = set()

        # Scan model fields for visibility metadata
        for field_name, field_info in cls.model_fields.items():
            json_schema_extra = getattr(field_info, "json_schema_extra", {})
            if json_schema_extra and "visible_to" in json_schema_extra:
                visible_to = json_schema_extra["visible_to"]
                for role in visible_to:
                    if role not in cls._role_visible_fields:
                        cls._role_visible_fields[role] = set()
                    cls._role_visible_fields[role].add(field_name)
