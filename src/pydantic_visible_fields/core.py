"""
Module for role-based field visibility for Pydantic models.
This module provides a mixin and supporting functions to restrict which
fields are visible in the model output based on a user's role.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_origin,
)

from pydantic import BaseModel, Field, create_model


# Global role configuration.
_ROLE_ENUM: Optional[Type[Enum]] = None
_ROLE_INHERITANCE: Dict[str, List[str]] = {}
_DEFAULT_ROLE: Optional[str] = None
_RESPONSE_MODEL_CACHE: Dict[Tuple[str, str, str], Type[BaseModel]] = {}

T = TypeVar("T", bound=BaseModel)
ModelT = TypeVar("ModelT", bound="VisibleFieldsModel")


def _safe_create_model(
    name: str, fields: Dict[str, Tuple[Type[Any], Any]]
) -> Type[BaseModel]:
    """
    Safely create a Pydantic model with error handling.
    """
    try:
        field_dict = {k: (v[0], v[1]) for k, v in fields.items()}
        # Only ignore the call-overload error, not all type checking
        return create_model(  # type: ignore[call-overload,no-any-return]
            name, **field_dict
        )
    except Exception as e:
        raise ValueError(f"Failed to create model {name}: {e}")


def field(*, visible_to: Optional[List[Any]] = None, **kwargs: Any) -> Any:
    """
    Field decorator that adds role visibility metadata to a Pydantic field.

    Args:
        visible_to: List of roles that can see this field.
        **kwargs: Additional arguments to pass to pydantic.Field.

    Returns:
        Pydantic Field with visibility metadata.
    """
    field_kwargs = kwargs.copy()

    if visible_to is not None:
        # Convert role enums to strings for serialization.
        visible_to_str = [r.value if isinstance(r, Enum) else r for r in visible_to]

        # Ensure json_schema_extra exists.
        if "json_schema_extra" not in field_kwargs:
            field_kwargs["json_schema_extra"] = {}
        elif field_kwargs["json_schema_extra"] is None:
            field_kwargs["json_schema_extra"] = {}

        # Add visibility metadata.
        field_kwargs["json_schema_extra"]["visible_to"] = visible_to_str

    return Field(**field_kwargs)


def configure_roles(
    *,
    role_enum: Type[Enum],
    inheritance: Optional[Dict[Any, Any]] = None,
    default_role: Optional[Union[Enum, str]] = None,
) -> None:
    """
    Configure the role system for visible_fields.

    Args:
        role_enum: Enum class defining the available roles.
        inheritance: Dictionary mapping roles to the roles they inherit from.
        default_role: Default role to use when none is specified.
    """
    global _ROLE_ENUM, _ROLE_INHERITANCE, _DEFAULT_ROLE

    _ROLE_ENUM = role_enum

    if inheritance:
        # Convert enum values to strings in the inheritance dictionary.
        _ROLE_INHERITANCE = {
            (r.value if isinstance(r, Enum) else r): [
                ir.value if isinstance(ir, Enum) else ir for ir in inherited_roles
            ]
            for r, inherited_roles in inheritance.items()
        }

    _DEFAULT_ROLE = (
        default_role.value
        if isinstance(default_role, Enum) and default_role is not None
        else default_role
    )


def visible_fields_response(model: Any, role: Any = None) -> Any:
    """
    Create a response that includes only the fields visible to the specified role.
    This also handles objects that does not inherit from VisibleFieldsMixin,
    returning the item as-is.

    Args:
        model: The model to convert.
        role: The role to determine field visibility.

    Returns:
        Model with only the fields visible to the role.
    """
    if hasattr(model, "to_response_model"):
        return model.to_response_model(role=role)
    return model


class VisibleFieldsMixin:
    """
    Mixin class that adds role-based field visibility.
    This can be added to any Pydantic model.
    """

    # Add a declaration for model_fields so that mypy recognizes it.
    model_fields: ClassVar[Dict[str, Any]]

    # Define field visibility by role - can be auto-populated by field
    # decorators. This will be properly initialized in __init_subclass__.
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {}

    @property
    def _role_inheritance(self) -> Dict[str, List[str]]:
        return _ROLE_INHERITANCE

    @property
    def _default_role(self) -> str:
        if _DEFAULT_ROLE is None:
            return ""
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
            role: User role to determine field visibility. Defaults to the
                  class's _default_role.
            visited: Dict of already visited object IDs (for cycle detection).
            depth: Current recursion depth.

        Returns:
            Dictionary containing only the visible fields for the role.
        """
        role = role or self._default_role

        if visited is None:
            visited = {}

        obj_id = id(self)
        if obj_id in visited:
            if depth > 1:
                # Rename temporary variable to avoid duplicate definition.
                cycle_result = visited[obj_id].copy()
                cycle_result["__cycle_reference__"] = True
                return cycle_result
            return visited[obj_id]

        result: Dict[str, Any] = {}
        if hasattr(self, "id"):
            result["id"] = getattr(self, "id")

        visited[obj_id] = result

        visible_fields = self.__class__._get_all_visible_fields(role)

        for field_name in visible_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                result[field_name] = self._convert_field_to_dict(
                    value, role, visited, depth + 1
                )

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
            value: The field value to convert.
            role: The role to determine field visibility.
            visited: Dict of already visited object IDs (for cycle detection).
            depth: Current recursion depth.

        Returns:
            The converted value.
        """
        if visited is None:
            visited = {}

        if value is None:
            return None

        if (
            isinstance(value, BaseModel)
            and hasattr(value, "visible_dict")
            and callable(getattr(value, "visible_dict"))
        ):
            obj_id = id(value)
            if obj_id in visited and depth > 1:
                result = visited[obj_id].copy()
                result["__cycle_reference__"] = True
                return result

            return value.visible_dict(role, visited, depth + 1)

        if isinstance(value, BaseModel):
            return value.model_dump()

        if isinstance(value, list):
            return [
                self._convert_field_to_dict(item, role, visited, depth + 1)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                k: self._convert_field_to_dict(v, role, visited, depth + 1)
                for k, v in value.items()
            }

        return value

    @classmethod
    def _get_all_visible_fields(cls, role: str) -> Set[str]:
        """
        Get all fields visible to a role, including inherited fields.

        Args:
            role: The role to get visible fields for.

        Returns:
            Set of all field names visible to the role.
        """
        if not issubclass(cls, VisibleFieldsModel) and hasattr(
            cls, "_role_visible_fields"
        ):
            visible_fields = set(cls._role_visible_fields.get(role, set()))
        else:
            visible_fields = set(
                getattr(cls, "_role_visible_fields", {}).get(role, set())
            )

        inherited_roles = _ROLE_INHERITANCE.get(role, [])
        for inherited_role in inherited_roles:
            visible_fields.update(cls._get_all_visible_fields(inherited_role))

        for base in cls.__bases__:
            if hasattr(base, "_get_all_visible_fields") and base != VisibleFieldsMixin:
                visible_fields.update(base._get_all_visible_fields(role))

        return visible_fields

    def to_response_model(self, role: Optional[str] = None) -> BaseModel:
        """Convert this model to a response model for the specified role."""
        role = role or self._default_role

        if role == self._default_role:
            model_name = f"{self.__class__.__name__}Response"
        else:
            model_name = f"{self.__class__.__name__}{role.capitalize()}Response"

        module = sys.modules[self.__class__.__module__]
        model_cls = getattr(module, model_name, None)

        if model_cls is None:
            model_cls = self.__class__.create_response_model(role)

        visible_data = self.visible_dict(role)

        try:
            serialized_data = json.dumps(visible_data)
            processed_data = json.loads(serialized_data)
        except (TypeError, ValueError, OverflowError):
            processed_data = self._sanitize_dict_for_json(visible_data)

        try:
            return model_cls.model_construct(**processed_data)
        except Exception as _:  # noqa: F841
            try:
                return model_cls.model_validate(processed_data)
            except Exception as _:  # noqa: F841
                try:
                    dynamic_model = self._create_dynamic_model(
                        model_name, processed_data
                    )
                    return dynamic_model.model_validate(processed_data)
                except Exception as _:  # noqa: F841
                    # Use a proper type for Any that works in tuples
                    fields: Dict[str, Tuple[Type[Any], Any]] = {}
                    for k in processed_data.keys():
                        # Use type(None) for default of None, and object for the type
                        fields[k] = (object, None)
                    ResponseModel = _safe_create_model(model_name + "Fallback", fields)
                    return ResponseModel.model_validate(processed_data)

    def _sanitize_dict_for_json(self, data: Any) -> Any:
        """
        Sanitize dictionary to make it JSON serializable by removing circular
        references.
        """
        if isinstance(data, dict):
            result_dict: Dict[Any, Any] = {}
            for k, v in data.items():
                try:
                    json.dumps(v)
                    result_dict[k] = v
                except (TypeError, ValueError, OverflowError):
                    result_dict[k] = self._sanitize_dict_for_json(v)
            return result_dict
        elif isinstance(data, list):
            result_list: List[Any] = []
            for item in data:
                try:
                    json.dumps(item)
                    result_list.append(item)
                except (TypeError, ValueError, OverflowError):
                    result_list.append(self._sanitize_dict_for_json(item))
            return result_list
        else:
            return str(data)

    def _create_dynamic_model(
        self, base_name: str, data: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create a dynamic model that exactly matches the structure of the data."""
        model_name = f"{base_name}Dynamic"
        fields: Dict[str, Tuple[Type[Any], Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                fields[key] = (Dict[str, Any], Field(default_factory=lambda: {}))
            elif isinstance(value, list):
                fields[key] = (List[Any], Field(default_factory=lambda: []))
            elif value is None:
                # Use object instead of Any to fix the type error
                fields[key] = (object, None)
            else:
                fields[key] = (type(value), Field(default=None))
        return _safe_create_model(model_name, fields)

    @classmethod
    def create_response_model(
        cls, role: str, model_name_suffix: str = "Response"
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model with only the fields visible to the specified
        role.
        """
        cache_key = (cls.__name__, role, model_name_suffix)
        if cache_key in _RESPONSE_MODEL_CACHE:
            return _RESPONSE_MODEL_CACHE[cache_key]

        visible_fields = cls._get_all_visible_fields(role)
        fields: Dict[str, Tuple[Any, Any]] = {}
        for field_name in visible_fields:
            # Use the declared model_fields attribute.
            if field_name not in cls.model_fields:
                continue

            field_info = cls.model_fields[field_name]
            annotation = field_info.annotation
            origin = get_origin(annotation)

            if origin is Union:
                fields[field_name] = (Dict[str, Any], field_info)
            elif origin is list or origin is List:
                fields[field_name] = (List[Any], field_info)
            elif origin is dict or origin is Dict:
                fields[field_name] = (Dict[str, Any], field_info)
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                fields[field_name] = (Dict[str, Any], field_info)
            else:
                fields[field_name] = (annotation, field_info)

        if role == _DEFAULT_ROLE:
            model_name = f"{cls.__name__}{model_name_suffix}"
        else:
            model_name = f"{cls.__name__}{role.capitalize()}{model_name_suffix}"

        response_model = create_model(model_name, **fields)  # type: ignore
        setattr(response_model, "model_config", {"extra": "ignore"})

        _RESPONSE_MODEL_CACHE[cache_key] = response_model
        return cast(Type[BaseModel], response_model)

    @classmethod
    def configure_visibility(cls, role: str, visible_fields: Set[str]) -> None:
        """
        Configure the visibility of fields for a specific role.

        Args:
            role: Role to configure visibility for.
            visible_fields: Set of field names that should be visible to the role.
        """
        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
            cls._role_visible_fields = {}

        cls._role_visible_fields[role] = set(visible_fields)


class VisibleFieldsModel(BaseModel, VisibleFieldsMixin):
    """
    Base class for models with field-level visibility control.

    Use this instead of BaseModel when you want field-level visibility control.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the _role_visible_fields for each subclass."""
        cls._role_visible_fields = {}

        if _ROLE_ENUM:
            for role in _ROLE_ENUM.__members__.values():
                role_value = role.value if isinstance(role, Enum) else role
                cls._role_visible_fields[role_value] = set()

        for field_name, field_info in cls.model_fields.items():
            json_schema_extra = getattr(field_info, "json_schema_extra", {})
            if json_schema_extra and "visible_to" in json_schema_extra:
                visible_to = json_schema_extra["visible_to"]
                for role in visible_to:
                    role_key = role.value if isinstance(role, Enum) else role
                    if role_key not in cls._role_visible_fields:
                        cls._role_visible_fields[role_key] = set()
                    cls._role_visible_fields[role_key].add(field_name)
