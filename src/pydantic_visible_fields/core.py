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
    ForwardRef,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined

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
        return create_model(
            name,
            __config__=dict(extra="ignore", populate_by_name=True),
            **field_dict,
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
        visible_to_str = [r.value if isinstance(r, Enum) else r for r in visible_to]
        if "json_schema_extra" not in field_kwargs:
            field_kwargs["json_schema_extra"] = {}
        elif field_kwargs["json_schema_extra"] is None:
            field_kwargs["json_schema_extra"] = {}
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
    # Check if the object has the method before calling
    if isinstance(model, VisibleFieldsMixin):
        return model.to_response_model(role=role)
    # Handle lists: recursively call on items
    if isinstance(model, list):
        return [visible_fields_response(item, role) for item in model]
    # Handle dicts: recursively call on values (assuming keys don't need conversion)
    if isinstance(model, dict):
        return {k: visible_fields_response(v, role) for k, v in model.items()}
    # Otherwise, return the item as is
    return model


class VisibleFieldsMixin:
    """
    Mixin class that adds role-based field visibility.
    This can be added to any Pydantic model.
    """

    model_fields: ClassVar[Dict[str, Any]]
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {}

    @property
    def _role_inheritance(self) -> Dict[str, List[str]]:
        return _ROLE_INHERITANCE

    @property
    def _default_role(self) -> str:
        if _DEFAULT_ROLE is None:
            # Provide a default empty string if not configured
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
        that should be visible to the specified role. Includes cycle detection.
        """
        role = role or self._default_role
        if visited is None:
            visited = {}

        obj_id = id(self)
        if obj_id in visited:
            # Return cycle marker immediately if visited
            cycle_data = {"__cycle_reference__": True}
            # Attempt to add ID for better reference, ignore if no 'id' attribute
            try:
                cycle_data["id"] = getattr(self, "id")
            except AttributeError:
                pass
            return cycle_data

        # Store placeholder temporarily for cycle detection within this object's graph
        temp_placeholder = {"__processing_placeholder__": True}
        try:
            temp_placeholder["id"] = getattr(self, "id")
        except AttributeError:
            pass
        visited[obj_id] = temp_placeholder

        result: Dict[str, Any] = {}
        visible_fields = self.__class__._get_all_visible_fields(role)

        for field_name in visible_fields:
            # Use getattr_static for potentially faster access if available and safe,
            # otherwise stick to hasattr/getattr
            try:
                 value = getattr(self, field_name)
                 result[field_name] = self._convert_value_to_dict_recursive(
                     value, role, visited, depth + 1
                 )
            except AttributeError:
                 # Field listed in _role_visible_fields but not actually on the instance? Log warning?
                 pass


        # Replace placeholder with the actual result before returning
        visited[obj_id] = result
        # Clean up from visited *after* full processing of this node to allow siblings to use it
        # del visited[obj_id] # Careful: This might break complex shared object scenarios
        # Consider only removing if depth is 0? Or manage visited differently.
        # For now, keep it simple: let visited grow, assuming it's managed per top-level call.

        return result

    def _convert_value_to_dict_recursive(
        self,
        value: Any,
        role: str,
        visited: Dict[int, Dict[str, Any]],
        depth: int = 0,
    ) -> Any:
        """
        Recursively convert field values based on role visibility.
        Handles nested models, lists, dicts, and primitives.
        Keeps Enum members as Enum members.
        """
        if value is None:
            return None

        # Check for objects using this mixin first
        if isinstance(value, VisibleFieldsMixin):
            return value.visible_dict(role, visited, depth)

        # Handle other Pydantic models (not using the mixin)
        if isinstance(value, BaseModel):
            # Dump these models as they are, assuming they don't need role filtering
            try:
                 return value.model_dump()
            except Exception:
                 # Fallback if model_dump fails for some reason
                 return str(value) # Or some other representation

        # Handle lists recursively
        if isinstance(value, list):
            return [
                self._convert_value_to_dict_recursive(item, role, visited, depth + 1)
                for item in value
            ]

        # Handle dictionaries recursively
        if isinstance(value, dict):
            return {
                k: self._convert_value_to_dict_recursive(v, role, visited, depth + 1)
                for k, v in value.items()
            }

        # --- FIX: Removed Enum value conversion ---
        # Keep Enum members as actual Enum members for model_construct
        # if isinstance(value, Enum):
        #     return value.value

        # Otherwise, return the primitive value as is
        return value

    @classmethod
    def _get_all_visible_fields(cls, role: str) -> Set[str]:
        """
        Get all fields visible to a role, including inherited fields.
        """
        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
            # Ensure the class attribute exists, possibly initializing if needed
             cls._role_visible_fields = {} # Or initialize based on bases/decorators

        visible_fields = set(cls._role_visible_fields.get(role, set()))

        inherited_roles = _ROLE_INHERITANCE.get(role, [])
        for inherited_role in inherited_roles:
            visible_fields.update(cls._get_all_visible_fields(inherited_role))

        for base in cls.__bases__:
            if issubclass(base, VisibleFieldsMixin) and base is not VisibleFieldsMixin:
                visible_fields.update(base._get_all_visible_fields(role))

        return visible_fields

    @classmethod
    def _get_recursive_response_type(
        cls,
        annotation: Type[Any],
        role: str,
        model_name_suffix: str,
        visited_fwd_refs: Optional[Set[str]] = None,
    ) -> Type[Any]:
        """
        Recursively determine the appropriate response model type for a given annotation and role.
        """
        if visited_fwd_refs is None:
            visited_fwd_refs = set()

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union:
            is_optional = len(args) == 2 and args[1] is type(None)
            processed_args = []
            for arg in args:
                if arg is type(None):
                    continue # Handle None separately
                processed_args.append(
                    cls._get_recursive_response_type(
                        arg, role, model_name_suffix, visited_fwd_refs
                    )
                )
            if is_optional:
                if not processed_args: return Optional[Any] # Should have had one non-None arg
                # Optional[T] -> Optional[ProcessedT]
                return Optional[processed_args[0]] # type: ignore
            else:
                if not processed_args: return Any # Union without types?
                # Union[T1, T2] -> Union[ProcessedT1, ProcessedT2]
                return Union[tuple(processed_args)] # type: ignore

        if origin is list or origin is List:
            if not args: return List[Any]
            nested_type = cls._get_recursive_response_type(
                args[0], role, model_name_suffix, visited_fwd_refs
            )
            return List[nested_type] # type: ignore

        if origin is dict or origin is Dict:
            if not args or len(args) != 2: return Dict[Any, Any]
            key_type = args[0]
            value_type = cls._get_recursive_response_type(
                args[1], role, model_name_suffix, visited_fwd_refs
            )
            return Dict[key_type, value_type] # type: ignore

        if isinstance(annotation, ForwardRef):
            fwd_arg = annotation.__forward_arg__
            if fwd_arg in visited_fwd_refs:
                return annotation
            visited_fwd_refs.add(fwd_arg)
            try:
                module = sys.modules[cls.__module__]
                actual_type = annotation._evaluate(module.__dict__, globals(), frozenset())
                resolved_type = cls._get_recursive_response_type(
                    actual_type, role, model_name_suffix, visited_fwd_refs
                )
            except Exception:
                resolved_type = annotation # Fallback
            finally:
                visited_fwd_refs.remove(fwd_arg)
            return resolved_type

        if isinstance(annotation, type) and issubclass(annotation, VisibleFieldsMixin):
            return annotation.create_response_model(role, model_name_suffix)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # Return Dict for other BaseModels to avoid issues if they weren't designed for this
            return Dict[str, Any]

        return annotation

    @classmethod
    def create_response_model(
        cls, role: str, model_name_suffix: str = "Response"
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model with only the fields visible to the specified role,
        recursively handling nested types.
        """
        cache_key = (cls.__name__, role, model_name_suffix)
        if cache_key in _RESPONSE_MODEL_CACHE:
            return _RESPONSE_MODEL_CACHE[cache_key]

        visible_fields = cls._get_all_visible_fields(role)
        new_fields_definition: Dict[str, Tuple[Any, Any]] = {}
        visited_fwd_refs_for_creation = set()

        for field_name in visible_fields:
            if field_name not in cls.model_fields:
                continue

            original_field_info = cls.model_fields[field_name]
            original_annotation = original_field_info.annotation

            response_annotation = cls._get_recursive_response_type(
                original_annotation, role, model_name_suffix, visited_fwd_refs_for_creation
            )

            new_field_kwargs = {}
            # Copy relevant attributes from original FieldInfo to new Field definition
            if original_field_info.description:
                new_field_kwargs["description"] = original_field_info.description
            if original_field_info.title:
                 new_field_kwargs["title"] = original_field_info.title
            # Handle alias - critical for mapping data if visible_dict uses field names
            if original_field_info.alias: # Keep alias if it exists
                 new_field_kwargs["alias"] = original_field_info.alias
            if original_field_info.examples:
                new_field_kwargs["examples"] = original_field_info.examples
            # Copy constraints
            if original_field_info.metadata:
                for meta_key in [
                    "ge", "gt", "le", "lt", "multiple_of", "min_length",
                    "max_length", "pattern",
                ]:
                     # Use getattr with default PydanticUndefined for safety
                     meta_value = getattr(original_field_info, meta_key, PydanticUndefined)
                     if meta_value is not PydanticUndefined:
                        new_field_kwargs[meta_key] = meta_value

            # Copy default/default_factory
            if original_field_info.default is not PydanticUndefined:
                new_field_kwargs["default"] = original_field_info.default
            elif original_field_info.default_factory is not None:
                new_field_kwargs["default_factory"] = original_field_info.default_factory

            new_field = Field(**new_field_kwargs)
            new_fields_definition[field_name] = (response_annotation, new_field)

        if _DEFAULT_ROLE is not None and role == _DEFAULT_ROLE:
            model_name = f"{cls.__name__}{model_name_suffix}"
        else:
            model_name = f"{cls.__name__}{role.capitalize()}{model_name_suffix}"

        # Use _safe_create_model which sets config (extra='ignore', populate_by_name=True)
        response_model = _safe_create_model(model_name, new_fields_definition)

        typed_response_model = cast(Type[BaseModel], response_model)
        _RESPONSE_MODEL_CACHE[cache_key] = typed_response_model
        return typed_response_model

    def _construct_nested_model(self, value: Dict[str, Any], annotation: Type[BaseModel]) -> Any:
         """Helper to construct a nested model instance from a dictionary."""
         inner_data_for_construct = {}
         if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
              return value # Cannot construct if annotation is not a model type

         for nested_field_name, nested_field_info in annotation.model_fields.items():
             nested_value = PydanticUndefined
             # Check field name first, then alias
             if nested_field_name in value:
                 nested_value = value[nested_field_name]
             elif nested_field_info.alias and nested_field_info.alias in value:
                  nested_value = value[nested_field_info.alias]

             if nested_value is not PydanticUndefined:
                  # Recursively process this value *before* putting it in the data dict
                  processed_nested_value = self._process_value_for_construct_recursive(
                      nested_value, nested_field_info.annotation
                  )
                  inner_data_for_construct[nested_field_name] = processed_nested_value

         try:
             # Construct the nested model instance
             return annotation.model_construct(**inner_data_for_construct)
         except Exception:
             # Log error?
             return value # Fallback: return original dict if construction fails

    def _process_value_for_construct_recursive(
            self, value: Any, annotation: Type[Any]
    ) -> Any:
        """
        Helper to recursively process/construct nested models if the value is a dict/list
        and the target annotation indicates a nested structure.
        """
        if value is None:
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types (including Optional)
        if origin is Union:
            is_optional = len(args) == 2 and type(None) in args
            possible_types = [arg for arg in args if arg is not type(None)]

            # Handle Optional[T] if value is not None
            if is_optional and len(possible_types) == 1 and value is not None:
                return self._process_value_for_construct_recursive(value,
                                                                   possible_types[0])
            elif is_optional and value is None:
                return None  # Value is None for Optional, return None

            # Handle discriminated unions if value is a dict and we have model types
            if isinstance(value, dict):
                potential_models = [
                    t for t in possible_types if
                    isinstance(t, type) and issubclass(t, BaseModel)
                ]
                if potential_models:
                    # --- Refined Discriminator Logic ---
                    discriminator_field_name = "type"  # Assuming 'type' is the discriminator
                    discriminator_value = value.get(discriminator_field_name)
                    matched_model_type = None

                    if discriminator_value is not None:
                        # Convert incoming discriminator to string for comparison consistency
                        # (visible_dict might return Enum or str depending on implementation)
                        discriminator_str = str(
                            discriminator_value.value) if isinstance(
                            discriminator_value, Enum) else str(discriminator_value)

                        for model_type in potential_models:
                            type_field_info = model_type.model_fields.get(
                                discriminator_field_name)
                            if not type_field_info: continue

                            # 1. Check Literal annotation
                            ann_origin = get_origin(type_field_info.annotation)
                            ann_args = get_args(type_field_info.annotation)
                            from typing import Literal  # Ensure imported
                            if ann_origin is Literal and discriminator_str in ann_args:
                                matched_model_type = model_type
                                break

                            # 2. Check default value if not Literal
                            if matched_model_type is None:
                                field_default = getattr(type_field_info, 'default',
                                                        PydanticUndefined)
                                if field_default != PydanticUndefined:
                                    # Convert default to string for comparison
                                    default_str = str(
                                        field_default.value) if isinstance(
                                        field_default, Enum) else str(field_default)
                                    if default_str == discriminator_str:
                                        matched_model_type = model_type
                                        break
                    # --- End Refined Discriminator Logic ---

                    # Construct using matched type ONLY if found
                    if matched_model_type:
                        try:
                            constructed = self._construct_nested_model(value,
                                                                       matched_model_type)
                            # Ensure construction returned the expected type
                            if isinstance(constructed, matched_model_type):
                                return constructed
                            # Else: construction failed or returned original dict, fall through
                        except Exception:
                            # Log construction error? Fall through
                            pass

                # If no specific model type matched or constructed, return original dict
                return value

            # Value is not a dict or not a Union of models
            return value

        # Handle List[T]
        if origin is list or origin is List:
            if not args or not isinstance(value, list):
                return value
            nested_annotation = args[0]
            return [
                self._process_value_for_construct_recursive(item, nested_annotation)
                for item in value
            ]

        # Handle Dict[K, V]
        if origin is dict or origin is Dict:
            if not args or len(args) != 2 or not isinstance(value, dict):
                return value
            value_annotation = args[1]
            return {
                k: self._process_value_for_construct_recursive(v, value_annotation)
                for k, v in value.items()
            }

        # Direct nested BaseModel check (ensure annotation is a type)
        if isinstance(annotation, type) and issubclass(annotation,
                                                       BaseModel) and isinstance(value,
                                                                                 dict):
            if value.get("__cycle_reference__") is True:
                return value  # Return cycle marker dict
            return self._construct_nested_model(value, annotation)

        # Default: return primitive types, Enums, already constructed objects, etc.
        return value
    

    def to_response_model(self, role: Optional[str] = None) -> BaseModel:
        """
        Convert this model to a response model instance for the specified role,
        recursively constructing nested models.
        """
        role = role or self._default_role
        model_cls = self.__class__.create_response_model(role)
        visible_data_dict: Dict[str, Any] = self.visible_dict(role)
        data_for_construction: Dict[str, Any] = {}

        for field_name, field_info in model_cls.model_fields.items():
            value_from_dict = PydanticUndefined
            # Use field name first
            if field_name in visible_data_dict:
                value_from_dict = visible_data_dict[field_name]
            # Fallback to alias if field name not found
            elif field_info.alias and field_info.alias in visible_data_dict:
                value_from_dict = visible_data_dict[field_info.alias]

            if value_from_dict is not PydanticUndefined:
                data_for_construction[field_name] = self._process_value_for_construct_recursive(
                    value_from_dict, field_info.annotation
                )

        try:
            return model_cls.model_construct(**data_for_construction)
        except Exception as e:
            raise e

    @classmethod
    def configure_visibility(cls, role: str, visible_fields: Set[str]) -> None:
        """
        Configure the visibility of fields for a specific role.
        """
        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
            cls._role_visible_fields = {}
        cls._role_visible_fields[role] = set(visible_fields)


class VisibleFieldsModel(BaseModel, VisibleFieldsMixin):
    """
    Base class for models with field-level visibility control.
    """
    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the _role_visible_fields for each subclass."""
        base_vis_fields: Dict[str, Set[str]] = {}
        for base in cls.__bases__:
             if issubclass(base, VisibleFieldsMixin) and hasattr(base, "_role_visible_fields"):
                  for r, flds in getattr(base, "_role_visible_fields", {}).items():
                        if r not in base_vis_fields:
                             base_vis_fields[r] = set()
                        base_vis_fields[r].update(flds)

        cls._role_visible_fields = base_vis_fields

        if _ROLE_ENUM:
            for role in _ROLE_ENUM.__members__.values():
                role_value = role.value if isinstance(role, Enum) else role
                if role_value not in cls._role_visible_fields:
                    cls._role_visible_fields[role_value] = set()

        for field_name, field_info in cls.model_fields.items():
            json_schema_extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(json_schema_extra, dict) and "visible_to" in json_schema_extra:
                visible_to = json_schema_extra["visible_to"]
                if isinstance(visible_to, list):
                    for role in visible_to:
                        role_key = role.value if isinstance(role, Enum) else role
                        if role_key not in cls._role_visible_fields:
                            cls._role_visible_fields[role_key] = set()
                        cls._role_visible_fields[role_key].add(field_name)