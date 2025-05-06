# src/pydantic_visible_fields/core.py
"""
Core implementation for role-based field visibility in Pydantic models.

Provides the `VisibleFieldsMixin`, the `VisibleFieldsModel` base class,
the `field` function for declaring visible fields, and configuration functions.
"""

from __future__ import annotations

import logging
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
from pydantic.fields import FieldInfo

# Only import necessary components from pydantic_core
from pydantic_core import PydanticUndefined

# Standard logger for the module
logger = logging.getLogger(__name__)

# --- Global Role Configuration ---
# Stores the application-wide role settings used by the mixin.

# The Enum defining application roles (e.g., class Role(str, Enum): ...)
_ROLE_ENUM: Optional[Type[Enum]] = None
# Dictionary defining role inheritance { "role_str": ["inherited_role_str"] }
_ROLE_INHERITANCE: Dict[str, List[str]] = {}
# Default role (string value) used if no role is specified
_DEFAULT_ROLE: Optional[str] = None
# Cache for dynamically generated response model types to avoid recreation
_RESPONSE_MODEL_CACHE: Dict[Tuple[str, str, str], Type[BaseModel]] = {}
# --- End Global Role Configuration ---

# Generic TypeVar for Pydantic models
T = TypeVar("T", bound=BaseModel)
# Generic TypeVar specifically for models inheriting from VisibleFieldsModel
ModelT = TypeVar("ModelT", bound="VisibleFieldsModel")


def field(*, visible_to: Optional[List[Any]] = None, **kwargs: Any) -> Any:
    """
    Custom pydantic.Field function enabling role-based visibility.

    Use this in place of `pydantic.Field` to specify roles that can view
    a field. Visibility information is stored in `field_info.json_schema_extra`.

    Args:
        visible_to: A list of role identifiers (e.g., Enum members or their
            string values) allowed to view this field. If None, the field
            has no specific role restrictions (but might be hidden if the
            model itself is nested within another field with restrictions).
        **kwargs: Any other keyword arguments accepted by `pydantic.Field`
            (e.g., `default`, `alias`, `description`, `gt`, `le`).

    Returns:
        A Pydantic FieldInfo object configured with potential visibility
        metadata. This object is used by Pydantic during model creation.
    """
    field_kwargs = kwargs.copy()

    if visible_to is not None:
        # Convert role identifiers to strings for consistent storage
        visible_to_str = [
            str(r.value) if isinstance(r, Enum) else str(r) for r in visible_to
        ]
        # Ensure json_schema_extra exists and is a dictionary
        json_schema_extra = field_kwargs.get("json_schema_extra")
        if not isinstance(json_schema_extra, dict):
            json_schema_extra = {}
        # Store the string role list under the 'visible_to' key
        json_schema_extra["visible_to"] = visible_to_str
        field_kwargs["json_schema_extra"] = json_schema_extra

    # Create and return the standard Pydantic FieldInfo object
    return Field(**field_kwargs)


def configure_roles(
    *,
    role_enum: Type[Enum],
    inheritance: Optional[Dict[Any, Any]] = None,
    default_role: Optional[Union[Enum, str]] = None,
) -> None:
    """
    Configure the global role system for `VisibleFieldsMixin`.

    This must be called once, typically during application startup, before
    using models that rely on role-based visibility.

    Args:
        role_enum: The Enum class defining all possible application roles.
        inheritance: A dictionary defining role inheritance relationships.
            Keys are roles, and values are lists of roles whose permissions
            they inherit. Roles can be specified as Enum members or their
            string values. Example: `{Role.ADMIN: [Role.EDITOR]}` means
            ADMIN can see everything EDITOR can see.
        default_role: The default role (Enum member or string value) to use
            when `visible_dict` or `to_response_model` is called without an
            explicit role. If None, only fields explicitly visible to an
            empty string role "" (or universally visible fields) are shown
            by default.

    Raises:
        TypeError: If `role_enum` is not a subclass of `Enum`.
    """
    global _ROLE_ENUM, _ROLE_INHERITANCE, _DEFAULT_ROLE

    if not issubclass(role_enum, Enum):
        raise TypeError("role_enum must be an Enum subclass.")

    _ROLE_ENUM = role_enum
    _ROLE_INHERITANCE = {}
    if inheritance:
        # Convert all role keys and values in the inheritance map to strings
        _ROLE_INHERITANCE = {
            (str(r.value) if isinstance(r, Enum) else str(r)): [
                str(ir.value) if isinstance(ir, Enum) else str(ir)
                for ir in inherited_roles
            ]
            for r, inherited_roles in inheritance.items()
        }

    # Convert the default role to its string representation
    if default_role is None:
        _DEFAULT_ROLE = None
    elif isinstance(default_role, Enum):
        _DEFAULT_ROLE = str(default_role.value)
    else:
        _DEFAULT_ROLE = str(default_role)

    # Clear the cache whenever roles are reconfigured
    _RESPONSE_MODEL_CACHE.clear()
    logger.info(
        f"Roles configured: Enum={role_enum.__name__}, "
        f"Inheritance={bool(inheritance)}, Default='{_DEFAULT_ROLE}'"
    )


def visible_fields_response(model: Any, role: Any = None) -> Any:
    """
    Convert a model instance or collection to its role-specific representation.

    If the input `model` is an instance of `VisibleFieldsMixin`, this calls
    its `to_response_model` method with the specified `role`. If it's a list
    or dictionary, it recursively calls itself on the elements/values.
    Otherwise, it returns the input object unchanged.

    Args:
        model: The model instance, list, or dict to convert.
        role: The target role (Enum member or string value) for visibility
            filtering. Uses the globally configured default role if None.

    Returns:
        The role-specific response model instance(s), or the original
        object/collection if no conversion is applicable.
    """
    if isinstance(model, VisibleFieldsMixin):
        # Convert the role argument to its string form for the mixin method
        role_str: Optional[str]
        if role is None:
            role_str = _DEFAULT_ROLE
        elif isinstance(role, Enum):
            role_str = str(role.value)
        else:
            role_str = str(role)
        # Call the instance's method to get the filtered response model
        return model.to_response_model(role=role_str)
    if isinstance(model, list):
        # Recursively process each item in the list
        return [visible_fields_response(item, role) for item in model]
    if isinstance(model, dict):
        # Recursively process each value in the dictionary
        return {k: visible_fields_response(v, role) for k, v in model.items()}
    # Return non-convertible items (primitives, other objects) directly
    return model


class VisibleFieldsMixin:
    """
    Mixin class adding methods for role-based field visibility and conversion.

    Apply this mixin to Pydantic models to enable filtering based on roles.
    It works with `field(visible_to=...)` or a `_role_visible_fields` map.

    Provides `visible_dict` for filtered dictionaries and `to_response_model`
    for creating validated, role-specific Pydantic model instances.
    """

    # Pydantic populates this ClassVar with FieldInfo objects for the model
    model_fields: ClassVar[Dict[str, FieldInfo]]
    # ClassVar storing role-to-visible-field-names mapping, initialized later
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {}

    @property
    def _role_inheritance(self) -> Dict[str, List[str]]:
        """Read-only access to the globally configured role inheritance map."""
        return _ROLE_INHERITANCE

    @property
    def _default_role(self) -> Optional[str]:
        """Read-only access to the globally configured default role string."""
        return _DEFAULT_ROLE

    def visible_dict(
        self,
        role: Optional[Union[Enum, str]] = None,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a dictionary containing only fields visible to the role.

        Recursively converts nested `VisibleFieldsMixin` instances and handles
        circular references by inserting a marker dictionary.

        Args:
            role: The target role identifier (Enum, string, or None). Uses the
                configured default role if None.
            visited: Internal dictionary used for cycle detection during
                recursion. Maps object id() to its already generated dict.
            depth: Internal recursion depth counter (primarily for debugging).

        Returns:
            A dictionary representation of the model, filtered by role.
            Detected cycles are represented as:
            `{'__cycle_reference__': True, 'id': <optional_id>}`.
        """
        # Determine the effective role string to use internally
        role_str: str
        if role is None:
            # Use configured default, fallback to empty string if no default
            role_str = self._default_role or ""
        elif isinstance(role, Enum):
            role_str = str(role.value)
        else:
            role_str = str(role)

        # Initialize visited dict for the top-level call
        if visited is None:
            visited = {}

        obj_id = id(self)
        # Check if this object instance is already being processed (cycle)
        if obj_id in visited:
            logger.debug(
                f"Cycle detected for {self.__class__.__name__} "
                f"(id: {obj_id}) at depth {depth}."
            )
            # Return the cached result (could be placeholder or final dict)
            cycle_data = visited[obj_id]
            # Ensure the marker is present if returning a placeholder
            if cycle_data.get("__processing_placeholder__"):
                # Explicitly type the marker dict
                marker: Dict[str, Any] = {"__cycle_reference__": True}
                try:
                    # Try to add 'id' field if model has one
                    obj_instance_id = getattr(self, "id", PydanticUndefined)
                    if obj_instance_id is not PydanticUndefined:
                        # Assign value safely
                        marker["id"] = obj_instance_id
                except AttributeError:
                    pass
                return marker
            else:
                # Return the previously computed full dict for this object
                return cycle_data

        # Mark current object as visited with a temporary placeholder
        # Explicitly type the placeholder dict
        temp_placeholder: Dict[str, Any] = {"__processing_placeholder__": True}
        try:
            # Add 'id' to placeholder if available, aids debugging cycles
            placeholder_id = getattr(self, "id", PydanticUndefined)
            if placeholder_id is not PydanticUndefined:
                # Assign value safely
                temp_placeholder["id"] = placeholder_id
        except AttributeError:
            pass
        visited[obj_id] = temp_placeholder

        result_dict: Dict[str, Any] = {}
        # Get all field names visible to this role (including inherited)
        visible_field_names = self.__class__._get_all_visible_fields(role_str)

        for field_name in visible_field_names:
            # Field existence already checked in _get_all_visible_fields
            try:
                value = getattr(self, field_name)
                # Recursively convert the field's value
                result_dict[field_name] = self._convert_value_to_dict_recursive(
                    value, role_str, visited, depth + 1
                )
            except AttributeError:
                # Should be rare if _get_all_visible_fields is correct
                logger.warning(
                    f"AttributeError getting field '{field_name}' for role "
                    f"'{role_str}' on {self.__class__.__name__}, "
                    f"though listed as visible."
                )

        # Replace placeholder with the final computed dict for this object
        visited[obj_id] = result_dict
        return result_dict

    def _convert_value_to_dict_recursive(
        self,
        value: Any,
        role: str,
        visited: Dict[int, Dict[str, Any]],
        depth: int = 0,
    ) -> Any:
        """
        Recursively convert values for `visible_dict`, handling nested types.

        Args:
            value: The value to convert.
            role: The current role string.
            visited: The dictionary tracking visited object IDs for cycles.
            depth: Current recursion depth.

        Returns:
            The converted value (e.g., dict, list, primitive).
        """
        if value is None:
            return None

        # Nested VisibleFieldsMixin: Recurse using visible_dict
        if isinstance(value, VisibleFieldsMixin):
            return value.visible_dict(role, visited, depth)

        # Other Pydantic models: Dump to dict without role filtering
        if isinstance(value, BaseModel):
            try:
                # Use standard model_dump
                return value.model_dump()
            except Exception:
                logger.warning(
                    f"Could not dump nested BaseModel: {value!r}", exc_info=True
                )
                return str(value)  # Fallback to string representation

        # Lists: Recursively convert each item
        if isinstance(value, list):
            return [
                self._convert_value_to_dict_recursive(item, role, visited, depth + 1)
                for item in value
            ]

        # Dictionaries: Recursively convert each value
        if isinstance(value, dict):
            return {
                k: self._convert_value_to_dict_recursive(v, role, visited, depth + 1)
                for k, v in value.items()
            }

        # Enums, primitives, etc.: Return as is
        return value

    @classmethod
    def _get_all_visible_fields(cls, role: str) -> Set[str]:
        """
        Calculate the effective set of visible field names for a role.

        Considers fields directly visible to the role, fields visible to
        roles inherited by the target role, and fields inherited from base
        classes that also use `VisibleFieldsMixin`. Filters result against
        actual fields defined in `cls.model_fields`.

        Args:
            role: The target role identifier (string).

        Returns:
            A set containing all field names visible to the role for this class.
        """
        if not hasattr(cls, "_role_visible_fields"):
            # Ensure the class attribute exists, even if empty
            cls._role_visible_fields = {}

        # Start with fields directly visible to the role in this class
        visible_fields = set(cls._role_visible_fields.get(role, set()))

        # Gather fields from inherited roles using the global inheritance map
        roles_to_process = set(_ROLE_INHERITANCE.get(role, []))
        processed_inheritance_roles = {role}  # Avoid checking the role itself

        while roles_to_process:
            inherited_role = roles_to_process.pop()
            if inherited_role in processed_inheritance_roles:
                continue  # Avoid cycles and redundant checks
            processed_inheritance_roles.add(inherited_role)

            # Add fields visible to this inherited role *in the current class*
            visible_fields.update(cls._role_visible_fields.get(inherited_role, set()))
            # Add roles that *this* inherited role inherits from
            roles_to_process.update(_ROLE_INHERITANCE.get(inherited_role, []))

        # Add fields from base classes in the Method Resolution Order (MRO)
        for base in cls.__mro__[1:]:  # Skip cls itself
            # Stop if we reach the mixin, BaseModel, or object
            if base is VisibleFieldsMixin or base is BaseModel or base is object:
                continue
            # Check if the base is also a VisibleFieldsMixin user
            if issubclass(base, VisibleFieldsMixin) and hasattr(
                base, "_role_visible_fields"
            ):
                # Recursively call on the base class to get *its* fully resolved
                # set of visible fields for the role (handles inheritance within base)
                visible_fields.update(base._get_all_visible_fields(role))

        # Final filter: Ensure all listed fields actually exist on this model
        return {f for f in visible_fields if f in cls.model_fields}

    @classmethod
    def _get_recursive_response_type(
        cls,
        annotation: Type[Any],
        role: str,
        model_name_suffix: str,
        visited_fwd_refs: Optional[Set[str]] = None,
        # Add tracker for ongoing model creations
        creating_models: Optional[Set[Tuple[str, str, str]]] = None,
    ) -> Any:
        """
        Determine the appropriate type annotation for a response model field.

        Handles nesting, generics (List, Dict, Union, Optional), ForwardRefs,
        and nested `VisibleFieldsMixin` models recursively. Includes checks
        to prevent infinite recursion during model creation. Falls back to Any
        for ForwardRefs to ensure robustness against resolution errors.

        Args:
            annotation: The original type annotation of the field.
            role: The target role identifier (string).
            model_name_suffix: Suffix for generated response model names.
            visited_fwd_refs: Set tracking visited forward references to
                detect type definition cycles.
            creating_models: Set tracking response models currently being
                created to detect model creation cycles.

        Returns:
            The calculated type annotation (e.g., `int`, `List[str]`,
            `Optional[NestedResponseModel]`, `Any`). Returns `Any` for
            ForwardRefs or to break creation cycles.
        """
        if visited_fwd_refs is None:
            visited_fwd_refs = set()
        if creating_models is None:
            creating_models = set()

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types (like Optional[T] or Union[A, B])
        if origin is Union:
            processed_args = []
            none_present = False
            for arg in args:
                if arg is type(None):
                    none_present = True
                    continue
                processed_args.append(
                    cls._get_recursive_response_type(
                        arg,
                        role,
                        model_name_suffix,
                        visited_fwd_refs.copy(),
                        creating_models=creating_models,
                    )
                )

            valid_processed_args = list(dict.fromkeys(processed_args))
            if Any in valid_processed_args and type(None) in valid_processed_args:
                valid_processed_args.remove(type(None))
            if not valid_processed_args:
                return Optional[Any] if none_present else Any

            if none_present:
                if not valid_processed_args:
                    return Optional[Any]
                if len(valid_processed_args) == 1:
                    return Optional[valid_processed_args[0]]
                return Optional[Union[tuple(valid_processed_args)]]
            else:
                if len(valid_processed_args) == 1:
                    return valid_processed_args[0]
                return Union[tuple(valid_processed_args)]

        # Handle List[T] -> List[ProcessedT]
        if origin is list or origin is List:
            if not args:
                return List[Any]
            nested_type = cls._get_recursive_response_type(
                args[0],
                role,
                model_name_suffix,
                visited_fwd_refs.copy(),
                creating_models=creating_models,
            )
            return List[nested_type]  # type: ignore[valid-type]

        # Handle Dict[K, V] -> Dict[K, ProcessedV]
        if origin is dict or origin is Dict:
            if not args or len(args) != 2:
                return Dict[Any, Any]
            key_type = args[0]
            value_type = cls._get_recursive_response_type(
                args[1],
                role,
                model_name_suffix,
                visited_fwd_refs.copy(),
                creating_models=creating_models,
            )
            return Dict[key_type, value_type]  # type: ignore[valid-type]

        # Handle Forward References (e.g., type hints as strings)
        if isinstance(annotation, ForwardRef):
            fwd_arg = annotation.__forward_arg__  # type: ignore[unreachable]
            if fwd_arg in visited_fwd_refs:
                logger.debug(
                    f"Cycle detected resolving ForwardRef "
                    f"'{fwd_arg}', returning Any."
                )
                return Any
            visited_fwd_refs.add(fwd_arg)

            # Pragmatic Choice: Return Any for all ForwardRefs
            # Avoids complex evaluation and potential NameErrors, but loses type info.
            logger.debug(
                f"Treating ForwardRef '{fwd_arg}' as Any "
                f"for response model type hint."
            )
            # Ensure removal from visited set even though we return Any
            # This is important if the same ForwardRef appears elsewhere non-cyclically
            visited_fwd_refs.remove(fwd_arg)
            return Any

        # Handle nested models that use VisibleFieldsMixin
        if isinstance(annotation, type) and issubclass(annotation, VisibleFieldsMixin):
            model_key = (annotation.__name__, role, model_name_suffix)
            if model_key in creating_models:
                logger.warning(
                    f"Detected creation recursion for {model_key}. "
                    f"Returning Any type to break loop."
                )
                return Any

            # Pass the tracker down when creating nested response models
            return annotation.create_response_model(
                role, model_name_suffix, creating_models=creating_models
            )

        # Handle other nested Pydantic models (not using the mixin)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            logger.debug(
                f"Treating non-mixin BaseModel "
                f"{annotation.__name__} as Dict[str, Any] "
                f"in response type hint for role '{role}'."
            )
            return Dict[str, Any]

        # Return primitive types (int, str) or other unmodified annotations
        return annotation

    @classmethod
    def create_response_model(
        cls,
        role: str,
        model_name_suffix: str = "Response",
        # Add creating_models tracker argument
        creating_models: Optional[Set[Tuple[str, str, str]]] = None,
    ) -> Type[BaseModel]:
        """
        Create (or retrieve from cache) a Pydantic model for a specific role.

        Defines a new Pydantic model containing only the fields visible to
        the given role, with appropriate nested types and copied constraints.
        Includes logic to prevent infinite recursion during creation of
        mutually dependent response models.

        Args:
            role: The target role identifier (string).
            model_name_suffix: Suffix for generated response model names.
            creating_models: Internal set used to track models currently
                under creation to detect recursion.

        Returns:
            The dynamically created Pydantic model Type, or `Any` if a
            creation cycle is detected involving this specific model key.

        Raises:
            ValueError: If `create_model` or `model_rebuild` fails.
        """
        # Initialize tracker if it's the top-level call for this path
        if creating_models is None:
            creating_models = set()

        cache_key = (cls.__name__, role, model_name_suffix)

        # --- Check for creation recursion BEFORE cache check ---
        if cache_key in creating_models:
            logger.warning(
                f"Detected direct creation recursion for {cache_key}. "
                f"Returning Any type to break loop."
            )
            # Return Any to break the cycle. Needs cast for type checker.
            return cast(Type[BaseModel], Any)

        cached_model = _RESPONSE_MODEL_CACHE.get(cache_key)
        if cached_model:
            return cached_model

        logger.debug(
            f"Creating response model for {cls.__name__}, "
            f"role='{role}', suffix='{model_name_suffix}'"
        )

        # --- Add current model to the tracker BEFORE processing fields ---
        creating_models.add(cache_key)
        # --------------------------------------------------------------

        # Get visible fields and prepare definitions
        visible_fields = cls._get_all_visible_fields(role)
        new_fields_definition: Dict[str, Tuple[Any, Any]] = {}
        visited_fwd_refs_for_creation: Set[str] = set()
        response_model: Type[BaseModel]  # Define variable before try block

        try:  # Wrap the creation process in try/finally to remove from tracker
            # --- Loop to build new_fields_definition ---
            for field_name in visible_fields:
                original_field_info = cls.model_fields[field_name]
                original_annotation = original_field_info.annotation

                if original_annotation is None:
                    logger.warning(
                        f"Field '{field_name}' in {cls.__name__} has "
                        f"no annotation, skipping."
                    )
                    continue

                # --- Pass the tracker down when resolving types ---
                response_annotation = cls._get_recursive_response_type(
                    original_annotation,
                    role,
                    model_name_suffix,
                    visited_fwd_refs_for_creation,
                    creating_models=creating_models,  # Pass tracker
                )
                # ---------------------------------------------------

                # --- Determine Field Definition ---
                field_definition_value: Any
                needs_field_wrapper = False
                field_kwargs = {}

                # 1. Handle Default/Required
                if original_field_info.is_required():
                    field_definition_value = ...
                else:
                    if original_field_info.default is not PydanticUndefined:
                        field_kwargs["default"] = original_field_info.default
                        needs_field_wrapper = True
                    elif original_field_info.default_factory is not None:
                        field_kwargs["default_factory"] = (
                            original_field_info.default_factory
                        )
                        needs_field_wrapper = True
                    else:
                        field_kwargs["default"] = None
                        needs_field_wrapper = True

                # 2. Copy Metadata/Constraints
                if original_field_info.description:
                    field_kwargs["description"] = original_field_info.description
                    needs_field_wrapper = True
                if original_field_info.title:
                    field_kwargs["title"] = original_field_info.title
                    needs_field_wrapper = True
                if (
                    original_field_info.alias
                    and original_field_info.alias != field_name
                ):
                    field_kwargs["alias"] = original_field_info.alias
                    needs_field_wrapper = True
                if original_field_info.examples:
                    field_kwargs["examples"] = original_field_info.examples
                    needs_field_wrapper = True

                original_extra_maybe = original_field_info.json_schema_extra
                extra_to_copy: Dict[str, Any] = {}
                if isinstance(original_extra_maybe, dict):
                    extra_to_copy = {
                        k: v
                        for k, v in original_extra_maybe.items()
                        if k != "visible_to"
                    }
                    if extra_to_copy:
                        field_kwargs["json_schema_extra"] = extra_to_copy
                        needs_field_wrapper = True
                elif original_extra_maybe is not None:
                    logger.warning(
                        f"Cannot copy non-dict json_schema_extra "
                        f"for field {field_name}"
                    )

                is_resolved_to_any = response_annotation is Any or (
                    get_origin(response_annotation) is Union
                    and Any in get_args(response_annotation)
                )

                for constraint_key in [
                    "gt",
                    "ge",
                    "lt",
                    "le",
                    "multiple_of",
                    "min_length",
                    "max_length",
                    "min_items",
                    "max_items",
                    "discriminator",
                    "frozen",
                    "strict",
                ]:
                    if constraint_key == "discriminator" and is_resolved_to_any:
                        logger.debug(
                            f"Skipping discriminator copy for field "
                            f"'{field_name}' because type resolved to Any."
                        )
                        continue
                    val = getattr(
                        original_field_info, constraint_key, PydanticUndefined
                    )
                    if val is not PydanticUndefined and val is not None:
                        field_kwargs[constraint_key] = val
                        needs_field_wrapper = True

                pattern_val = getattr(original_field_info, "pattern", PydanticUndefined)
                if pattern_val is not PydanticUndefined and pattern_val is not None:
                    pattern_str = None
                    if isinstance(pattern_val, str):
                        pattern_str = pattern_val
                    elif hasattr(pattern_val, "pattern"):
                        pattern_str = pattern_val.pattern
                    if pattern_str:
                        field_kwargs["pattern"] = pattern_str
                        needs_field_wrapper = True
                    else:
                        logger.warning(
                            f"Could not serialize pattern constraint "
                            f"{pattern_val!r} for field {field_name}"
                        )

                if original_field_info.metadata:
                    constraint_map = {
                        "gt": "gt",
                        "ge": "ge",
                        "lt": "lt",
                        "le": "le",
                        "multiple_of": "multiple_of",
                        "min_length": "min_length",
                        "max_length": "max_length",
                        "pattern": "pattern",
                        "min_items": "min_items",
                        "max_items": "max_items",
                        "strict": "strict",
                        "lower_case": "lower_case",
                        "upper_case": "upper_case",
                    }
                    for meta_item in original_field_info.metadata:
                        for meta_attr, field_kwarg in constraint_map.items():
                            if hasattr(meta_item, meta_attr):
                                if field_kwarg not in field_kwargs:
                                    constraint_val = getattr(meta_item, meta_attr)
                                    if (
                                        field_kwarg == "pattern"
                                        and not isinstance(constraint_val, str)
                                        and hasattr(constraint_val, "pattern")
                                    ):
                                        constraint_val = constraint_val.pattern
                                    if constraint_val is not None:
                                        field_kwargs[field_kwarg] = constraint_val
                                        needs_field_wrapper = True
                                        logger.debug(
                                            f"Copied constraint '{field_kwarg}="
                                            f"{constraint_val}' from metadata "
                                            f"for field '{field_name}'"
                                        )

                # 3. Create Field() object or use default
                if needs_field_wrapper:
                    field_definition_value = Field(**field_kwargs)
                elif original_field_info.is_required():
                    field_definition_value = ...
                else:
                    field_definition_value = None

                new_fields_definition[field_name] = (
                    response_annotation,
                    field_definition_value,
                )
            # --- End loop ---

            # Determine model name (unchanged)
            default_role_str = _DEFAULT_ROLE or ""
            if role == default_role_str:
                model_name = f"{cls.__name__}{model_name_suffix}"
            else:
                role_suffix = role.replace("_", " ").title().replace(" ", "")
                model_name = f"{cls.__name__}{role_suffix}{model_name_suffix}"

            # Create the Pydantic model type
            try:
                model_config = ConfigDict(
                    extra="ignore",
                    populate_by_name=True,
                    arbitrary_types_allowed=True,
                )
                response_model = create_model(
                    model_name, __config__=model_config, **new_fields_definition
                )  # type: ignore[call-overload]

                # Force resolution of forward references etc.
                response_model.model_rebuild(force=True)

            except Exception as e:
                # (Keep existing detailed error handling)
                problematic_field = "Unknown"
                for fname, fdef in new_fields_definition.items():
                    try:
                        test_config = ConfigDict(
                            extra="ignore", arbitrary_types_allowed=True
                        )
                        temp_model = create_model(
                            f"_Test_{fname}", __config__=test_config, **{fname: fdef}
                        )  # type: ignore[call-overload]
                        temp_model.model_rebuild(force=True)
                    except Exception as test_e:
                        problematic_field = f"{fname}: {fdef!r} (Error: {test_e})"
                        break
                error_context = (
                    f" Problem likely near field definition: {problematic_field}."
                )
                logger.error(
                    f"Failed to create/rebuild response model {model_name}."
                    f"{error_context}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Failed to create/rebuild response model {model_name}."
                    f"{error_context} Original error: {e}"
                ) from e

            # Cache the successfully created and rebuilt model type
            _RESPONSE_MODEL_CACHE[cache_key] = response_model
            logger.debug(
                f"Successfully created, rebuilt, and cached response model: "
                f"{model_name}"
            )
            return response_model

        finally:
            # --- Ensure current model is removed from tracker ---
            creating_models.remove(cache_key)
            # ----------------------------------------------------

    @classmethod
    def _construct_nested_model(
        cls, value: Dict[str, Any], annotation: Type[BaseModel]
    ) -> Any:
        """
        Construct a nested BaseModel instance using `model_construct`.
        (Method body unchanged)
        """
        inner_data_for_construct = {}
        target_model_fields = annotation.model_fields

        for nested_field_name, nested_field_info in target_model_fields.items():
            nested_value = PydanticUndefined
            if nested_field_name in value:
                nested_value = value[nested_field_name]
            elif nested_field_info.alias and nested_field_info.alias in value:
                nested_value = value[nested_field_info.alias]

            if nested_value is not PydanticUndefined:
                processed_nested_value = cls._process_value_for_construct_recursive(
                    nested_value, nested_field_info.annotation
                )
                inner_data_for_construct[nested_field_name] = processed_nested_value

        try:
            return annotation.model_construct(**inner_data_for_construct)
        except Exception:
            logger.error(
                f"Failed model_construct for {annotation.__name__} with data "
                f"{inner_data_for_construct!r}",
                exc_info=True,
            )
            return value

    @classmethod
    def _process_value_for_construct_recursive(
        cls, value: Any, annotation: Optional[Type[Any]]
    ) -> Any:
        """
        Recursively prepare data for construction or validation.
        (Method body unchanged)
        """
        if value is None:
            return None
        if annotation is None:
            logger.debug("Processing value with None annotation, returning as is.")
            return value
        if isinstance(value, dict) and value.get("__cycle_reference__") is True:
            logger.debug(
                f"Passing cycle marker through for {annotation}. "
                f"Returning None for construction."
            )
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union:
            possible_types = [arg for arg in args if arg is not type(None)]
            for possible_type in possible_types:
                if isinstance(possible_type, type) and isinstance(value, possible_type):
                    return value

            if isinstance(value, dict):
                potential_models = [
                    t
                    for t in possible_types
                    if isinstance(t, type) and issubclass(t, BaseModel)
                ]
                if len(potential_models) == 1:
                    matched_model_type = potential_models[0]
                    try:
                        constructed = cls._construct_nested_model(
                            value, matched_model_type
                        )
                        if isinstance(constructed, matched_model_type):
                            return constructed
                        else:
                            logger.warning(
                                f"Construction failed for Union "
                                f"type {matched_model_type.__name__}"
                            )
                            return value
                    except Exception:
                        logger.warning(
                            f"Exception during construction for "
                            f"Union type {matched_model_type.__name__}",
                            exc_info=True,
                        )
                        return value
                logger.debug(
                    f"Returning dict for Union {annotation}, "
                    f"no single matching model or construction failed."
                )
                return value

            if len(possible_types) == 1:
                return cls._process_value_for_construct_recursive(
                    value, possible_types[0]
                )
            return value

        if origin is list or origin is List:
            if not args or not isinstance(value, list):
                return value
            nested_annotation = args[0]
            return [
                cls._process_value_for_construct_recursive(item, nested_annotation)
                for item in value
            ]

        if origin is dict or origin is Dict:
            if not args or len(args) != 2 or not isinstance(value, dict):
                return value
            value_annotation = args[1]
            return {
                k: cls._process_value_for_construct_recursive(v, value_annotation)
                for k, v in value.items()
            }

        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and isinstance(value, dict)
        ):
            return cls._construct_nested_model(value, annotation)

        return value

    def to_response_model(self, role: Optional[str] = None) -> BaseModel:
        """
        Convert this model instance to its role-specific response model.

        Generates the filtered dictionary using `visible_dict`, recursively
        replaces internal cycle markers with None, and then validates this
        processed data against the target response model type using
        `model_validate`.

        Args:
            role: The target role identifier (string). Uses the configured
                default role if None.

        Returns:
            An instance of the role-specific response model, validated, with
            cycle markers replaced by None.

        Raises:
            pydantic.ValidationError: If the filtered data fails validation
                against the generated response model schema.
            ValueError: If the response model type itself cannot be created.
        """
        role_str = role or self._default_role or ""
        # Get the dynamically created response model *type*
        # This requires the recursion-breaking fix using `creating_models`
        # and the `model_rebuild` call inside create_response_model.
        model_cls = self.__class__.create_response_model(role_str)

        # Get the dictionary filtered by role visibility (may contain markers)
        visible_data_dict: Dict[str, Any] = self.visible_dict(role_str)

        # --- Helper function to recursively replace cycle markers ---
        def _replace_markers_recursive(data: Any) -> Any:
            """Recursively traverses data, replacing cycle markers with None."""
            if isinstance(data, dict):
                # Check if this dictionary itself is the cycle marker
                if data.get("__cycle_reference__") is True:
                    # Replace the entire marker dict with None
                    return None
                # Otherwise, recurse into its values
                return {
                    key: _replace_markers_recursive(value)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                # Recurse into list items
                return [_replace_markers_recursive(item) for item in data]
            else:
                # Return primitives and other types unchanged
                return data

        # --- End Helper ---

        # --- Preprocess the entire visible_data_dict structure ---
        data_for_validation = _replace_markers_recursive(visible_data_dict)
        # Now, all {'__cycle_reference__': True} dicts are replaced by None
        # --- End Preprocessing ---

        try:
            # Validate the *preprocessed* data against the response model type
            # Pydantic will handle None correctly for Optional fields.
            # If a cycle marker was replaced for a *required* field,
            # model_validate will now raise a "missing required field" error
            # because the value became None, which is the correct behavior.
            return model_cls.model_validate(data_for_validation)
        except Exception as e:
            # Log detailed error and re-raise
            logger.error(
                f"Validation failed for {model_cls.__name__} with preprocessed "
                f"(cycle markers replaced) data: {data_for_validation!r}",
                exc_info=True,
            )
            raise e

    @classmethod
    def configure_visibility(
        cls, role: Union[Enum, str], visible_fields: Set[str]
    ) -> None:
        """
        Dynamically configure field visibility for a specific role on this class.
        (Method body unchanged)
        """
        if not hasattr(cls, "_role_visible_fields"):
            cls._role_visible_fields = {}

        role_key: str
        if isinstance(role, Enum):
            role_key = str(role.value)
        else:
            role_key = str(role)

        cls._role_visible_fields[role_key] = set(visible_fields)

        keys_to_remove = [k for k in _RESPONSE_MODEL_CACHE if k[0] == cls.__name__]
        if keys_to_remove:
            logger.debug(
                f"Configuring visibility for {cls.__name__}, role "
                f"'{role_key}'. Clearing {len(keys_to_remove)} cache "
                f"entries."
            )
            for key in keys_to_remove:
                del _RESPONSE_MODEL_CACHE[key]


class VisibleFieldsModel(BaseModel, VisibleFieldsMixin):
    """
    Base class for Pydantic models supporting role-based field visibility.
    (Class body unchanged)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """
        Initialize role visibility rules when a subclass is defined.
        (Method body unchanged)
        """
        super().__pydantic_init_subclass__(**kwargs)

        base_vis_fields: Dict[str, Set[str]] = {}
        for base in reversed(cls.__mro__):
            if (
                issubclass(base, VisibleFieldsMixin)
                and base is not VisibleFieldsMixin
                and base is not cls
                and hasattr(base, "_role_visible_fields")
            ):
                current_base_fields = getattr(base, "_role_visible_fields", None)
                if isinstance(current_base_fields, dict):
                    for r, flds in current_base_fields.items():
                        role_key = str(r.value) if isinstance(r, Enum) else str(r)
                        if role_key not in base_vis_fields:
                            base_vis_fields[role_key] = set()
                        if isinstance(flds, set):
                            base_vis_fields[role_key].update(flds)

        cls._role_visible_fields = {k: v.copy() for k, v in base_vis_fields.items()}

        valid_role_values: Set[str] = set()
        if _ROLE_ENUM:
            valid_role_values = {str(r.value) for r in _ROLE_ENUM}
            for role_value in valid_role_values:
                if role_value not in cls._role_visible_fields:
                    cls._role_visible_fields[role_value] = set()

        cls_annotations = getattr(cls, "__annotations__", {})
        for field_name, field_info in cls.model_fields.items():
            is_defined_on_cls = field_name in cls_annotations

            if is_defined_on_cls:
                json_schema_extra = getattr(field_info, "json_schema_extra", None)
                visible_to_roles = []
                if isinstance(json_schema_extra, dict):
                    visible_to_raw = json_schema_extra.get("visible_to")
                    if isinstance(visible_to_raw, list):
                        visible_to_roles = [
                            str(r.value) if isinstance(r, Enum) else str(r)
                            for r in visible_to_raw
                        ]

                if visible_to_roles:
                    for role_key in visible_to_roles:
                        if _ROLE_ENUM:
                            if role_key in valid_role_values:
                                if role_key not in cls._role_visible_fields:
                                    cls._role_visible_fields[role_key] = set()
                                cls._role_visible_fields[role_key].add(field_name)
                            else:
                                logger.warning(
                                    f"Role '{role_key}' used in 'visible_to' "
                                    f"for field '{cls.__name__}.{field_name}' "
                                    f"is not defined in configured Role Enum "
                                    f"'{_ROLE_ENUM.__name__}'."
                                )
                        else:
                            logger.warning(
                                f"Role '{role_key}' used in 'visible_to' for "
                                f"field '{cls.__name__}.{field_name}' but no "
                                f"Role Enum is configured globally."
                            )
                            if role_key not in cls._role_visible_fields:
                                cls._role_visible_fields[role_key] = set()
                            cls._role_visible_fields[role_key].add(field_name)

        logger.debug(
            f"Initialized visibility for {cls.__name__}: " f"{cls._role_visible_fields}"
        )
