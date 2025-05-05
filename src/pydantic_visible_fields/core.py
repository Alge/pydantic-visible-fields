"""
Module for role-based field visibility for Pydantic models.

This module provides a mixin (`VisibleFieldsMixin`) and supporting functions
(`configure_roles`, `field`, `visible_fields_response`) to restrict which fields
are included when converting Pydantic models to other formats (like dictionaries
or specific response models), based on a user's assigned role.

It supports role inheritance and dynamically creates filtered Pydantic models
for type safety and integration with frameworks like FastAPI.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    ForwardRef,
    List,
    Literal,
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

logger = logging.getLogger(__name__)

# --- Global Role Configuration ---
# These variables store the role system's configuration.
_ROLE_ENUM: Optional[Type[Enum]] = None
_ROLE_INHERITANCE: Dict[str, List[str]] = {}
_DEFAULT_ROLE: Optional[str] = None
# Cache for dynamically generated response model types.
_RESPONSE_MODEL_CACHE: Dict[Tuple[str, str, str], Type[BaseModel]] = {}
# --- End Global Role Configuration ---

T = TypeVar("T", bound=BaseModel)
ModelT = TypeVar("ModelT", bound="VisibleFieldsModel")


def field(*, visible_to: Optional[List[Any]] = None, **kwargs: Any) -> Any:
    """
    Custom `pydantic.Field` replacement enabling role-based visibility.

    Use this instead of `pydantic.Field` to specify which roles can see a field.
    The visibility information is stored in the field's metadata.

    Args:
        visible_to: A list of role identifiers (e.g., Enum members or strings)
                    that are allowed to view this field.
        **kwargs: Any other keyword arguments accepted by `pydantic.Field`.

    Returns:
        A Pydantic FieldInfo object configured with visibility metadata.
    """
    field_kwargs = kwargs.copy()

    if visible_to is not None:
        # Store role identifiers as strings in metadata for consistency.
        visible_to_str = [r.value if isinstance(r, Enum) else r for r in visible_to]
        # Ensure json_schema_extra exists and is a dict.
        json_schema_extra = field_kwargs.get("json_schema_extra")
        if not isinstance(json_schema_extra, dict):
            json_schema_extra = {}
        json_schema_extra["visible_to"] = visible_to_str
        field_kwargs["json_schema_extra"] = json_schema_extra

    return Field(**field_kwargs)


def configure_roles(
    *,
    role_enum: Type[Enum],
    inheritance: Optional[Dict[Any, Any]] = None,
    default_role: Optional[Union[Enum, str]] = None,
) -> None:
    """
    Configure the global role system used by `VisibleFieldsMixin`.

    This must be called once before using models that rely on role visibility.

    Args:
        role_enum: The Enum class defining all possible application roles.
        inheritance: A dictionary defining role inheritance. Keys are roles,
                     and values are lists of roles they inherit permissions from.
                     Example: `{Role.ADMIN: [Role.EDITOR], Role.EDITOR: [Role.VIEWER]}`.
                     Roles can be Enum members or their string values.
        default_role: The default role to use if no specific role is provided
                      during conversion. Can be an Enum member or its string value.
    """
    global _ROLE_ENUM, _ROLE_INHERITANCE, _DEFAULT_ROLE

    if not issubclass(role_enum, Enum):
        raise TypeError("role_enum must be an Enum subclass.")

    _ROLE_ENUM = role_enum
    _ROLE_INHERITANCE = {}
    if inheritance:
        # Convert keys and values to string representations for internal use.
        _ROLE_INHERITANCE = {
            (r.value if isinstance(r, Enum) else str(r)): [
                ir.value if isinstance(ir, Enum) else str(ir) for ir in inherited_roles
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
    Convert a model instance to its role-specific response representation.

    If the model inherits from `VisibleFieldsMixin`, its `to_response_model`
    method is called. Otherwise, the model is returned as is, or processed
    recursively if it's a list or dict.

    Args:
        model: The model instance or collection to convert.
        role: The target role (Enum member or string value) for visibility filtering.
              Uses the configured default role if None.

    Returns:
        The role-specific response model instance, or the original object/collection
        if no conversion is applicable.
    """
    if isinstance(model, VisibleFieldsMixin):
        return model.to_response_model(role=role)
    if isinstance(model, list):
        # Recursively process items in a list
        return [visible_fields_response(item, role) for item in model]
    if isinstance(model, dict):
        # Recursively process values in a dict
        return {k: visible_fields_response(v, role) for k, v in model.items()}
    # Return non-convertible items directly
    return model


class VisibleFieldsMixin:
    """
    Mixin class providing methods for role-based field visibility.

    Apply this mixin to Pydantic models where you need to control which fields
    are included in outputs based on roles. Works in conjunction with the
    `field()` decorator or a `_role_visible_fields` class variable.
    """

    # ClassVars that Pydantic models using this mixin will have.
    model_fields: ClassVar[Dict[str, Any]] # Populated by Pydantic
    # Stores role-to-visible-field-names mapping. Initialized by VisibleFieldsModel.__pydantic_init_subclass__
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {}

    @property
    def _role_inheritance(self) -> Dict[str, List[str]]:
        """Read-only access to the globally configured role inheritance."""
        return _ROLE_INHERITANCE

    @property
    def _default_role(self) -> Optional[str]:
        """Read-only access to the globally configured default role."""
        return _DEFAULT_ROLE

    def visible_dict(
        self,
        role: Optional[str] = None,
        visited: Optional[Dict[int, Dict[str, Any]]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Generates a dictionary containing only fields visible to the specified role.

        Recursively converts nested `VisibleFieldsMixin` instances. Handles cycles.

        Args:
            role: The target role identifier (string). Uses default role if None.
            visited: Internal dict used for cycle detection during recursion.
            depth: Internal recursion depth counter.

        Returns:
            A dictionary representation of the model, filtered by role visibility.
            Includes `{'__cycle_reference__': True, 'id': ...}` for detected cycles.
        """
        role = role or self._default_role or "" # Fallback to empty string if no default
        if visited is None:
            visited = {}

        obj_id = id(self)
        if obj_id in visited:
            # Cycle detected, return marker
            cycle_data = {"__cycle_reference__": True}
            try:
                cycle_data["id"] = getattr(self, "id")
            except AttributeError:
                pass # ID not required, marker is sufficient
            return cycle_data

        # Mark current object as visited (initially with placeholder)
        temp_placeholder = {"__processing_placeholder__": True}
        try:
            temp_placeholder["id"] = getattr(self, "id")
        except AttributeError:
            pass
        visited[obj_id] = temp_placeholder

        result: Dict[str, Any] = {}
        visible_fields = self.__class__._get_all_visible_fields(role)

        for field_name in visible_fields:
            try:
                 value = getattr(self, field_name)
                 # Recursively convert the value
                 result[field_name] = self._convert_value_to_dict_recursive(
                     value, role, visited, depth + 1
                 )
            except AttributeError:
                 logger.warning(f"Field '{field_name}' listed as visible for role '{role}' "
                                f"but not found on instance of {self.__class__.__name__}.")

        # Update visited cache with the final result for this object
        visited[obj_id] = result
        return result

    def _convert_value_to_dict_recursive(
        self,
        value: Any,
        role: str,
        visited: Dict[int, Dict[str, Any]],
        depth: int = 0,
    ) -> Any:
        """
        Internal helper to recursively convert values within `visible_dict`.

        Handles nested `VisibleFieldsMixin` instances, other Pydantic models,
        lists, dicts, Enums, and primitive types.
        """
        if value is None:
            return None

        # Recursively call visible_dict for nested mixin instances
        if isinstance(value, VisibleFieldsMixin):
            return value.visible_dict(role, visited, depth)

        # Dump non-mixin Pydantic models (no role filtering applied here)
        if isinstance(value, BaseModel):
            try:
                 return value.model_dump()
            except Exception:
                 logger.warning(f"Could not dump nested BaseModel: {value!r}", exc_info=True)
                 return str(value) # Fallback representation

        # Recursively process lists
        if isinstance(value, list):
            return [
                self._convert_value_to_dict_recursive(item, role, visited, depth + 1)
                for item in value
            ]

        # Recursively process dict values
        if isinstance(value, dict):
            return {
                k: self._convert_value_to_dict_recursive(v, role, visited, depth + 1)
                for k, v in value.items()
            }

        # Keep Enums and other primitives as they are
        return value

    @classmethod
    def _get_all_visible_fields(cls, role: str) -> Set[str]:
        """
        Calculates the complete set of visible field names for a given role.

        Includes fields directly visible to the role, fields inherited from
        other roles (based on `_ROLE_INHERITANCE`), and fields inherited from
        base classes that also use `VisibleFieldsMixin`.

        Args:
            role: The target role identifier (string).

        Returns:
            A set containing all field names visible to the role.
        """
        if not isinstance(role, str):
             role = str(role.value) if isinstance(role, Enum) else str(role)

        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
             # Should be initialized by __pydantic_init_subclass__
             cls._role_visible_fields = {}

        # Use .get(role, set()) to safely handle roles not explicitly defined for this class
        visible_fields = set(cls._role_visible_fields.get(role, set()))

        # Add fields from inherited roles
        inherited_roles = _ROLE_INHERITANCE.get(role, [])
        for inherited_role in inherited_roles:
            # Prevent infinite recursion if roles inherit cyclically
            # (This assumes _ROLE_INHERITANCE is acyclic)
            visible_fields.update(cls._get_all_visible_fields(inherited_role))

        # Add fields from visible base classes (MRO besides self and direct mixin)
        for base in cls.__mro__[1:]: # Skip cls itself
            if base is VisibleFieldsMixin or base is BaseModel or base is object:
                 continue
            if issubclass(base, VisibleFieldsMixin):
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
        Internal helper to determine the correct type annotation for a field
        in a dynamically generated response model, handling nesting.

        Args:
            annotation: The original type annotation of the field.
            role: The target role identifier (string).
            model_name_suffix: Suffix used for response model names (e.g., "Response").
            visited_fwd_refs: Set to track visited forward references during recursion.

        Returns:
            The calculated type annotation for the response model field.
        """
        if visited_fwd_refs is None:
            visited_fwd_refs = set()

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types (Optional[T] is a specific case)
        if origin is Union:
            is_optional = len(args) == 2 and type(None) in args
            processed_args = []
            none_present = False
            for arg in args:
                if arg is type(None):
                    none_present = True
                    continue
                # Recursively get response type for non-None arguments
                processed_args.append(
                    cls._get_recursive_response_type(
                        arg, role, model_name_suffix, visited_fwd_refs
                    )
                )
            if is_optional:
                # Reconstruct Optional[ProcessedT]
                if not processed_args: return Optional[Any] # Original was Optional[None]? Unlikely
                return Optional[processed_args[0]]
            else:
                # Reconstruct Union[ProcessedT1, ProcessedT2, ...]
                if none_present: processed_args.append(type(None))
                if not processed_args: return Any # Empty Union?
                # Need at least two unique types to form a Union type hint
                unique_processed_args = tuple(dict.fromkeys(processed_args))
                return Union[unique_processed_args] if len(unique_processed_args) > 1 else unique_processed_args[0]


        # Handle List[T] -> List[ProcessedT]
        if origin is list or origin is List:
            if not args: return List[Any]
            nested_type = cls._get_recursive_response_type(
                args[0], role, model_name_suffix, visited_fwd_refs
            )
            return List[nested_type]

        # Handle Dict[K, V] -> Dict[K, ProcessedV]
        if origin is dict or origin is Dict:
            if not args or len(args) != 2: return Dict[Any, Any]
            key_type = args[0] # Assume key type doesn't change
            value_type = cls._get_recursive_response_type(
                args[1], role, model_name_suffix, visited_fwd_refs
            )
            return Dict[key_type, value_type]

        # Handle Forward References
        if isinstance(annotation, ForwardRef):
            fwd_arg = annotation.__forward_arg__
            if fwd_arg in visited_fwd_refs:
                return annotation # Cycle detected, return original ref
            visited_fwd_refs.add(fwd_arg)
            try:
                module_name = getattr(cls, '__module__', '__main__')
                module = sys.modules.get(module_name)
                if module is None:
                     raise NameError(f"Module '{module_name}' not found for ForwardRef '{fwd_arg}'")
                actual_type = annotation._evaluate(module.__dict__, globals(), frozenset())
                # Recursively process the resolved type
                resolved_type = cls._get_recursive_response_type(
                    actual_type, role, model_name_suffix, visited_fwd_refs
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate ForwardRef '{fwd_arg}': {e}, using original.")
                resolved_type = annotation # Fallback
            finally:
                # Ensure removal from visited set even if evaluation failed
                if fwd_arg in visited_fwd_refs:
                    visited_fwd_refs.remove(fwd_arg)
            return resolved_type

        # Handle nested models inheriting from VisibleFieldsMixin
        if isinstance(annotation, type) and issubclass(annotation, VisibleFieldsMixin):
            # Create the specific response model type for the nested model and role
            return annotation.create_response_model(role, model_name_suffix)

        # Handle other Pydantic models (not using the mixin)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # Represent these as simple dictionaries in the response model
            # to avoid potential issues with their internal fields/validation.
            logger.debug(f"Treating non-mixin BaseModel {annotation.__name__} as Dict[str, Any]")
            return Dict[str, Any]

        # Return primitive types or other annotations unmodified
        return annotation

    @classmethod
    def create_response_model(
        cls, role: str, model_name_suffix: str = "Response"
    ) -> Type[BaseModel]:
        """
        Creates or retrieves from cache a Pydantic model definition containing
        only the fields visible to the specified role.

        Recursively defines appropriate types for nested fields (lists, dicts,
        other VisibleFieldsMixin models). Copies field constraints and metadata.

        Args:
            role: The target role identifier (string).
            model_name_suffix: Suffix to append to the original class name for the
                               response model name (default: "Response").

        Returns:
            The dynamically created Pydantic model Type.
        """
        if not isinstance(role, str):
             role = str(role.value) if isinstance(role, Enum) else str(role)

        cache_key = (cls.__name__, role, model_name_suffix)
        if cache_key in _RESPONSE_MODEL_CACHE:
            return _RESPONSE_MODEL_CACHE[cache_key]

        logger.debug(f"Cache miss. Creating response model for {cls.__name__}, role='{role}'")
        visible_fields = cls._get_all_visible_fields(role)
        new_fields_definition: Dict[str, Tuple[Type[Any], Any]] = {}
        visited_fwd_refs_for_creation = set()

        for field_name in visible_fields:
            if field_name not in cls.model_fields:
                logger.warning(f"Field '{field_name}' listed as visible for role '{role}' "
                               f"in {cls.__name__} but not found in model_fields.")
                continue

            original_field_info = cls.model_fields[field_name]
            original_annotation = original_field_info.annotation

            # Determine the correct annotation type for the response model field
            response_annotation = cls._get_recursive_response_type(
                original_annotation, role, model_name_suffix, visited_fwd_refs_for_creation
            )

            # Collect keyword arguments for pydantic.Field()
            field_kwargs = {}
            needs_field_wrapper = False

            # Copy descriptive/display properties
            if original_field_info.description:
                field_kwargs["description"] = original_field_info.description
                needs_field_wrapper = True
            if original_field_info.title:
                 field_kwargs["title"] = original_field_info.title
                 needs_field_wrapper = True
            if original_field_info.alias:
                 field_kwargs["alias"] = original_field_info.alias
                 needs_field_wrapper = True
            if original_field_info.examples:
                field_kwargs["examples"] = original_field_info.examples
                needs_field_wrapper = True

            # Copy constraints by inspecting the metadata list (Pydantic v2 style)
            field_metadata = getattr(original_field_info, 'metadata', None)
            if isinstance(field_metadata, (list, tuple)):
                for constraint_obj in field_metadata:
                    if hasattr(constraint_obj, 'gt'):
                        field_kwargs['gt'] = constraint_obj.gt
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'ge'):
                        field_kwargs['ge'] = constraint_obj.ge
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'lt'):
                        field_kwargs['lt'] = constraint_obj.lt
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'le'):
                        field_kwargs['le'] = constraint_obj.le
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'multiple_of'):
                        field_kwargs['multiple_of'] = constraint_obj.multiple_of
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'min_length') and not hasattr(constraint_obj, 'min_items'):
                        field_kwargs['min_length'] = constraint_obj.min_length
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'max_length') and not hasattr(constraint_obj, 'max_items'):
                        field_kwargs['max_length'] = constraint_obj.max_length
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'pattern'):
                        if isinstance(constraint_obj.pattern, str):
                             field_kwargs['pattern'] = constraint_obj.pattern
                             needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'min_items'):
                        field_kwargs['min_items'] = constraint_obj.min_items
                        needs_field_wrapper = True
                    elif hasattr(constraint_obj, 'max_items'):
                         field_kwargs['max_items'] = constraint_obj.max_items
                         needs_field_wrapper = True

            # Determine required status and copy default/default_factory
            is_required = True
            if original_field_info.default is not PydanticUndefined:
                field_kwargs["default"] = original_field_info.default
                is_required = False
                needs_field_wrapper = True
            elif original_field_info.default_factory is not None:
                field_kwargs["default_factory"] = original_field_info.default_factory
                is_required = False
                needs_field_wrapper = True

            # Assign the final definition value for create_model
            if needs_field_wrapper:
                 field_definition_value = Field(**field_kwargs)
            elif is_required:
                 field_definition_value = ... # Ellipsis for required field without extra settings
            else:
                 # This state (not required, no wrapper needed) is unlikely but use None as fallback
                 field_definition_value = None

            new_fields_definition[field_name] = (response_annotation, field_definition_value)

        # Determine response model name based on role and suffix
        default_role_str = _DEFAULT_ROLE or "" # Handle None case
        if role == default_role_str:
            model_name = f"{cls.__name__}{model_name_suffix}"
        else:
            # Ensure role name is suitable for a class name (e.g., capitalize)
            role_suffix = role.replace("_", " ").title().replace(" ", "")
            model_name = f"{cls.__name__}{role_suffix}{model_name_suffix}"

        # Create the actual model type
        try:
             response_model = create_model(
                 model_name,
                 __config__=ConfigDict(extra='ignore', populate_by_name=True),
                 **new_fields_definition
             )
        except Exception as e:
             # Attempt to identify problematic field definition for better error message
             problematic_field = "Unknown"
             for fname, fdef in new_fields_definition.items():
                  try:
                       _ = create_model(f"_Test_{fname}", __config__=ConfigDict(extra='ignore'), **{fname: fdef})
                  except Exception:
                       problematic_field = f"{fname}: {fdef!r}"
                       break
             error_context = f" Problem likely near field definition: {problematic_field}."
             logger.error(f"Failed to create response model {model_name}.{error_context}", exc_info=True)
             # Raise a more informative error
             raise ValueError(f"Failed to create response model {model_name}.{error_context} Original error: {e}")


        typed_response_model = cast(Type[BaseModel], response_model)
        _RESPONSE_MODEL_CACHE[cache_key] = typed_response_model
        return typed_response_model

    @classmethod
    def _construct_nested_model(cls, value: Dict[str, Any], annotation: Type[BaseModel]) -> Any:
         """Internal helper to construct a nested model instance using model_construct."""
         inner_data_for_construct = {}
         if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
              logger.warning(f"Attempted _construct_nested_model for non-BaseModel type: {annotation}")
              return value # Cannot construct

         for nested_field_name, nested_field_info in annotation.model_fields.items():
             nested_value = PydanticUndefined
             # Check field name first, then alias from input dict 'value'
             if nested_field_name in value:
                 nested_value = value[nested_field_name]
             elif nested_field_info.alias and nested_field_info.alias in value:
                  nested_value = value[nested_field_info.alias]

             if nested_value is not PydanticUndefined:
                  # Recursively process this value before storing for construction
                  processed_nested_value = cls._process_value_for_construct_recursive(
                      nested_value, nested_field_info.annotation
                  )
                  # Store using the field name expected by the model constructor
                  inner_data_for_construct[nested_field_name] = processed_nested_value

         try:
             # Construct the nested model instance without validation
             return annotation.model_construct(**inner_data_for_construct)
         except Exception as e:
             logger.error(f"Failed model_construct for {annotation.__name__} with data {inner_data_for_construct}", exc_info=True)
             return value # Fallback: return original dict if construction fails

    @classmethod
    def _process_value_for_construct_recursive(
            cls, value: Any, annotation: Type[Any]
    ) -> Any:
        """
        Internal helper to recursively prepare data for `model_construct`.

        It processes nested structures (lists, dicts, unions, nested models)
        by calling `_construct_nested_model` where appropriate, ensuring that
        values passed to the final `model_construct` are either primitives or
        already constructed nested model instances.
        """
        if value is None:
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types, including Optional and discriminated unions
        if origin is Union:
            is_optional = len(args) == 2 and type(None) in args
            possible_types = [arg for arg in args if arg is not type(None)]

            if is_optional and len(possible_types) == 1:
                 # Optional[T]
                 return cls._process_value_for_construct_recursive(value, possible_types[0]) if value is not None else None

            # Attempt discriminated union logic if value is dict
            if isinstance(value, dict):
                potential_models = [
                    t for t in possible_types if isinstance(t, type) and issubclass(t, BaseModel)
                ]
                if potential_models:
                    discriminator_field_name = "type"
                    discriminator_value = value.get(discriminator_field_name)
                    matched_model_type = None

                    if discriminator_value is not None:
                        discriminator_str = str(discriminator_value.value) if isinstance(discriminator_value, Enum) else str(discriminator_value)
                        for model_type in potential_models:
                            type_field_info = model_type.model_fields.get(discriminator_field_name)
                            if not type_field_info: continue

                            # Check Literal annotation
                            ann_origin = get_origin(type_field_info.annotation)
                            ann_args = get_args(type_field_info.annotation)
                            if ann_origin is Literal and discriminator_str in ann_args:
                                matched_model_type = model_type
                                break
                            # Check default value
                            if matched_model_type is None:
                                field_default = getattr(type_field_info, 'default', PydanticUndefined)
                                if field_default != PydanticUndefined:
                                    default_str = str(field_default.value) if isinstance(field_default, Enum) else str(field_default)
                                    if default_str == discriminator_str:
                                        matched_model_type = model_type
                                        break

                    # Construct using matched type if found
                    if matched_model_type:
                        try:
                            constructed = cls._construct_nested_model(value, matched_model_type)
                            if isinstance(constructed, matched_model_type):
                                return constructed
                        except Exception:
                             logger.warning(f"Failed construction for matched Union type {matched_model_type.__name__}", exc_info=True)

                # Fallback if no specific model type could be determined/constructed
                logger.debug(f"Returning dict for Union, could not determine/construct specific type: {value!r}")
                return value

            # Value is not a dict, or it's a Union of non-models (e.g., Union[int, str])
            return value

        # Handle List[T]
        if origin is list or origin is List:
            if not args or not isinstance(value, list):
                return value # Cannot process further
            nested_annotation = args[0]
            return [
                cls._process_value_for_construct_recursive(item, nested_annotation)
                for item in value
            ]

        # Handle Dict[K, V]
        if origin is dict or origin is Dict:
            if not args or len(args) != 2 or not isinstance(value, dict):
                return value # Cannot process further
            value_annotation = args[1]
            # Assume key type (args[0]) doesn't need recursive processing
            return {
                k: cls._process_value_for_construct_recursive(v, value_annotation)
                for k, v in value.items()
            }

        # Handle direct nested BaseModel if value is a dictionary
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
             if value.get("__cycle_reference__") is True:
                 return value # Pass cycle marker through
             return cls._construct_nested_model(value, annotation)

        # Return primitives, Enums, already constructed objects, etc. as is
        return value

    def to_response_model(self, role: Optional[str] = None) -> BaseModel:
        """
        Convert this model instance to a role-specific response model instance.

        This method generates a dictionary filtered by role visibility (`visible_dict`),
        recursively prepares this dictionary for instantiation by constructing nested
        response models (`_process_value_for_construct_recursive`), and finally
        validates the prepared data against the target response model type using
        `model_validate`.

        Args:
            role: The target role identifier (string). Uses default role if None.

        Returns:
            An instance of the dynamically created response model, validated against
            its definition.

        Raises:
            pydantic.ValidationError: If the data derived for the given role fails
                                      validation against the generated response model.
            ValueError: If the response model type itself cannot be created.
        """
        role = role or self._default_role or "" # Ensure role is a string
        # Get the dynamically created response model *type* for this class and role
        model_cls = self.__class__.create_response_model(role)
        # Get the dictionary representation filtered by role
        visible_data_dict: Dict[str, Any] = self.visible_dict(role)
        # Prepare data by recursively constructing nested response model *instances*
        data_for_validation: Dict[str, Any] = {}

        for field_name, field_info in model_cls.model_fields.items():
            value_from_dict = PydanticUndefined
            # Check for field by name or alias in the visible dict
            if field_name in visible_data_dict:
                value_from_dict = visible_data_dict[field_name]
            elif field_info.alias and field_info.alias in visible_data_dict:
                value_from_dict = visible_data_dict[field_info.alias]

            # If field exists in visible dict, process it recursively
            if value_from_dict is not PydanticUndefined:
                data_for_validation[field_name] = self._process_value_for_construct_recursive(
                    value_from_dict, field_info.annotation
                )
            # Else: Field is not in visible dict. If it's required in model_cls without
            # a default, model_validate below will raise the appropriate error.

        try:
            # Validate the prepared data (with nested instances) against the response model type
            return model_cls.model_validate(data_for_validation)
        except Exception as e:
            logger.error(f"Validation failed for {model_cls.__name__} with prepared data: {data_for_validation!r}", exc_info=True)
            raise e # Re-raise the validation or other error

    @classmethod
    def configure_visibility(cls, role: str, visible_fields: Set[str]) -> None:
        """
        Dynamically configure the visibility of fields for a specific role on this class.

        Note: This modifies the class attribute `_role_visible_fields` directly.
              Consider thread-safety implications if used in concurrent environments.

        Args:
            role: Role identifier (string) to configure visibility for.
            visible_fields: Set of field names that should be visible to the role.
                            This *replaces* any existing definition for this role.
        """
        if not hasattr(cls, "_role_visible_fields") or cls._role_visible_fields is None:
            cls._role_visible_fields = {}
        if not isinstance(role, str):
             role = str(role.value) if isinstance(role, Enum) else str(role)
        cls._role_visible_fields[role] = set(visible_fields)
        # Clear the response model cache as visibility rules have changed
        keys_to_remove = [k for k in _RESPONSE_MODEL_CACHE if k[0] == cls.__name__]
        for key in keys_to_remove:
             del _RESPONSE_MODEL_CACHE[key]


class VisibleFieldsModel(BaseModel, VisibleFieldsMixin):
    """
    Base class for Pydantic models that require role-based field visibility.

    Inherit from this class instead of `pydantic.BaseModel`. It integrates
    `VisibleFieldsMixin` and automatically initializes the role visibility
    mechanism based on `field(visible_to=...)` decorators used on fields.

    Includes `populate_by_name=True` in its default configuration.
    """
    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """
        Initializes role visibility rules when a subclass is defined.

        Collects visibility rules from base classes and merges them with rules
        defined directly on the subclass using the `field(visible_to=...)` decorator.
        """
        # Properly inherit and merge visibility rules from base classes in MRO
        base_vis_fields: Dict[str, Set[str]] = {}
        for base in reversed(cls.__mro__):
             if issubclass(base, VisibleFieldsMixin) and base is not VisibleFieldsMixin and hasattr(base, "_role_visible_fields"):
                  current_base_fields = getattr(base, "_role_visible_fields", {})
                  if isinstance(current_base_fields, dict):
                    for r, flds in current_base_fields.items():
                            if r not in base_vis_fields:
                                base_vis_fields[r] = set()
                            if isinstance(flds, set):
                                base_vis_fields[r].update(flds)

        # Start with potentially inherited fields
        cls._role_visible_fields = base_vis_fields

        # Ensure all configured roles have at least an empty set initialized
        if _ROLE_ENUM:
            for role_enum_member in _ROLE_ENUM.__members__.values():
                role_value = role_enum_member.value if isinstance(role_enum_member, Enum) else str(role_enum_member)
                if role_value not in cls._role_visible_fields:
                    cls._role_visible_fields[role_value] = set()

        # Add fields defined directly on *this* class using the 'field' decorator
        # Pydantic v2 populates model_fields before this hook runs
        for field_name, field_info in cls.model_fields.items():
             # Check if the field info actually corresponds to a field defined on *this* specific class
             # (to avoid re-processing inherited fields unnecessarily)
             # This check might be complex; relying on json_schema_extra is often sufficient if used consistently.
             is_defined_on_cls = field_name in cls.__annotations__ # Heuristic

             if is_defined_on_cls: # Process only fields directly defined here
                json_schema_extra = getattr(field_info, "json_schema_extra", None)
                if isinstance(json_schema_extra, dict) and "visible_to" in json_schema_extra:
                    visible_to = json_schema_extra["visible_to"]
                    if isinstance(visible_to, list):
                        for role_ref in visible_to:
                            role_key = role_ref.value if isinstance(role_ref, Enum) else str(role_ref)
                            # Ensure the role exists from configuration before adding
                            if _ROLE_ENUM and role_key in [r.value for r in _ROLE_ENUM]:
                                if role_key not in cls._role_visible_fields:
                                    cls._role_visible_fields[role_key] = set()
                                cls._role_visible_fields[role_key].add(field_name)
                            else:
                                logger.warning(f"Role '{role_key}' used in 'visible_to' for field "
                                               f"'{cls.__name__}.{field_name}' is not defined in configured Role Enum.")

        # No need to call super().__pydantic_init_subclass__(**kwargs) for BaseModel