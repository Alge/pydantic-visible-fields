Okay, let's update the README to accurately reflect the current state, including the capabilities and the important limitation regarding module-level dynamic model creation for documentation purposes.

```markdown
# Pydantic Visible Fields

A flexible field-level visibility control system for Pydantic models. This library allows you to define which fields of your models are visible to different user roles, making it easy to implement role-based access control at the data model level, especially for API responses.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.0+-green.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/pydantic-visible-fields.svg)](https://badge.fury.io/py/pydantic-visible-fields)
[![Downloads](https://static.pepy.tech/badge/pydantic-visible-fields)](https://pepy.tech/project/pydantic-visible-fields)

## Overview

`pydantic-visible-fields` provides a simple way to control which fields of your Pydantic models are included when generating responses, based on user roles. It dynamically creates filtered Pydantic models at runtime, ensuring type safety for the visible data.

It also provides a `PaginatedResponse` class which makes it easy to generate paginated API responses with automatic conversion of objects to the correct visibility level.

### Key Features

- **Field-level visibility control** using a simple `field` decorator.
- **Role inheritance** support to define hierarchical permissions.
- **Nested model support** with full recursive visibility filtering.
- **Circular reference handling** replacing cycles with `None` in the output.
- **Pydantic V2 Compatible** leveraging modern Pydantic features.
- **Simple integration** with FastAPI and other web frameworks via a helper function.

## Installation

```bash
pip install pydantic-visible-fields
```

## Basic Usage

### 1. Define Your Roles

First, define your application's roles using Python's `Enum`:

```python
from enum import Enum

class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
```

### 2. Configure Role System

Configure the role system once, usually at application startup. Define inheritance (e.g., ADMIN inherits EDITOR's visibility) and an optional default role.

```python
from pydantic_visible_fields import configure_roles

configure_roles(
    role_enum=Role,
    inheritance={
        Role.ADMIN: [Role.EDITOR], # Admin sees what Editor sees
        Role.EDITOR: [Role.VIEWER], # Editor sees what Viewer sees
    },
    default_role=Role.VIEWER # Role used if none is specified
)
```

### 3. Create Models with Visibility Rules

Inherit from `VisibleFieldsModel` and use the custom `field` function to specify which roles can see each field via the `visible_to` argument.

```python
from pydantic_visible_fields import VisibleFieldsModel, field

class User(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    username: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    email: str = field(visible_to=[Role.EDITOR, Role.ADMIN]) # Only Editor and Admin
    hashed_password: str = field(visible_to=[Role.ADMIN]) # Only Admin
    is_active: bool = field(visible_to=[Role.ADMIN]) # Only Admin
```

### 4. Use in API Responses (Runtime Filtering)

In your API endpoint or application logic, use the `visible_fields_response` helper function to convert your model instance into a role-specific response just before returning it.

```python
from fastapi import FastAPI, Depends
from typing import Any # Use Any for dynamic return type
from pydantic_visible_fields import visible_fields_response
# Assume User model, Role enum, configure_roles, get_user_by_id, get_current_user_role exist

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: str, current_user_role: Role = Depends(get_current_user_role)) -> Any:
    user: User = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Convert the full user model to a filtered response based on the role
    # This dynamically generates and validates the correct response model instance.
    response_data = visible_fields_response(user, role=current_user_role)
    return response_data
```

This approach ensures that complex models with forward references or cycles work reliably, as the response model generation happens at runtime when all types are defined.

**Note on FastAPI Documentation:** When using `visible_fields_response`, the return type hint in your endpoint (`-> Any`) won't allow FastAPI to generate precise OpenAPI documentation for the different role-specific response schemas. See the "FastAPI Integration & Documentation" section for strategies to improve documentation if needed.

## Advanced Usage

### Class-Level Visibility (Alternative)

Instead of decorating each field, you can define visibility rules at the class level using the `VisibleFieldsMixin` and a `_role_visible_fields` class variable. Note that field-level `visible_to` takes precedence if both are used.

```python
from pydantic import BaseModel
from pydantic_visible_fields import VisibleFieldsMixin
from typing import ClassVar, Dict, Set

class UserSettings(BaseModel, VisibleFieldsMixin):
    # Map role *string values* to sets of visible field names
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {
        Role.VIEWER.value: {"id", "theme"},
        Role.EDITOR.value: {"notifications"}, # Editor also inherits Viewer's fields
        Role.ADMIN.value: {"advanced_options", "debug_mode"}, # Admin inherits Editor/Viewer
    }

    id: str
    theme: str
    notifications: bool
    advanced_options: dict
    debug_mode: bool

    # Need to explicitly call model_rebuild if using ForwardRefs within the class
    model_rebuild()
```

### Nested Models

Visibility filtering automatically applies recursively to nested models that also inherit from `VisibleFieldsModel` or `VisibleFieldsMixin`.

```python
class Address(VisibleFieldsModel):
    street: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    city: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    country: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    postal_code: str = field(visible_to=[Role.EDITOR, Role.ADMIN])

class FullUser(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    # ... other user fields ...
    # The address field itself is visible to all, but the *content* of the
    # Address object will be filtered based on the role when processed.
    address: Address = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
```

A Viewer requesting a `FullUser` would see the `address` field containing an `Address` object with only the `city` and `country` fields populated.

### Working with Collections

Visibility filtering also works automatically for models within `List` and `Dict` structures.

```python
from typing import List, Dict

# Assume User and Setting models are defined using VisibleFieldsModel/Mixin
class Team(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    # Each User object in this list will be filtered based on the role
    members: List[User] = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    # Each Setting object in this dict's values will be filtered
    settings: Dict[str, Setting] = field(visible_to=[Role.EDITOR, Role.ADMIN])
```

### Circular References

The library handles circular object references safely. When converting to a dictionary (`visible_dict`) or response model (`to_response_model`), cycles are detected, and the reference causing the loop is replaced with `None` in the final output object instance if the field type allows it (e.g., `Optional[...]`).

### Dynamic Visibility Configuration

You can dynamically update visibility rules for a class *after* initial definition using `configure_visibility`. **Note:** This modifies the class directly and is generally not thread-safe. It also clears the internal cache for response models associated with that class.

```python
# Make 'email' visible to VIEWER role for the User class dynamically
User.configure_visibility(Role.VIEWER, {"id", "username", "email"})

# Overwrite all fields visible to ADMIN
User.configure_visibility(Role.ADMIN, {"id", "username", "email", "hashed_password", "is_active"})
```

## FastAPI Integration & Documentation

The recommended way to use this library with FastAPI is to use the `visible_fields_response` helper function within your endpoint logic before returning the data, as shown in the Basic Usage example.

**Challenge:** FastAPI generates OpenAPI documentation based on static type hints provided in the `response_model` parameter of route decorators. Since the actual response model type generated by `visible_fields_response` depends on the *runtime* role, using a static hint like `response_model=User` will show the full model in docs, while `response_model=Any` provides no schema detail.

**Strategies for Documentation:**

1.  **Accept Less Specific Docs:** Use `response_model=Any` or `response_model=YourBaseModel` in the decorator and rely on the `summary`/`description` fields or manually added OpenAPI `responses` information to clarify the different role-based outputs. Runtime filtering still works correctly.
2.  **Manually Define Response Models:** For endpoints where precise documentation is critical, manually define the specific response model variations (e.g., `UserAdminResponse`, `UserUserResponse`) in your code. Use these static models in the `response_model` decorator. You can still use `visible_fields_response` at runtime for consistent filtering logic, or map data manually. This duplicates effort but gives accurate static docs.

**Module-Level Creation Limitation:**

Due to limitations in Python's import system and Pydantic's Forward Reference resolution, **calling `YourModel.create_response_model(role)` directly at the module level is strongly discouraged and likely to fail** if your models involve complex forward references or cyclic dependencies (e.g., `Optional["MyUnion"]` where `MyUnion` is defined later or contains forward references). Please use the runtime `visible_fields_response` approach instead.

## API Reference

### Core Classes

-   `VisibleFieldsModel`: Inherit from this instead of `pydantic.BaseModel` to use the `field` decorator for visibility.
-   `VisibleFieldsMixin`: Mixin to add visibility features to existing `pydantic.BaseModel` classes (use with class-level `_role_visible_fields`).
-   `PaginatedResponse`: Generic model for paginated API responses (see `pydantic_visible_fields.paginatedresponse`).

### Functions

-   `field(*, visible_to: Optional[List[Any]] = None, **kwargs) -> Any`: Pydantic field replacement enabling `visible_to`.
-   `configure_roles(*, role_enum: Type[Enum], inheritance: Optional[Dict[Any, Any]] = None, default_role: Optional[Union[Enum, str]] = None) -> None`: Configures the global role system. Must be called once at startup.
-   `visible_fields_response(model: Any, role: Any = None) -> Any`: Runtime helper to convert a model instance (or list/dict of instances) to its role-specific filtered response.

### Methods (on `VisibleFieldsMixin` / `VisibleFieldsModel` instances/classes)

-   `model_instance.visible_dict(role=None) -> Dict[str, Any]`: Returns a dictionary containing only fields visible to the role (internal cycle marker `{'__cycle_reference__': True}` used).
-   `model_instance.to_response_model(role=None) -> BaseModel`: Returns a validated Pydantic model instance containing only fields visible to the role (cycles replaced with `None`). **Use `visible_fields_response` helper function for most use cases.**
-   `ModelCls.create_response_model(role: str, model_name_suffix: str = "Response") -> Type[BaseModel]`: Dynamically creates the Pydantic model *type* for a given role. **Avoid calling at module level for complex models; prefer using `visible_fields_response` at runtime.**
-   `ModelCls.configure_visibility(role: Union[Enum, str], visible_fields: Set[str]) -> None`: Dynamically updates visibility rules for a class (not thread-safe).

## Development

### Increasing version numbers

Use `bump2version` to increase the version number (included in the dev dependencies). E.g., `bump2version patch` or `bump2version minor`.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
