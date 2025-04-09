# Pydantic Visible Fields

A flexible field-level visibility control system for Pydantic models. This library allows you to define which fields of your models are visible to different user roles, making it easy to implement role-based access control at the data model level.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.0+-green.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`pydantic-visible-fields` provides a simple way to control which fields of your Pydantic models are visible to different user roles. It is particularly useful for API responses where you want to return different data depending on the user's role.

It also provides a `PaginatedResponse` class which makes it easy to generate paginated user responses with automatic conversion of objects to the correct visibility level.
### Key Features

- 🔒 **Field-level visibility control** using a simple decorator
- 🧩 **Class-level visibility control** for more complex scenarios
- 👥 **Role inheritance** to simplify permission management
- 🔄 **Nested model support** with full recursive visibility control
- 🌲 **Circular reference handling** to safely process complex object graphs
- 🚀 **Simple integration** with FastAPI and other web frameworks

## Installation

```bash
pip install git+https://github.com/Alge/pydantic-visible-fields.git
```

## Basic Usage

### 1. Define Your Roles

First, define your roles as an enum:

```python
from enum import Enum

class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
```

### 2. Configure Role System

Configure the role inheritance hierarchy:

```python
from pydantic_visible_fields import configure_roles

configure_roles(
    role_enum=Role,
    inheritance={
        Role.ADMIN: [Role.EDITOR],
        Role.EDITOR: [Role.VIEWER],
    },
    default_role=Role.VIEWER
)
```

### 3. Create Models with Visibility Rules

Use the `VisibleFieldsModel` base class and `field` decorator to control field visibility:

```python
from pydantic_visible_fields import VisibleFieldsModel, field

class User(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    username: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    email: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    hashed_password: str = field(visible_to=[Role.ADMIN])
    is_active: bool = field(visible_to=[Role.ADMIN])
```

### 4. Use in API Responses

In your API handlers, convert models to role-specific responses:

```python
from fastapi import Depends
from pydantic_visible_fields import visible_fields_response

@app.get("/users/{user_id}")
async def get_user(user_id: str, current_user = Depends(get_current_user)):
    user = await get_user_by_id(user_id)

    # Return different fields based on user's role
    role = get_user_role(current_user)
    return visible_fields_response(user, role=role)
```

## Advanced Usage

### Class-Level Visibility

For more complex scenarios, you can define visibility at the class level:

```python
from pydantic import BaseModel
from pydantic_visible_fields import VisibleFieldsMixin
from typing import ClassVar, Dict, Set

class UserSettings(BaseModel, VisibleFieldsMixin):
    _role_visible_fields: ClassVar[Dict[str, Set[str]]] = {
        Role.VIEWER: {"id", "theme"},
        Role.EDITOR: {"notifications"},
        Role.ADMIN: {"advanced_options", "debug_mode"},
    }

    id: str
    theme: str
    notifications: bool
    advanced_options: dict
    debug_mode: bool
```

### Nested Models

Visibility control works recursively with nested models:

```python
class Address(VisibleFieldsModel):
    street: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    city: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    country: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    postal_code: str = field(visible_to=[Role.EDITOR, Role.ADMIN])

class FullUser(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    username: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    email: str = field(visible_to=[Role.EDITOR, Role.ADMIN])
    address: Address = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
```

### Working with Collections

Visibility control works with lists and dictionaries of models:

```python
class Team(VisibleFieldsModel):
    id: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    name: str = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    members: List[User] = field(visible_to=[Role.VIEWER, Role.EDITOR, Role.ADMIN])
    settings: Dict[str, Setting] = field(visible_to=[Role.EDITOR, Role.ADMIN])
```

### Dynamic Visibility Configuration

You can dynamically configure visibility rules:

```python
# Add a field to the VIEWER role
User.configure_visibility(Role.VIEWER.value, {"id", "username", "email"})

# Change all visible fields for ADMIN
User.configure_visibility(Role.ADMIN.value, {"id", "username", "email", "hashed_password", "is_active", "last_login"})
```

## FastAPI Integration Example

Here's a complete example of how to use the library with FastAPI:

```python
from enum import Enum
from fastapi import FastAPI, Depends, HTTPException
from typing import List

from pydantic_visible_fields import (
    VisibleFieldsModel, visible_fields_response, field, configure_roles
)

# Define roles
class Role(str, Enum):
    PUBLIC = "public"
    USER = "user"
    ADMIN = "admin"

# Configure role system
configure_roles(
    role_enum=Role,
    inheritance={
        Role.ADMIN: [Role.USER],
        Role.USER: [Role.PUBLIC],
    },
    default_role=Role.PUBLIC
)

# Models with visibility rules
class Item(VisibleFieldsModel):
    id: int = field(visible_to=[Role.PUBLIC, Role.USER, Role.ADMIN])
    name: str = field(visible_to=[Role.PUBLIC, Role.USER, Role.ADMIN])
    description: str = field(visible_to=[Role.USER, Role.ADMIN])
    price: float = field(visible_to=[Role.USER, Role.ADMIN])
    cost: float = field(visible_to=[Role.ADMIN])
    supplier: str = field(visible_to=[Role.ADMIN])

# Sample database
items_db = [
    Item(id=1, name="Item 1", description="Description 1", price=10.99, cost=5.50, supplier="Supplier A"),
    Item(id=2, name="Item 2", description="Description 2", price=20.99, cost=12.75, supplier="Supplier B"),
]

# FastAPI app
app = FastAPI()

# Dependency to get user role (in a real app, this would use auth logic)
def get_user_role(role_name: str = "public"):
    if role_name == "admin":
        return Role.ADMIN
    elif role_name == "user":
        return Role.USER
    return Role.PUBLIC

@app.get("/items/", response_model=List)
async def get_items(role = Depends(get_user_role)):
    # Convert items to role-specific responses
    return [visible_fields_response(item, role=role) for item in items_db]

@app.get("/items/{item_id}")
async def get_item(item_id: int, role = Depends(get_user_role)):
    for item in items_db:
        if item.id == item_id:
            # Return different fields based on user's role
            return visible_fields_response(item, role=role)
    raise HTTPException(status_code=404, detail="Item not found")
```

## API Reference

### Core Classes

- `VisibleFieldsModel` - Base model class with field-level visibility control
- `VisibleFieldsMixin` - Mixin class that can be added to any Pydantic model
- `PaginatedResponse` - Class for handling pagination, automatically converts models to the correct visibility level

### Functions

- `field(visible_to=None, **kwargs)` - Field decorator to specify visibility
- `configure_roles(role_enum, inheritance=None, default_role=None)` - Configure the role system
- `visible_fields_response(model, role=None)` - Create a response with only visible fields

### Methods

- `model.visible_dict(role=None)` - Convert a model to a dictionary with only visible fields
- `model.to_response_model(role=None)` - Convert a model to a response model class
- `Model.configure_visibility(role, visible_fields)` - Configure visibility rules for a role

## Development

### Increasing version numbers
Use `bump2version` to increase the version number (included in the dev dependencies).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
