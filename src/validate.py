"""Module for validating function definitions using Pydantic models.

Defines data models for function parameters and function definitions
with validation rules.
"""

from pydantic import BaseModel, Field
from typing import Dict


class Prompts(BaseModel):
    prompt: str = Field(min_length=3)


class FunctionParam(BaseModel):
    """Represents a function parameter with type information.

    Attributes:
        type: The parameter type (must be at least 3 characters).
    """
    type: str = Field(min_length=3)


class Functiondef(BaseModel):
    """Represents a complete function definition.

    Attributes:
        name: Function name (must be at least 3 characters).
        description: Function description (min 3 characters).
        parameters: Dictionary mapping parameter names to
                   FunctionParam objects.
        returns: Return type of the function.
    """
    name: str = Field(min_length=3)
    description: str = Field(min_length=3)
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam
