from pydantic import BaseModel, Field
from typing import Dict


class FunctionParam(BaseModel):
    type: str = Field(min_length=3)


class Functiondef(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(min_length=3)
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam
