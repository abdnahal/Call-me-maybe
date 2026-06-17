from pydantic import BaseModel
from typing import Dict


class FunctionParam(BaseModel):
    type: str


class Functiondef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam
