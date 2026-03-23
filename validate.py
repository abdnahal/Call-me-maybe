from pydantic import Basemodel
from typing import Dict


class FunctionParam(Basemodel):
    type: str


class Functiondef(Basemodel):
    name: str
    description: str
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam
