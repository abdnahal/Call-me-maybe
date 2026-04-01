import json
import sys
from validate import Functiondef
from typing import Dict, List, Union
from pydantic import ValidationError


def functiondefs(filename: str) -> Union[list, Dict]:
    try:
        with open(filename, 'r') as f:
            defs = json.load(f)
            funcs = [Functiondef(**item) for item in defs]
            funcs.clear()
            return defs
    except json.JSONDecodeError as e:
        print(f'error loading json file {filename}: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print(f"File '{filename}' Not found")
        sys.exit(1)
    except ValidationError as e:
        print(e)


def prompts(filename: str) -> Union[list, Dict]:
    try:
        with open(filename, 'r') as f:
            prompts = json.load(f)
            for prompt in prompts:
                if not isinstance(prompt['prompt'], str):
                    raise TypeError("Prompts should be a string")
            return prompts
    except json.JSONDecodeError as e:
        print(f'error loading json file {filename}: {e}')
        sys.exit(1)
    except TypeError as e:
        print(f"Error Parsing '{filename}': {e}")
