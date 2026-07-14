import json
import sys
from .validate import Functiondef
from typing import Dict, Tuple, List
from pydantic import ValidationError


def functiondefs(filename: str) -> Tuple[Functiondef, Dict]:
    try:
        with open(filename, 'r') as f:
            defs = json.load(f)
            funcs = [Functiondef(**item) for item in defs]
            # funcs.clear()
            return (funcs, defs)
    except json.JSONDecodeError as e:
        print(f'error loading json file {filename}: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print(f"File '{filename}' Not found")
        sys.exit(1)
    except ValidationError as e:
        print(e)
        sys.exit(1)
    return funcs


def prompts(filename: str) -> List[Dict[str, str]]:
    try:
        with open(filename, 'r') as f:
            prompts: List[Dict[str, str]] = json.load(f)
            for prompt in prompts:
                if not isinstance(prompt['prompt'], str):
                    raise TypeError("Prompts should be a string")
                if not prompt['prompt'].strip():
                    raise ValueError("Prompts cannot be empty!")
            return prompts
    except json.JSONDecodeError as e:
        print(f'error loading json file {filename}: {e}')
        sys.exit(1)
    except (TypeError, ValueError) as e:
        print(f"Error Parsing '{filename}': {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error opening {filename}: {e}")
        sys.exit(1)
