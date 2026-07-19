"""Module for parsing function definitions and test prompts from JSON files.

Provides utilities to load and validate function definitions and test
prompts from JSON files with error handling.
"""

import json
import sys
from .validate import Functiondef, Prompts
from typing import Dict, Tuple, List
from pydantic import ValidationError


def functiondefs(filename: str) -> Tuple[Functiondef, Dict]:
    """Load and parse function definitions from a JSON file.

    Args:
        filename: Path to the JSON file containing function definitions.

    Returns:
        Tuple[Functiondef, Dict]: A tuple containing:
            - List of validated Functiondef objects
            - List of raw function definition dictionaries

    Exits:
        Prints error and exits if file not found, JSON is invalid,
        or validation fails.
    """
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
    """Load and validate test prompts from a JSON file.

    Args:
        filename: Path to the JSON file containing test prompts.

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries, each containing
                             a 'prompt' key with a non-empty string value.

    Raises:
        SystemExit: Prints error and exits if file not found, JSON is
                   invalid, prompts are not strings, or prompts are empty.
    """
    try:
        with open(filename, 'r') as f:
            prompts: List[Dict[str, str]] = json.load(f)
            prmpts = [Prompts(**prompt) for prompt in prompts]
            return (prompts, prmpts)
    except json.JSONDecodeError as e:
        print(f'error loading json file {filename}: {e}')
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error opening {filename}: {e}")
        sys.exit(1)
    except ValidationError as e:
        print(e)
        sys.exit(1)
    return prompts
