"""Main module for the Call-me-maybe function calling system.

This module orchestrates the end-to-end process of:
1. Loading function definitions and test prompts
2. Using an LLM to select appropriate functions based on prompts
3. Generating parameters for the selected functions
4. Outputting results to a JSON file
"""

import argparse
from typing import Any
from pathlib import Path
from .parser import functiondefs, prompts
from .tokenization import Tokenization
from llm_sdk.llm_sdk import Small_LLM_Model
from .generation import generate_response, build_selection_prompt
from .generation import get_parameters
import json


def argparser() -> Any:
    """Parse command line arguments.

    Returns:
        Any: Parsed arguments containing paths to functions definition,
             input prompts, and output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default='data/input/functions_definition.json')
    parser.add_argument('--input',
                        default='data/input/function_calling_tests.json')
    parser.add_argument('--output', default='data/output/function_calls.json')
    args = parser.parse_args()
    return args


def main() -> None:
    """Execute the main function calling workflow.

    Loads function definitions and test prompts, iterates through each prompt
    to select appropriate functions and generate parameters, then writes
    results to an output JSON file.
    """
    print("Hello from call-me-maybe!\n")
    args = argparser()
    functions = functiondefs(args.functions_definition)
    model = Small_LLM_Model()
    final = []
    tokenizer = Tokenization(model)
    proms = prompts(args.input)
    funcs = [func.name for func in functions[0]]
    funcs.append('fn_none')
    for prom in proms:
        res = {}
        prompt = build_selection_prompt(prom["prompt"], functions[0])
        response = generate_response(model, prompt, funcs, tokenizer)
        res["prompt"] = prom["prompt"]
        res["name"] = response
        print(response)
        parameters = "{}"
        if response == "fn_none":
            res['parameters'] = {}
        else:
            func = [f for f in functions[1] if f['name'] == res['name']]
            parameters = get_parameters(model, func[0], prom["prompt"],
                                        tokenizer)
            parameters = parameters.replace("Ġ", " ").replace("Ċ", "\n")
            res["parameters"] = json.loads(parameters)
        print(parameters)
        final.append(res)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
