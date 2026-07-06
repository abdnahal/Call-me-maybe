import argparse
from typing import Any
from .parser import functiondefs, prompts
from .tokenization import Tokenization
from .llm_sdk.llm_sdk import Small_LLM_Model
from .generation import generate_response, build_selection_prompt
from .generation import get_parameters
import json
# import ast


def argparser() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default='data/input/functions_definition.json')
    parser.add_argument('--input',
                        default='data/input/function_calling_tests.json')
    parser.add_argument('--output', default='data/output/function_calls.json')
    args = parser.parse_args()
    return args


def main():
    print("Hello from call-me-maybe!\n")
    args = argparser()
    functions = functiondefs(args.functions_definition)
    model = Small_LLM_Model()
    final = []
    tokenizer = Tokenization(model)
    proms = prompts(args.input)
    for prom in proms:
        res = {}
        funcs = [func.name for func in functions[0]]
        prompt = build_selection_prompt(prom["prompt"], functions[0])
        response = generate_response(model, prompt, funcs, tokenizer)
        res["prompt"] = prom["prompt"]
        res["name"] = response
        func = [f for f in functions[1] if f['name'] == res['name']]
        parameters = get_parameters(model, func[0], prom["prompt"], tokenizer)
        parameters = parameters.replace("Ġ", " ").replace("Ċ", "\n")
        print(parameters)
        res["parameters"] = json.loads(parameters)
        final.append(res)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
