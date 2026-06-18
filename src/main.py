from parser import functiondefs, prompts
from tokenization import Tokenization
from llm_sdk.llm_sdk import Small_LLM_Model
from generation import generate_response, build_selection_prompt, get_parameters
import json
import ast


def main():
    print("Hello from call-me-maybe!\n")
    functions = functiondefs("data/input/functions_definition.json")
    model = Small_LLM_Model()
    final = []
    proms = prompts("data/input/function_calling_tests.json")
    for prom in proms:
        res = {}
        funcs = [func.name for func in functions[0]]
        prompt = build_selection_prompt(prom["prompt"], functions[0])
        response = generate_response(model, prompt, funcs)
        res["prompt"] = prom["prompt"]
        res["name"] = response
        parameters = get_parameters(model, response, prom["prompt"])
        parameters = parameters.replace("Ġ", " ")
        res["parameters"] = ast.literal_eval(parameters)
        final.append(res)
    with open('output.json', 'x') as f:
        json.dump(final, f)


if __name__ == "__main__":
    main()
