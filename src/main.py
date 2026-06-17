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
    print(final)


if __name__ == "__main__":
    main()

[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {"a": 265, "b": 345},
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": {"s": "Shrek"}},
    },
    {"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": {"s": "John"}}},
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {"s": "hello"},
    },
    {
        "prompt": "Reverse the string 'world'",
        "name": "fn_reverse_string",
        "parameters": {"s": "world"},
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": {"a": 16.0},
    },
    {
        "prompt": "Calculate the square root of 144",
        "name": "fn_get_square_root",
        "parameters": {"a": 144},
    },
    {
        "prompt": "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS",
        "name": "fn_substitute_string_with_regex",
        "parameters": {"numbers": ["34", "233"]},
    },
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "([aeiouAEIOU])",
            "replacement": "*",
        },
    },
    {
        "prompt": "Substitute the word cat with dog in 'The cat sat on the mat with another cat'",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog",
        },
    },
]


[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {"a": 265, "b": 345},
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": "fn_greetas", "value": "shrek"},
    },
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {"name": "fn_greetas", "value": "john"},
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {"s": "hello"},
    },
    {
        "prompt": "Reverse the string 'world'",
        "name": "fn_reverse_string",
        "parameters": {"s": "world"},
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": {"a": 16.0},
    },
    {
        "prompt": "Calculate the square root of 144",
        "name": "fn_get_square_root",
        "parameters": {"a": 144.0},
    },
    {
        "prompt": "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS",
        "name": "fn_substitute_string_with_regex",
        "parameters": {"a": 34, "b": 233},
    },
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {"a": 10, "b": 20},
    },
    {
        "prompt": "Substitute the word cat with dog in 'The cat sat on the mat with another cat'",
        "name": "fn_substitute_string_with_regex",
        "parameters": {"s": "dog"},
    },
]


[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {"s": {"type": "string"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_get_square_root",
        "description": "Calculate the square root of a number.",
        "parameters": {"a": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_substitute_string_with_regex",
        "description": "Replace all occurrences matching a regex pattern in a string.",
        "parameters": {
            "source_string": {"type": "string"},
            "regex": {"type": "string"},
            "replacement": {"type": "string"},
        },
        "returns": {"type": "string"},
    },
]
