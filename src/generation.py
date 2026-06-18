from typing import List, Dict, Any
from llm_sdk.llm_sdk import Small_LLM_Model
from tokenization import Tokenization
from numpy import argmax
from validate import Functiondef


def build_selection_prompt(prompt: str, functions: list[Functiondef]) -> str:
    fn_desc = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    return (
        f"Available functions:\n{fn_desc}\n\n"
        f"Question: {prompt}\n"
        f"Function to call: "
    )


def generate_response(model: Small_LLM_Model, prompt: str, possible_values: List[str]):
    so_far = ""
    ids = model.encode(prompt)
    ids = ids.tolist()[0]
    tokenize = Tokenization(model)
    for _ in range(100):
        valid = tokenize.get_valid_tokens(so_far, possible_values)
        logits = tokenize.apply_mask(valid, ids)
        tok = int(argmax(logits))
        if so_far in possible_values:
            break
        so_far += tokenize.id_to_token()[tok]
        ids.append(tok)
    return so_far


def get_parameters(model: Small_LLM_Model, func: str, prompt: str) -> str:
    examp = [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        },
        {
            "prompt": "Reverse the string 'hello'",
            "name": "fn_reverse_string",
            "parameters": {"s": "hello"},
        },
        {
            "prompt": "Substitute the word cat with dog in this string 'the cat ate some meat'",
            "name": "fn_substitute_string_with_regex",
            "parameters": {
                "source_string": {"s": "the cat ate some meat"},
                "regex": {"a": "cat"},
                "replacement": {"b": "dog"},
            },
        },
        {
            "prompt": "Greet mark",
            "name": "fn_greet",
            "parameters": {
                "name": {"s": "Mark"},
            },
        },
    ]
    prompt = f"Give me the function parameters in json format of : {func}\
as parameter: value\n\
Examples:\n{examp}\n\
'prompt': {prompt},\n\
'parameters':"
    ids = model.encode(prompt).tolist()[0]
    so_far = ""
    tokenize = Tokenization(model)
    for _ in range(100):
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        # print(tokenize.id_to_token()[tok])
        token = tokenize.id_to_token()[tok]
        if "{" in token:
            so_far += token
            ids.append(tok)
            break
        ids.append(tok)
    for _ in range(100):
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        token = tokenize.id_to_token()[tok]
        # print(token)
        if token in ["!<", "!\n", "\n", "!", "Ċ", "ĊĊ"]:
            break
        if "}" in token:
            count = sum([1 for c in so_far if c == '{'])
            count_clo = sum([1 for c in so_far if c == '}'])
            if count - 1 == count_clo:
                so_far += token.split('}')[0] + '}'
                break
        so_far += token
        ids.append(tok)
    return so_far
