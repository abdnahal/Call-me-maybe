from typing import List
from .llm_sdk.llm_sdk import Small_LLM_Model
from .tokenization import Tokenization
from numpy import argmax
from .validate import Functiondef


def build_selection_prompt(prompt: str, functions: list[Functiondef]) -> str:
    fn_desc = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    return (
        f"Available functions:\n{fn_desc}\n\n"
        f"Question: {prompt}\n"
        f"Function to call: "
    )


def generate_response(model: Small_LLM_Model, prompt: str,
                      possible_values: List[str], tokenize: Tokenization):
    so_far = ""
    ids = model.encode(prompt)
    ids = ids.tolist()[0]
    for _ in range(100):
        if so_far in possible_values:
            break
        valid = tokenize.get_valid_tokens(so_far, possible_values)
        if not valid:
            break
        logits = tokenize.apply_mask(valid, ids)
        tok = int(argmax(logits))
        so_far += tokenize.id_token[valid[tok]]
        ids.append(tok)
    return so_far


def get_parameters(model: Small_LLM_Model, func: str, prompt: str,
                   tokenize: Tokenization) -> str:
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
            "prompt": "Replace all vowels in 'Programming is fun' \
with asterisks",
            "name": "fn_substitute_string_with_regex",
            "parameters": {
                "source_string": "Programming is fun",
                "regex": "([aeiouAEIOU])",
                "replacement": "*",
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
    prompt = f"Give me the function parameters of : {func}\n\
Examples:\n{examp}\n\
I need valid JSON format with double quotes instead of single quotes!\n\
'prompt': {prompt},\n\
'parameters':"
    ids = model.encode(prompt).tolist()[0]
    so_far = ""
    token = ""
    while '{' not in token:
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        token = tokenize.id_token[tok]
        ids.append(tok)
    if "'" in token:
        token = token.replace("'", '"')
        ids[-1] = tokenize.token_id[token]
    so_far += token
    for _ in range(100):
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        token = tokenize.id_token[tok]
        if token in ["!<", "!\n", "\n", "!", "Ċ", "ĊĊ"]:
            break
        if "'" in token:
            count = sum([1 for c in so_far if c == '{'])
            if count == 1:
                token = token.replace("'", '"')
                tok = tokenize.token_id[token]
        if "}" in token:
            count = sum([1 for c in so_far if c == '{'])
            count_clo = sum([1 for c in so_far if c == '}'])
            clo_tok = sum([1 for c in token if c == '}'])
            if count <= count_clo + clo_tok:
                so_far += token.split('}')[0]
                so_far += '}' * (count - count_clo)
                return so_far
        so_far += token
        ids.append(tok)
    return so_far
