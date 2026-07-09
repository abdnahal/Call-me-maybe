from typing import List, Dict, Any
from llm_sdk.llm_sdk import Small_LLM_Model
from .tokenization import Tokenization
from numpy import argmax
from .validate import Functiondef


def build_selection_prompt(prompt: str, functions: list[Functiondef]) -> str:
    fn_desc = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    fn_desc = fn_desc + "\n- fn_unkown: unkown function"
    return (
        f"choose the function to call from the available functions.\n"
        f"Available functions:\n{fn_desc}\n\n"
        f"Prompt: {prompt}\n"
        f"Function to call to answer the prompt: "
    )


def generate_response(
    model: Small_LLM_Model,
    prompt: str,
    possible_values: List[str],
    tokenize: Tokenization,
) -> str:
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


def get_parameters(model: Small_LLM_Model, func: Dict[str, Any],
                   prompt: str, tokenize: Tokenization) -> str:

    prompt = f"You are generating JSON parameters for a function call.\n\
Use exactly the parameter names from the function definition.\n\
Use literal values, not type descriptions.\n\
Rules:\n\
- If a parameter type is 'integer', output an integer literal like 7.\n\
- If a parameter type is 'number', output a numeric literal like 3 or 3.14.\n\
- If a parameter type is 'string', output a JSON string value.\n\
- If a parameter type is 'boolean', output true or false.\n\
- Never output 'type': '...' objects.\n\
- Never nest the parameter schema inside the output.\n\
Function name: {func['name']}\n\
Function definition:\n{func['parameters']}\n\
User prompt: {prompt}\n\
Output:"

#     prompt = f"Generate the function parameters.\n\
# Valid JSON format expected!\n\
# Examples: {examp[0]}\n{examp[1]}\n\
# The parameters required: {func['parameters']}\n\
# Generate actual values matching the types and names!\n\
# 'prompt': {prompt},\n\
# 'name': {func['name']}\n\
# 'parameters':"
    ids = model.encode(prompt).tolist()[0]
    so_far: str = ""
    token = ""
    while "{" not in token:
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        token = tokenize.id_token[tok]
        ids.append(tok)
    token = '{"'
    ids.append(tokenize.token_id[token])
    so_far += token
    for _ in range(100):
        logits = model.get_logits_from_input_ids(ids)
        tok = argmax(logits)
        token = tokenize.id_token[tok]
        if "'" in token and so_far.strip("Ġ") == '{"':
            token = token.replace("'", '"')
            tok = tokenize.token_id[token]
        if "}" in token:
            count = sum([1 for c in so_far if c == "{"])
            count_clo = sum([1 for c in so_far if c == "}"])
            clo_tok = sum([1 for c in token if c == "}"])
            if count <= count_clo + clo_tok:
                so_far += token.split("}")[0]
                so_far += "}" * (count - count_clo)
                return so_far
        so_far += token
        ids.append(tok)
    return so_far
