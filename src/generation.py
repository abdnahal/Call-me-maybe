"""Module for generating LLM prompts and responses.

Provides utilities for function selection and parameter generation.
"""

from typing import List, Dict, Any
from llm_sdk.llm_sdk import Small_LLM_Model
from .tokenization import Tokenization
from numpy import argmax
from .validate import Functiondef


def build_selection_prompt(prompt: str, functions: list[Functiondef]) -> str:
    """Build a prompt for selecting the appropriate function.

    Args:
        prompt: The user's input prompt/request.
        functions: List of available function definitions.

    Returns:
        str: A formatted prompt for the LLM to select the best function.
    """
    fn_desc = "\n".join(f"- {f.name}: {f.description}" for f in functions)
    fn_desc += "\n- fn_none: none of the available functions"
    return (
        f"You are a careful tool-selection assistant.\n"
        f"Select exactly one function that best matches the user's request.\n"
        f"Compare the request against every available function.\n"
        f"before answering.\n"
        f"Do not default to the most common or most general function.\n"
        f"If no function clearly matches, answer with `fn_none`.\n"
        f"Respond with only the function name, and nothing else.\n"
        f"Available functions:\n{fn_desc}\n\n"
        f"User request: {prompt}\n"
        f"Chosen function: "
    )


def generate_response(
    model: Small_LLM_Model,
    prompt: str,
    possible_values: List[str],
    tokenize: Tokenization,
) -> str:
    """Generate a constrained LLM response from possible values.

    Uses logit masking to constrain the model to only generate tokens
    that lead to one of the possible values.

    Args:
        model: The LLM model instance.
        prompt: The input prompt to the model.
        possible_values: List of valid output strings to choose from.
        tokenize: Tokenization utility for token/value mapping.

    Returns:
        str: The generated response constrained to one of possible_values.
    """
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
    """Generate JSON parameters for a function based on a user prompt.

    Uses the LLM to generate valid JSON parameters that match the
    function's parameter schema and satisfy the user's intent.

    Args:
        model: The LLM model instance.
        func: Function metadata including name and parameters.
        prompt: User request describing what parameters to generate.
        tokenize: Tokenization utility for token/value mapping.

    Returns:
        str: A JSON string containing the generated parameters.
    """

    prompt = f"You are extracting JSON parameters for a function call.\n\
Strictly use the parameter names and the format from the function \
definition.\n\
Use literal values, not type descriptions.\n\
Rules:\n\
- If a parameter type is 'integer', output an integer literal like 7.\n\
- If a parameter type is 'number', output a float like 3.0 or 3.14.\n\
- If a parameter type is 'string', output a JSON string value.\n\
- If a parameter type is 'boolean', output true or false.\n\
- Extract the parameters from the user's prompt; do not apply any changes to\n\
    them.\n\
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
        par = sum([1 for _ in func['parameters'].keys()])
        for i, para in enumerate(func['parameters'].keys()):
            if i+1 == par:
                if '"' + para + '":' in so_far:
                    if so_far.strip().endswith(':'):
                        token = ':'
                        while ',' not in token:
                            logits = model.get_logits_from_input_ids(ids)
                            tok = argmax(logits)
                            token = tokenize.id_token[tok]
                            ids.append(tok)
                            so_far += token
                            count = sum([1 for c in so_far if c == '{'])
                            count_clo = sum([1 for c in so_far if c == "}"])
                            if '}' in token and count == count_clo:
                                break
                        if so_far.strip('Ġ').endswith(','):
                            so_far = so_far.strip('Ġ').strip(',')
                            so_far += '}'
                        elif not so_far.strip('Ġ').endswith('}'):
                            so_far = so_far.split('}')[0] + '}'
                        return so_far
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
