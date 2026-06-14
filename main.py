from src.__main__ import argparser
from parser import functiondefs, prompts
from tokenization import Tokenization
from llm_sdk.llm_sdk import Small_LLM_Model
from numpy import argmax


def main():
    print("Hello from call-me-maybe!\n")
    functions = {"fn_add_numbers": "Add two numbers together and return their sum.",
                 "fn_greet": "Generate a greeting message for a person by name.",
                 "fn_reverse_string": "Reverse a string and return the reversed result.",
                 "fn_substitute_string_with_regex": "Replace all occurrences matching a regex pattern in a string.",
                 "fn_get_square_root": "Calculate the square root of a number."}
    model = Small_LLM_Model()
    tokenize = Tokenization(model)
    prompt = f"to calculate square root of a number from {functions} I choose"
    ids = model.encode(prompt)
    ids = ids.tolist()[0]
    so_far = ""
    for _ in range(20):
        valid = tokenize.get_valid_tokens(so_far)
        print(so_far)
        print([tokenize.id_to_token()[i] for i in valid])
        logits = tokenize.apply_mask(valid, ids)
        tok = int(argmax(logits))
        if so_far in functions.keys():
            break
        so_far += tokenize.id_to_token()[tok]
        ids.append(tok)

if __name__ == "__main__":
    main()
