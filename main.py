from src.__main__ import argparser
from parser import functiondefs, prompts
from tokenization import id_to_token, token_to_id, get_vocab
from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    print("Hello from call-me-maybe!\n")
    model = Small_LLM_Model()
    


if __name__ == "__main__":
    main()
