from src.__main__ import argparser
from parser import functiondefs, prompts
from tokenization import id_to_token, token_to_id, get_vocab
from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    print("Hello from call-me-maybe!\n")
    model = Small_LLM_Model()
    ids = model.encode("Say hi !")
    vocab = get_vocab(model)
    print(ids.tolist())
    logits = model.get_logits_from_input_ids(ids.tolist()[0])
    valid = [vocab[i] for i, logit in enumerate(logits) if logit > 0]
    print(valid)


if __name__ == "__main__":
    main()
