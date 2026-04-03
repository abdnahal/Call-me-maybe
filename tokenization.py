import json
from typing import List, Dict
from llm_sdk.llm_sdk import Small_LLM_Model


def id_to_token(vocab: Dict) -> Dict[int, str]:
    id_token = {int(k): v for v, k in vocab.items()}
    return id_token


def token_to_id(vocab: Dict) -> Dict[str, int]:
    token_id = {k: int(v) for k, v in vocab.items()}
    return token_id


def get_vocab(model: Small_LLM_Model):
    path = model.get_path_to_vocab_file()
    with open(path, 'r') as f:
        vocab = json.load(f)
        return vocab
