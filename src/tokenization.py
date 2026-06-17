import json
from typing import List, Dict
from llm_sdk.llm_sdk import Small_LLM_Model


class Tokenization:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self.vocab = self.get_vocab(model)
        self.id_token = self.id_to_token()
        self.token_id = self.token_to_id()

    def id_to_token(self) -> Dict[int, str]:
        id_token = {int(k): v for v, k in self.vocab.items()}
        return id_token

    def token_to_id(self) -> Dict[str, int]:
        token_id = {k: int(v) for k, v in self.vocab.items()}
        return token_id

    def get_vocab(self, model: Small_LLM_Model):
        path = model.get_path_to_vocab_file()
        with open(path, 'r') as f:
            vocab = json.load(f)
            return vocab

    def get_valid_tokens(self, so_far: List[int], functions: List[str]):
        valid = []
        for id, token in self.id_token.items():
            if not token:
                continue
            current = so_far + token
            for function in functions:
                if function.startswith(current):
                    if id in valid:
                        continue
                    valid.append(id)
        return valid

    def apply_mask(self, valid: List[int], ids: List[int]):
        logits = self.model.get_logits_from_input_ids(ids)
        masked = [logits[i] if i in valid else float("-inf")
                  for i, _ in enumerate(logits)]
        return masked
