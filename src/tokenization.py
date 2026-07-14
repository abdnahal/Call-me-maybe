"""Module for handling tokenization and token-to-ID mapping.

Provides utilities for converting between tokens and their IDs, and for
applying logit masks to constrain LLM output to valid tokens.
"""

import json
from typing import List, Dict, Any
from llm_sdk.llm_sdk import Small_LLM_Model


class Tokenization:
    """Manages tokenization and bidirectional token-ID mapping.

    Attributes:
        model: The LLM model instance.
        vocab: Dictionary mapping tokens to token IDs.
        id_token: Dictionary mapping token IDs to tokens.
        token_id: Dictionary mapping tokens to token IDs.
    """

    def __init__(self, model: Small_LLM_Model):
        """Initialize the Tokenization instance.

        Args:
            model: The LLM model instance to use for tokenization.
        """
        self.model = model
        self.vocab = self.get_vocab(model)
        self.id_token = self.id_to_token()
        self.token_id = self.token_to_id()

    def id_to_token(self) -> Dict[int, str]:
        """Create mapping from token IDs to tokens.

        Returns:
            Dict[int, str]: Dictionary mapping token IDs to token strings.
        """
        id_token = {int(k): v for v, k in self.vocab.items()}
        return id_token

    def token_to_id(self) -> Dict[str, int]:
        """Create mapping from tokens to token IDs.

        Returns:
            Dict[str, int]: Dictionary mapping tokens to token IDs.
        """
        token_id = {k: int(v) for k, v in self.vocab.items()}
        return token_id

    def get_vocab(self, model: Small_LLM_Model) -> Dict[str, Any]:
        """Load vocabulary from model's vocab file.

        Args:
            model: The LLM model instance.

        Returns:
            Dict[str, Any]: Dictionary mapping tokens to token IDs.
        """
        path = model.get_path_to_vocab_file()
        with open(path, 'r') as f:
            vocab: Dict[str, Any] = json.load(f)
            return vocab

    def get_valid_tokens(self, so_far: str, functions: List[str]) -> List[int]:
        """Find valid token IDs that could lead to target functions.

        Args:
            so_far: The string generated so far.
            functions: List of target function names to complete to.

        Returns:
            List[int]: List of token IDs that, when appended, keep us on
                      path to completing one of the target functions.
        """
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

    def apply_mask(self, valid: List[int], ids: List[int]) -> List[float]:
        """Apply logit mask to constrain output to valid tokens.

        Args:
            valid: List of valid token IDs to keep.
            ids: List of input token IDs.

        Returns:
            List[float]: Logit scores for the valid tokens only.
        """
        logits = self.model.get_logits_from_input_ids(ids)
        masked = [logits[tok] for tok in valid]
        return masked
