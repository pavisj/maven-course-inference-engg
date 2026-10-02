"""The Qwen tokenizer, pinned to the same revision the workers serve.

Only the tokenizer files are downloaded (a few MB), not the weights.
"""

from functools import lru_cache

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # same as cluster/modal_vllm.py


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def count_tokens(text: str) -> int:
    """Tokens in plain text, without special tokens or chat template."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))
