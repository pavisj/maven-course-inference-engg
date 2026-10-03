"""Guardrails: the first no. A request that fails here never reaches a GPU.

inspect(payload, tenant) -> Guard(ok, code, reason, prompt_tokens, max_tokens)

Checks, in order; the first failure wins:
1. bad_body              400  messages missing or malformed, or max_tokens not a positive integer
2. unknown_tenant        400  X-Tenant is not interactive or batch
3. prompt_too_long       400  prompt_tokens + max_tokens > MAX_MODEL_LEN
4. max_tokens_too_large  400  max_tokens above the tenant's budget

prompt_tokens is counted with the Qwen chat template, as the engine counts it.
A request without max_tokens gets the tenant's budget, so vLLM never falls back
to "the rest of the context".
"""

from dataclasses import dataclass

from app.tokenizer import get_tokenizer

MAX_MODEL_LEN = 8192                              # --max-model-len (cluster/modal_vllm.py; Part 1)
MAX_TOKENS = {"interactive": 512, "batch": 256}   # answer budget per tenant (decision 6, 2026-10-03)
TENANTS = tuple(MAX_TOKENS)
ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Guard:
    ok: bool
    code: int
    reason: str
    prompt_tokens: int = 0
    max_tokens: int = 0


def count_prompt_tokens(messages: list[dict]) -> int:
    """Tokens the engine will see: chat template applied, generation prompt added."""
    tok = get_tokenizer()
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok.encode(rendered, add_special_tokens=False))


def _body_ok(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            return False
    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
        return False
    return True


def inspect(payload, tenant: str | None) -> Guard:
    if not _body_ok(payload):
        return Guard(False, 400, "bad_body")
    if tenant not in TENANTS:
        return Guard(False, 400, "unknown_tenant")
    max_tokens = payload.get("max_tokens") or MAX_TOKENS[tenant]
    prompt_tokens = count_prompt_tokens(payload["messages"])
    if prompt_tokens + max_tokens > MAX_MODEL_LEN:
        return Guard(False, 400, "prompt_too_long", prompt_tokens, max_tokens)
    if max_tokens > MAX_TOKENS[tenant]:
        return Guard(False, 400, "max_tokens_too_large", prompt_tokens, max_tokens)
    return Guard(True, 200, "ok", prompt_tokens, max_tokens)
