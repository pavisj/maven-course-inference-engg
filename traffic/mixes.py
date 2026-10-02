"""Class 7 traffic mixes, shaped as tutor traffic. Each mix is a generator of requests.

A request is a dict: {"mix", "tenant", "question", "chapter"}.
"chapter" is None (retrieve from the whole corpus) or a chapter label
(retrieve only from that chapter).

- unique:          every question different, interactive tenant, whole corpus
- shared_chapter:  different questions, all retrieved from one chapter
- mixed:           interactive questions alternating with a batch tenant that
                   asks for 3 practice questions per chapter, chapter by chapter
"""

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "manifest.json"

TEMPLATES = [
    "What is {t}?",
    "Why does {t} matter when serving an LLM?",
    "Explain {t} with a small worked example.",
    "What goes wrong in production if you ignore {t}?",
    "How would you measure {t} on a live cluster?",
    "What is a common misconception about {t}?",
    "How does {t} affect time to first token?",
    "Compare {t} with the simplest alternative.",
]

TOPICS = [
    "the KV cache", "PagedAttention", "prefix caching", "continuous batching", "chunked prefill",
    "the prefill phase", "the decode phase", "time to first token", "time per output token",
    "tail latency at p99", "Little's Law", "admission control", "load shedding", "rate limiting",
    "a 429 response", "a 503 response", "request queueing", "the engine's waiting queue",
    "preemption", "KV cache eviction", "max-num-seqs", "max-num-batched-tokens",
    "gpu-memory-utilization", "memory bandwidth in decode", "compute bound prefill",
    "speculative decoding", "tensor parallelism", "prefill and decode disaggregation",
    "KV cache transfer between workers", "Mooncake", "warm-up of a new replica", "cold starts",
    "CUDA graphs", "torch.compile in vLLM", "quantization of weights", "FP8 KV cache",
    "least-loaded routing", "power of two choices routing", "prefix-aware routing",
    "an overflow gate", "a gateway in front of the engine", "Prometheus metrics from vLLM",
    "Grafana dashboards for inference", "multi-tenant fairness", "a tenant token window",
    "the vLLM scheduler", "SGLang's radix cache", "TensorRT-LLM",
]

BATCH_PROMPT = "Write 3 practice questions about {title}, based on the course note excerpts."


def chapters() -> list[tuple[str, str]]:
    """(chapter label, title) for every text_ok file, in manifest order, one per chapter label."""
    seen = {}
    for entry in json.loads(MANIFEST.read_text()):
        if entry["text_ok"] and entry["chapter"] not in seen:
            seen[entry["chapter"]] = entry["title"]
    return list(seen.items())


def _questions(seed: int):
    """Every template x topic once, shuffled; then again with a numbered suffix, forever. Never repeats."""
    pairs = [(tmpl, topic) for tmpl in TEMPLATES for topic in TOPICS]
    rng = random.Random(seed)
    for round_no in itertools.count():
        rng.shuffle(pairs)
        for tmpl, topic in pairs:
            q = tmpl.format(t=topic)
            yield q if round_no == 0 else f"{q} (variant {round_no + 1})"


def unique(seed: int = 0):
    for q in _questions(seed):
        yield {"mix": "unique", "tenant": "interactive", "question": q, "chapter": None}


def shared_chapter(chapter: str = "07", seed: int = 0):
    for q in _questions(seed):
        yield {"mix": "shared_chapter", "tenant": "interactive", "question": q, "chapter": chapter}


def mixed(seed: int = 0):
    interactive = _questions(seed)
    batch = itertools.cycle(chapters())
    while True:
        yield {"mix": "mixed", "tenant": "interactive", "question": next(interactive), "chapter": None}
        label, title = next(batch)
        yield {"mix": "mixed", "tenant": "batch", "question": BATCH_PROMPT.format(title=title), "chapter": label}


MIXES = {"unique": unique, "shared_chapter": shared_chapter, "mixed": mixed}
