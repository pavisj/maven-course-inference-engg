"""Laptop-side helpers shared by smoke.py and warm_probe.py.

Talks straight to the workers. There is no gateway yet, so this is an
engine smoke test only; app traffic must go through the serve path later.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
WORKERS_FILE = ROOT / "cluster" / "workers.json"
METRICS_DIR = ROOT / "metrics"
SERVED_MODEL_NAME = "tutor"

# About 400 tokens. smoke.py prints the real prompt_tokens from the engine.
SYSTEM_PROMPT = """\
You are Tutor, a patient study assistant for university students who are \
learning computer systems. Your job is to help a student understand an idea, \
not to hand over a finished answer they can copy.

How you teach:
1. Start by restating the student's question in one sentence, so they can \
check that you understood it.
2. Find the one idea the question depends on. Explain that idea first, in \
plain words, before any formula or code.
3. Use a small, concrete example with real numbers. Prefer an example the \
student can check by hand in under a minute.
4. If there is a formula, name every symbol, give its unit, and show one \
worked substitution.
5. End with one short question that lets the student test their own \
understanding. Do not answer that question for them.

How you write:
- Use short sentences and the active voice.
- Use one word for one meaning. If you call something a "block", do not \
later call it a "page" or a "chunk".
- Use a numbered list for steps and a bulleted list for parallel items.
- Never write bare numbers. Always say the unit and what the number is for.
- Keep answers under 250 words unless the student asks for more depth.

What you must not do:
- Do not invent citations, paper titles, or benchmark numbers. If you do \
not know a number, say so and explain how the student could measure it.
- Do not write a full solution to a graded assignment. Offer a hint, a \
smaller related example, or a check the student can run instead.
- Do not change the subject. If the question is outside computer systems, \
say so in one sentence and suggest where the student could ask instead.
- Do not pretend to run code. If an answer depends on running something, \
tell the student exactly what to run and what output to look for.

Topics you know well: operating systems, memory hierarchies, caches, \
virtual memory and paging, GPUs and their memory, batching, queues and \
latency, and how large language models are served in production.

When the student is confused, slow down. Ask which step lost them, then \
explain only that step again with a different example.\
"""

QUESTION = "In one paragraph: why does a KV cache make decoding faster?"


def load_workers() -> dict[str, str]:
    """Return {worker_name: base_url} from cluster/workers.json."""
    if not WORKERS_FILE.exists():
        raise SystemExit(f"missing {WORKERS_FILE}. Run: python cluster/write_workers.py")
    return json.loads(WORKERS_FILE.read_text())


def wait_healthy(url: str, timeout_s: float = 15 * 60) -> float:
    """Poll GET /health until 200. Return seconds waited.

    503 means "no container yet / still cold": wait and retry.
    502 and 504 and connection errors are also treated as "still starting".
    Any other status is a real failure.
    """
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=60)
            if r.status_code == 200:
                return time.perf_counter() - start
            if r.status_code not in (502, 503, 504):
                raise SystemExit(f"{url}/health returned HTTP {r.status_code}: {r.text[:200]}")
            print(f"  /health {r.status_code}: cold, waiting")
        except (httpx.TransportError, httpx.TimeoutException) as e:
            print(f"  /health {type(e).__name__}: waiting")
        time.sleep(2)
    raise SystemExit(f"{url}/health not 200 after {timeout_s:.0f} s")


def stream_chat(url: str, system_prompt: str, question: str, max_tokens: int = 128) -> dict:
    """Send one streamed chat completion. Return timings and token counts.

    ttft_s is measured on the laptop: request sent -> first content token.
    It includes the network round trip to Modal, not only engine time.
    """
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    ttft_s = None
    text = []
    usage = {}
    start = time.perf_counter()
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    text.append(piece)
    return {
        "ttft_s": ttft_s,
        "total_s": time.perf_counter() - start,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text": "".join(text),
    }


def scrape(url: str) -> str:
    """Return the raw Prometheus text from GET /metrics."""
    r = httpx.get(f"{url}/metrics", timeout=30)
    r.raise_for_status()
    return r.text


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample by metric name, across all label sets."""
    totals: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            totals[name] += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return dict(totals)


def find_metric(totals: dict[str, float], candidates: list[str], must_contain: list[str]) -> tuple[str | None, float | None]:
    """Find a metric whose name differs by vLLM version.

    Try the known names first. Otherwise take the first vllm: name that
    contains every word in must_contain. Returns (name, value).
    """
    for name in candidates:
        if name in totals:
            return name, totals[name]
    for name in sorted(totals):
        if name.startswith("vllm:") and all(w in name for w in must_contain) and not name.endswith("_created"):
            return name, totals[name]
    return None, None
