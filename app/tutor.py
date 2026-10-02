"""Tutor CLI: retrieve, build the prompt, stream the answer from a worker.

    python -m app.tutor "why does prefix caching save prefill time?" --tenant interactive
    python -m app.tutor "..." --tenant batch --base-url https://...workera...modal.direct

--base-url defaults to WorkerA in cluster/workers.json.
Logs one JSON row per call to metrics/app_calls.jsonl: TTFT, prompt tokens, output tokens.

There is no gateway yet: this calls the worker directly.
The tenant is only a label in the log; it does not change the request.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from app.prompt import build_messages
from app.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
WORKERS_FILE = ROOT / "cluster" / "workers.json"
APP_CALLS = ROOT / "metrics" / "app_calls.jsonl"
SERVED_MODEL_NAME = "tutor"
MAX_TOKENS = 512
TENANTS = ("interactive", "batch")


def default_base_url() -> str:
    if not WORKERS_FILE.exists():
        raise SystemExit(f"no --base-url and no {WORKERS_FILE}. Run: python cluster/write_workers.py")
    return json.loads(WORKERS_FILE.read_text())["WorkerA"]


def call_tutor(question: str, tenant: str, base_url: str, max_tokens: int = MAX_TOKENS,
               on_token=None, timeout_s: float = 300, chapter: str | None = None) -> dict:
    """Ask one question. Return a record; never raise on HTTP or network errors.

    ttft_s is measured here: request sent -> first content token. It includes the network.
    prompt_tokens and output_tokens come from the engine's usage report.
    chapter: if set, retrieval only looks at that chapter's chunks.
    """
    chunks = retrieve(question, chapter=chapter)
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": build_messages(question, chunks),
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    record = {
        "ts": time.time(),
        "tenant": tenant,
        "question": question,
        "chapter_filter": chapter,
        "base_url": base_url,
        "chunk_ids": [c["id"] for c in chunks],
        "chapters": sorted({c["chapter"] for c in chunks}),
        "prompt_tokens": None,
        "output_tokens": None,
        "ttft_s": None,
        "total_s": None,
        "status": None,
        "error": None,
    }
    start = time.perf_counter()
    try:
        with httpx.stream("POST", f"{base_url.rstrip('/')}/v1/chat/completions",
                          json=payload, timeout=timeout_s) as r:
            record["status"] = r.status_code
            if r.status_code != 200:
                record["error"] = r.read().decode(errors="replace")[:300]
                return record
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                if chunk.get("usage"):
                    record["prompt_tokens"] = chunk["usage"].get("prompt_tokens")
                    record["output_tokens"] = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices", []):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        if record["ttft_s"] is None:
                            record["ttft_s"] = time.perf_counter() - start
                        if on_token:
                            on_token(piece)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        record["status"] = record["status"] or "error"
        record["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        record["total_s"] = time.perf_counter() - start
    return record


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--tenant", choices=TENANTS, default="interactive")
    parser.add_argument("--base-url", default=None, help="worker URL; default: WorkerA in cluster/workers.json")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--log", type=Path, default=APP_CALLS, help="JSONL file to append the call record to")
    args = parser.parse_args()

    record = call_tutor(args.question, args.tenant, args.base_url or default_base_url(), args.max_tokens,
                        on_token=lambda piece: print(piece, end="", flush=True))
    print()
    append_jsonl(args.log, record)
    ttft = f"{record['ttft_s'] * 1000:.0f} ms" if record["ttft_s"] is not None else "none"
    print(f"[{record['tenant']}] status {record['status']}, TTFT {ttft}, "
          f"prompt {record['prompt_tokens']} tokens, output {record['output_tokens']} tokens",
          file=sys.stderr)
    if record["error"]:
        print(f"error: {record['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
