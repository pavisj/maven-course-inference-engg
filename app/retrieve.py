"""BM25 retrieval over app/chunks.jsonl.

    python -m app.retrieve "why does prefix caching save prefill?"

Words are lowercase letters and digits. No stopwords, no stemming.
"""

import json
import re
import sys
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.ingest import CHUNKS

TOP_K = 5
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict], BM25Okapi]:
    if not CHUNKS.exists():
        raise SystemExit(f"missing {CHUNKS}. Run: python -m app.ingest")
    chunks = [json.loads(line) for line in CHUNKS.read_text().splitlines()]
    return chunks, BM25Okapi([words(c["text"]) for c in chunks])


def all_chunks() -> list[dict]:
    return _index()[0]


def retrieve(question: str, top_k: int = TOP_K, chapter: str | None = None) -> list[dict]:
    """Return the top_k chunks by BM25 score, best first. Each gets a "score" field.

    chapter: if set (e.g. "07"), only chunks with that chapter label are candidates.
    """
    chunks, bm25 = _index()
    scores = bm25.get_scores(words(question))
    candidates = [i for i in range(len(chunks)) if chapter is None or chunks[i]["chapter"] == chapter]
    best = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is admission control?"
    for c in retrieve(question):
        print(f"{c['score']:6.2f}  {c['id']:40}  {c['text'][:80]}")
