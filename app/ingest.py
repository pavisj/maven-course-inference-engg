"""Turn the corpus PDFs into retrieval chunks.

Run from the repo root:
    python -m app.ingest

Reads corpus/manifest.json. For every file with text_ok true, extracts the
text page by page with pypdf and cuts each page into chunks of about
CHUNK_TOKENS Qwen tokens. Writes one JSON object per chunk to app/chunks.jsonl.

Chunks never cross a page boundary and do not overlap. The last chunk of a
page is whatever is left, so it can be short.
"""

import json
from pathlib import Path

from pypdf import PdfReader

from app.tokenizer import get_tokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.json"
CHUNKS = ROOT / "app" / "chunks.jsonl"
CHUNK_TOKENS = 350


def page_chunks(text: str, tokens_per_chunk: int = CHUNK_TOKENS) -> list[tuple[str, int]]:
    """Cut one page into (chunk_text, token_count) windows.

    Cuts on the tokenizer's character offsets, so a chunk never splits a character.
    """
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    for start in range(0, len(offsets), tokens_per_chunk):
        window = offsets[start : start + tokens_per_chunk]
        chunk = text[window[0][0] : window[-1][1]].strip()
        if chunk:
            out.append((chunk, len(window)))
    return out


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}. Build the corpus first (Task 0).")
    manifest = json.loads(MANIFEST.read_text())

    rows = []
    for entry in manifest:
        if not entry["text_ok"]:
            print(f"skip {entry['filename']}: text_ok is false")
            continue
        reader = PdfReader(CORPUS / entry["filename"])
        n_before = len(rows)
        for page_no, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())  # collapse whitespace
            for i, (chunk, n_tokens) in enumerate(page_chunks(text)):
                rows.append(
                    {
                        "id": f"{Path(entry['filename']).stem}:p{page_no}:c{i}",
                        "chapter": entry["chapter"],
                        "filename": entry["filename"],
                        "title": entry["title"],
                        "page": page_no,
                        "tokens": n_tokens,
                        "text": chunk,
                    }
                )
        print(f"{entry['filename']}: {len(rows) - n_before} chunks")

    CHUNKS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    total_tokens = sum(r["tokens"] for r in rows)
    print(f"wrote {len(rows)} chunks, {total_tokens} tokens, to {CHUNKS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
