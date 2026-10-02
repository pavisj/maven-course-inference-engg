"""Build the tutor prompt: system prompt, then chunks in a fixed order, then the question.

    python -m app.prompt "why does prefix caching save prefill time?"

Prints the token count of each part with the Qwen tokenizer.

Layout:
- system message: SYSTEM_PROMPT (the same for every request: the shared prefix)
- user message: the retrieved chunks, sorted by file, page and position
  (not by score), then the question
"""

import sys

from app.tokenizer import count_tokens, get_tokenizer

SYSTEM_PROMPT = """\
You are Tutor, a study assistant for students of a course on inference \
engineering and systems design. The course covers how large language models \
are served: inference engines, KV caches, batching, scheduling, admission \
control, queueing, routing, gateways, observability and cluster design.

Each question comes with numbered excerpts from the course's revision notes. \
Each excerpt shows its chapter and page.

How you answer:
1. Answer from the excerpts. When you use a fact from an excerpt, cite it as \
[chapter, page], for example [07, p3].
2. If the excerpts do not contain the answer, say so in one sentence. Then \
give your best general answer and label it "not from the course notes".
3. Start with the direct answer in one or two sentences. Then explain the \
idea the answer depends on.
4. Use one small, concrete example with real numbers and units. Prefer an \
example the student can check by hand.
5. If there is a formula, name every symbol and its unit, and show one \
worked substitution.
6. End with one short question that lets the student test their own \
understanding. Do not answer it.

How you write:
- Short sentences, active voice, plain words.
- One word for one meaning. If you call something a "block", do not later \
call it a "page".
- Numbered lists for steps, bullets for parallel items.
- Never write bare numbers. Always give the unit and what the number is for.
- Keep answers under 250 words unless the student asks for more depth.

What you must not do:
- Do not invent citations, chapter numbers, page numbers or benchmark results.
- Do not quote long passages. Summarise, then cite.
- Do not write a full solution to a graded assignment. Offer a hint or a \
smaller related example instead.
- Do not answer questions outside the course topics. Say so in one sentence.

When the student asks for practice questions, write exactly the number they \
ask for. Base each question on the excerpts, and give the [chapter, page] \
it tests, but not the answer.\
"""


def order_chunks(chunks: list[dict]) -> list[dict]:
    """Fixed order: file, then page, then position on the page. Not by score."""
    return sorted(chunks, key=lambda c: (c["filename"], c["page"], int(c["id"].rsplit(":c", 1)[1])))


def format_chunk(n: int, chunk: dict) -> str:
    return f"[{n}] Chapter {chunk['chapter']}, page {chunk['page']} ({chunk['filename']})\n{chunk['text']}"


def chunks_block(chunks: list[dict]) -> str:
    return "Course note excerpts:\n\n" + "\n\n".join(
        format_chunk(n, c) for n, c in enumerate(order_chunks(chunks), start=1)
    )


def question_block(question: str) -> str:
    return f"Question: {question}"


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": chunks_block(chunks) + "\n\n" + question_block(question)},
    ]


def token_report(question: str, chunks: list[dict]) -> dict:
    """Token count of each part, plus the full prompt as the engine will see it."""
    messages = build_messages(question, chunks)
    rendered = get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    total = count_tokens(rendered)
    system = count_tokens(SYSTEM_PROMPT)
    per_chunk = [count_tokens(format_chunk(n, c)) for n, c in enumerate(order_chunks(chunks), start=1)]
    chunks_total = count_tokens(chunks_block(chunks))
    question_tokens = count_tokens(question_block(question))
    return {
        "system": system,
        "chunks": per_chunk,
        "chunks_total": chunks_total,
        "question": question_tokens,
        "template_and_separators": total - system - chunks_total - question_tokens,
        "total": total,
    }


if __name__ == "__main__":
    from app.retrieve import retrieve

    question = " ".join(sys.argv[1:]) or "Why does prefix caching save prefill time?"
    report = token_report(question, retrieve(question))
    print(f"system prompt:            {report['system']:5d} tokens")
    for n, t in enumerate(report["chunks"], start=1):
        print(f"  chunk {n}:                {t:5d} tokens")
    print(f"chunks block:             {report['chunks_total']:5d} tokens")
    print(f"question:                 {report['question']:5d} tokens")
    print(f"template and separators:  {report['template_and_separators']:5d} tokens")
    print(f"total prompt:             {report['total']:5d} tokens")
