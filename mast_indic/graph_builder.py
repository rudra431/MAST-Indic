"""Build an entity-relationship graph over an already-chunked corpus.

SUPERSEDED: `entity_graph.py` now loads a richer, externally-produced
schema (typed entities, concepts, and head/relation/tail triplets, keyed by
a compound `"{docid}__chunk_NNNN"` id) from `index_store/entity_graph.jsonl`
-- see that module's docstring. This script's own output
(`index_store/relations.jsonl`, plain `(subject, relation, object)` triples)
is no longer what `CorpusInteractionEngine.graph_search` reads. Kept here
as a standalone, self-contained extraction pipeline in case you don't have
your own -- but its output needs its own loader if you want to use it (this
module doesn't write `entity_graph.jsonl`'s shape).

Reads `index_store/meta.jsonl` (written by `python -m mast_indic.index
build`) and asks the chat LLM to extract `(subject, relation, object)`
triples from each chunk's text, via a forced structured tool call so output
is always valid JSON rather than parsed free text. Writes one line per
extracted triple to `index_store/relations.jsonl`, tagged with the
(docid, chunk_id) it came from for provenance.

This is LLM-based extraction, not a trained NER/relation-extraction model --
consistent with this project's "flat file, brute force, dev-scale" approach
elsewhere (see `index.py`). Expect it to be noisy and non-exhaustive: the
model may miss relations, phrase the same entity two different ways across
chunks (no canonicalization is attempted), or occasionally hallucinate one
despite the prompt. Treat `graph_search` results as a hint that points back
to a real chunk to verify, not as ground truth.

Large chunks (index.py's `MAST_CHUNK_CHARS` default is 8192 characters --
~2048 tokens -- per chunk) are split into smaller, slightly-overlapping
windows before extraction -- a single long wall of text sent to the LLM
both risks slow generation / timeouts and tends to yield fewer,
lower-quality extracted relations than the same text split into focused
passages (see `--max-chunk-chars`).

Running an LLM over every chunk in a large corpus is expensive and slow --
resumable and checkpointed the same way `index.py`'s `build_index` is, and
supports `--limit` for a dev-scale subset. Completion is tracked per
*original* chunk (not per window) in a separate `relations_progress.jsonl`
sidecar file, written only once every window of a chunk has been processed
-- this also correctly resumes chunks that yielded zero relations, which
`relations.jsonl` alone can't distinguish from "not yet processed".
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from .config import config

RELATIONS_PATH_NAME = "relations.jsonl"
PROGRESS_PATH_NAME = "relations_progress.jsonl"
DEFAULT_MAX_CHUNK_CHARS = 1500

EXTRACT_SYSTEM_PROMPT = """Extract factual (subject, relation, object) triples about \
named entities (people, places, organizations, dates, events) from the passage below, \
by calling `extract_relations`. Only extract relations explicitly stated in the text \
-- never infer, guess, or use outside knowledge. Use concise entity names as they \
appear in the text and short verb-phrase relations (e.g. "founded", "located in", \
"alumnus of", "CEO of"). If the passage states no clear entity relationships, call \
`extract_relations` with an empty list.
"""

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_relations",
        "description": "Report the factual entity relationships found in the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "relation": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "relation", "object"],
                    },
                },
            },
            "required": ["triples"],
        },
    },
}


def _split_for_extraction(text: str, max_chars: int) -> list[str]:
    """Split an overly long chunk into smaller, slightly-overlapping windows.

    A word-level split (not a hard character cut) so windows stay readable;
    a small overlap keeps a relation whose subject/object straddle a split
    point from being cut in half. No-ops (returns `[text]`) for chunks
    already under `max_chars` -- at this project's default
    `MAST_CHUNK_CHARS=8192`, most chunks now exceed `--max-chunk-chars`
    (default 1500) and DO get split; this used to be the rare case back
    when chunks were sized in ~220 words (~1200-1400 chars).
    """
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    if not words:
        return []
    avg_chars_per_word = max(len(text) / len(words), 1.0)
    window_words = max(int(max_chars / avg_chars_per_word), 20)
    overlap_words = window_words // 8
    step = max(window_words - overlap_words, 1)

    windows = []
    for start in range(0, len(words), step):
        window = " ".join(words[start:start + window_words])
        if window:
            windows.append(window)
        if start + window_words >= len(words):
            break
    return windows


def _load_progress(progress_path: str) -> set[tuple[str, int]]:
    processed: set[tuple[str, int]] = set()
    if not os.path.exists(progress_path):
        return processed
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            processed.add((row["docid"], row["chunk_id"]))
    return processed


def _extract_from_text(client: OpenAI, model: str, text: str, chunk_label: str) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_relations"}},
            temperature=0.0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        args = json.loads(message.tool_calls[0].function.arguments or "{}")
        return args.get("triples", []) or []
    except Exception as exc:  # noqa: BLE001 -- one bad window must not sink the batch
        tqdm.write(f"[error] {chunk_label} failed: {exc!r}")
        return []


def _process_chunk(client: OpenAI, model: str, chunk: dict, max_chunk_chars: int) -> list[dict]:
    """Extract relations from one chunk, splitting it into windows first if it's large.

    Windows of the same chunk are processed sequentially (not fanned out to
    more threads) so this stays one unit of work for the outer thread pool
    and for progress tracking -- a chunk is only marked done once every one
    of its windows has been attempted.
    """
    label = f"{chunk['docid']}#{chunk['chunk_id']}"
    windows = _split_for_extraction(chunk["text"], max_chunk_chars)

    raw_triples = []
    for i, window in enumerate(windows):
        window_label = label if len(windows) == 1 else f"{label} (window {i + 1}/{len(windows)})"
        raw_triples.extend(_extract_from_text(client, model, window, window_label))

    rows = []
    for t in raw_triples:
        subject, relation, obj = t.get("subject"), t.get("relation"), t.get("object")
        if not (subject and relation and obj):
            continue
        rows.append({
            "subject": str(subject).strip(),
            "relation": str(relation).strip(),
            "object": str(obj).strip(),
            "docid": chunk["docid"],
            "chunk_id": chunk["chunk_id"],
        })
    return rows


def build_entity_graph(
    limit: int | None = None,
    resume: bool = True,
    concurrency: int = 8,
    model: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    meta_path = os.path.join(config.index_dir, "meta.jsonl")
    relations_path = os.path.join(config.index_dir, RELATIONS_PATH_NAME)
    progress_path = os.path.join(config.index_dir, PROGRESS_PATH_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No {meta_path} -- run `python -m mast_indic.index build` first."
        )

    chunks = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if limit is not None:
        chunks = chunks[:limit]

    processed = _load_progress(progress_path) if resume else set()
    if processed:
        print(f"Resuming: {len(processed)} chunk(s) already processed.")
    pending = [c for c in chunks if (c["docid"], c["chunk_id"]) not in processed]
    if not pending:
        print("Nothing to do -- all chunks already processed (pass --fresh to rebuild).")
        return relations_path

    long_chunks = sum(1 for c in pending if len(c["text"]) > max_chunk_chars)
    if long_chunks:
        print(f"{long_chunks} of {len(pending)} pending chunk(s) exceed --max-chunk-chars "
              f"({max_chunk_chars}) and will be split into windows before extraction.")

    client = OpenAI(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
        timeout=config.request_timeout,
        max_retries=config.request_max_retries,
    )
    extract_model = model or config.graph_model

    num_relations = 0
    with open(relations_path, "a" if processed else "w", encoding="utf-8") as rel_f, \
            open(progress_path, "a" if processed else "w", encoding="utf-8") as prog_f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_chunk = {
                pool.submit(_process_chunk, client, extract_model, c, max_chunk_chars): c
                for c in pending
            }
            for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk),
                                desc="extracting relations", unit="chunk"):
                chunk = future_to_chunk[future]
                for row in future.result():
                    rel_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    num_relations += 1
                prog_f.write(json.dumps(
                    {"docid": chunk["docid"], "chunk_id": chunk["chunk_id"]}, ensure_ascii=False
                ) + "\n")
                rel_f.flush()
                prog_f.flush()

    print(f"Extracted {num_relations} relation(s) from {len(pending)} chunk(s) -> {relations_path}")
    return relations_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--limit", type=int, default=None, help="only process first N chunks (dev/testing)")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent LLM extraction calls")
    parser.add_argument("--model", default=None, help="override MAST_GRAPH_MODEL for this run")
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS,
                         help="chunks longer than this are split into overlapping windows before extraction")
    parser.add_argument("--fresh", action="store_true",
                         help="ignore any existing relations.jsonl/relations_progress.jsonl and rebuild from scratch")
    args = parser.parse_args()

    if args.action == "build":
        build_entity_graph(
            limit=args.limit,
            resume=not args.fresh,
            concurrency=args.concurrency,
            model=args.model,
            max_chunk_chars=args.max_chunk_chars,
        )
