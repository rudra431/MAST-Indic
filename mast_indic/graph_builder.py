"""Build an entity-relationship graph over an already-chunked corpus.

Reads `index_store/meta.jsonl` (written by `python -m mast_indic.index
build`) and asks the chat LLM to extract `(subject, relation, object)`
triples from each chunk's text, via a forced structured tool call so output
is always valid JSON rather than parsed free text. Writes one line per
extracted triple to `index_store/relations.jsonl`, tagged with the
(docid, chunk_id) it came from for provenance. `entity_graph.py` loads that
file into an in-memory adjacency list for
`CorpusInteractionEngine.graph_search`.

This is LLM-based extraction, not a trained NER/relation-extraction model --
consistent with this project's "flat file, brute force, dev-scale" approach
elsewhere (see `index.py`). Expect it to be noisy and non-exhaustive: the
model may miss relations, phrase the same entity two different ways across
chunks (no canonicalization is attempted), or occasionally hallucinate one
despite the prompt. Treat `graph_search` results as a hint that points back
to a real chunk to verify, not as ground truth.

Running an LLM over every chunk in a large corpus is expensive and slow --
resumable and checkpointed the same way `index.py`'s `build_index` is
(appends to `relations.jsonl` as it goes, skips chunks already recorded
there unless `--fresh`), and supports `--limit` for a dev-scale subset.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from .config import config

RELATIONS_PATH_NAME = "relations.jsonl"

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


def _load_processed(relations_path: str) -> set[tuple[str, int]]:
    processed: set[tuple[str, int]] = set()
    if not os.path.exists(relations_path):
        return processed
    with open(relations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            processed.add((row["docid"], row["chunk_id"]))
    return processed


def _extract_one(client: OpenAI, model: str, chunk: dict) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": chunk["text"]},
            ],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_relations"}},
            temperature=0.0,
        )
        message = response.choices[0].message
        triples = []
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments or "{}")
            triples = args.get("triples", []) or []
    except Exception as exc:  # noqa: BLE001 -- one bad chunk must not sink the batch
        tqdm.write(f"[error] {chunk['docid']}#{chunk['chunk_id']} failed: {exc!r}")
        triples = []

    rows = []
    for t in triples:
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
) -> str:
    meta_path = os.path.join(config.index_dir, "meta.jsonl")
    relations_path = os.path.join(config.index_dir, RELATIONS_PATH_NAME)
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

    processed = _load_processed(relations_path) if resume else set()
    if processed:
        print(f"Resuming: {len(processed)} chunk(s) already processed.")
    pending = [c for c in chunks if (c["docid"], c["chunk_id"]) not in processed]
    if not pending:
        print("Nothing to do -- all chunks already processed (pass --fresh to rebuild).")
        return relations_path

    client = OpenAI(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
        timeout=config.request_timeout,
        max_retries=config.request_max_retries,
    )
    extract_model = model or config.graph_model

    num_relations = 0
    with open(relations_path, "a" if processed else "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_extract_one, client, extract_model, c) for c in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="extracting relations", unit="chunk"):
                for row in future.result():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    num_relations += 1
                f.flush()

    print(f"Extracted {num_relations} relation(s) from {len(pending)} chunk(s) -> {relations_path}")
    return relations_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--limit", type=int, default=None, help="only process first N chunks (dev/testing)")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent LLM extraction calls")
    parser.add_argument("--model", default=None, help="override MAST_GRAPH_MODEL for this run")
    parser.add_argument("--fresh", action="store_true",
                         help="ignore any existing relations.jsonl and rebuild from scratch")
    args = parser.parse_args()

    if args.action == "build":
        build_entity_graph(
            limit=args.limit,
            resume=not args.fresh,
            concurrency=args.concurrency,
            model=args.model,
        )
