"""CLI: run the Interact-RAG-style agent (`interact_agent.py`) over a
language's queries, write submission JSONL.

Output: runs/{chat_model}/interact_{language}.jsonl -- same record shape as
`runner.py` (`{query_id, language, retriever, llm, tool_call_counts,
retrieved_docids, result}`), so it drops straight into `mast_indic/eval.py`
unchanged. `retriever` is tagged `interact-rag/{embed_model}` to distinguish
these runs from plain single-`search`-tool runs when comparing summaries.
"""
from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from .config import INDIC_LANGUAGES, config
from .index import SearchIndex
from .interact_agent import InteractAgent
from .queries import load_queries


def run_language(language: str, limit: int | None = None, save_transcripts: bool = False) -> str:
    queries = load_queries(language)
    if limit is not None:
        queries = queries[:limit]

    search_index = SearchIndex()
    agent = InteractAgent(search_index=search_index)

    out_dir = os.path.join(config.runs_dir, config.chat_model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"interact_{language}.jsonl")

    # Per-query checkpoint files: InteractAgent.answer() overwrites
    # checkpoint_path with the trajectory-so-far (scratchpad included)
    # after every turn, so a hard crash/kill mid-query -- not just the
    # exceptions caught below and inside answer() itself -- still leaves
    # the latest completed turn on disk. Removed once a query's real
    # record lands in out_path; left behind otherwise for postmortem.
    checkpoint_dir = os.path.join(out_dir, "checkpoints", language)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_failed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"answering [interact/{language}]"):
            checkpoint_path = os.path.join(checkpoint_dir, f"{q.qid}.json")
            try:
                # answer() itself catches mid-run failures (timeouts,
                # connection drops, 5xx) and returns whatever partial trace
                # it gathered rather than raising -- this except is now only
                # a last-resort safety net for something unexpected outside
                # that (e.g. a bug in our own code before any LLM call).
                agent_result = agent.answer(q.qid, q.query, language=q.language, checkpoint_path=checkpoint_path)
            except Exception as exc:  # noqa: BLE001 -- one bad query must not sink the batch
                num_failed += 1
                tqdm.write(f"[error] {q.qid} failed after retries: {exc!r}")
                record = {
                    "query_id": q.qid,
                    "language": q.language,
                    "retriever": f"interact-rag/{config.embed_model}",
                    "llm": config.chat_model,
                    "tool_call_counts": {},
                    "retrieved_docids": [],
                    "result": [],
                    "error": repr(exc),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            record = {
                "query_id": agent_result.query_id,
                "language": agent_result.language,
                "retriever": f"interact-rag/{config.embed_model}",
                "llm": config.chat_model,
                "tool_call_counts": agent_result.tool_call_counts,
                "retrieved_docids": agent_result.retrieved_docids,
                "result": agent_result.result,
            }
            if agent_result.error:
                record["error"] = agent_result.error
                num_failed += 1
                tqdm.write(f"[error] {q.qid} failed mid-run (partial trace saved): {agent_result.error}")
            if save_transcripts:
                record["transcript"] = agent_result.transcript
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass  # nothing was ever written (e.g. failed before the planner call) -- fine

    print(f"Wrote {len(queries)} records ({num_failed} failed) -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=INDIC_LANGUAGES + ["all"], default="hi")
    parser.add_argument("--limit", type=int, default=None, help="only run first N queries (dev/testing)")
    parser.add_argument("--save-transcripts", action="store_true")
    parser.add_argument("--debug", action="store_true",
                         help="print each turn's reasoning/tool-calls/answer to stderr")
    args = parser.parse_args()

    if args.debug:
        config.debug = True

    langs = INDIC_LANGUAGES if args.language == "all" else [args.language]
    for lang in langs:
        run_language(lang, limit=args.limit, save_transcripts=args.save_transcripts)
