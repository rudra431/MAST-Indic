# MAST Indic (Track 2) — agentic search skeleton

A minimal, working skeleton for MAST @ FIRE 2026 Track 2: an OpenAI-compatible
chat LLM drives a tool-calling search loop over a local, embedding-indexed
copy of the BrowseComp-Plus corpus, answering queries from the
`mast-benchmark/indic-queries-2026` dataset in 9 Indic languages.

```
mast_indic/
  config.py    # env-driven settings
  index.py     # build/query the local embedding index (OpenAI-compatible embeddings)
  queries.py   # loads indic-queries-2026 per language
  agent.py     # tool-calling agent loop (OpenAI-compatible client)
  runner.py    # CLI: runs a language's queries -> submission JSONL
scripts/
  build_index.sh
  run_track2.sh
```

## Setup

```bash
pip install -e .
cp .env.example .env   # then edit OPENAI_BASE_URL / OPENAI_API_KEY / MAST_CHAT_MODEL
```

Defaults to `embeddinggemma:300m` for embeddings, since that's what's already
pulled locally (`ollama list`) and served through Ollama's OpenAI-compatible
`/v1/embeddings` route. To use a different embedding model instead, `ollama
pull <model>` and set `MAST_EMBED_MODEL` in `.env`.

The chat model and the embedding model are intentionally decoupled and both
go through OpenAI-compatible clients: `OPENAI_BASE_URL` for chat and
`MAST_EMBED_BASE_URL` for embeddings, each pointable at OpenAI itself or any
OpenAI-compatible server (vLLM, Ollama's `/v1`, TEI, etc.). To use a
vLLM-hosted embedding model, set `MAST_EMBED_BASE_URL` to that server's
`/v1` and `MAST_EMBED_MODEL` to the model it's serving.

## 1. Build the retrieval index

```bash
./scripts/build_index.sh --limit 2000   # dev smoke test, a couple minutes
./scripts/build_index.sh                # full ~100k-doc corpus
```

This streams `Tevatron/browsecomp-plus-corpus`, splits each document into
~220-word overlapping chunks, embeds every chunk via the configured
embedding server, and writes:

- `index_store/embeddings.npy` — L2-normalized float32 chunk vectors
- `index_store/meta.jsonl` — one line per chunk: `{docid, chunk_id, url, text}`
- `index_store/manifest.json` — embed model + chunking config used

**Resumable by default:** if `./scripts/build_index.sh` is interrupted (crash,
network drop, killed job) and rerun with the same command, it skips chunks
already recorded in `meta.jsonl` (matched by `docid` + `chunk_id`) instead of
re-chunking and re-embedding everything, and checkpoints `embeddings.npy`
every 2000 newly-embedded chunks (`--checkpoint-every N` to change that) so
a crash only costs the last checkpoint, not the whole run. Pass `--fresh` to
ignore any existing index and rebuild from scratch (e.g. after changing
`MAST_CHUNK_WORDS`/`MAST_EMBED_MODEL`).

**Scaling note:** the full corpus is ~100k docs / ~5,200 words average, so
full-corpus indexing means embedding several hundred thousand chunks locally —
expect this to take a while and to use several GB of RAM for the brute-force
matrix. For iteration, use `--limit`. For a real submission at scale, either
run indexing against a GPU-backed embedding server (e.g. vLLM), or swap
`SearchIndex` for FAISS / the BrowseComp-Plus repo's pre-built BM25 or
Qwen3-Embedding indexes
(see `scripts_build_index/download_indexes.sh` in
[texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)).

## 2. Run the agent

```bash
./scripts/run_track2.sh --language hi --limit 5    # smoke test, 5 queries
./scripts/run_track2.sh --language hi               # full Hindi run
./scripts/run_track2.sh --language all              # all 9 languages
```

Writes `runs/{MAST_CHAT_MODEL}/{language}.jsonl`, one record per query:

```json
{
  "query_id": "hi-235",
  "language": "hi",
  "retriever": "embeddinggemma-300m",
  "llm": "gpt-4o-mini",
  "tool_call_counts": {"search": 3},
  "retrieved_docids": [["d123", "d456"], ["d789", "d123"]],
  "result": [
    {"type": "reasoning", "tool_name": null, "arguments": null, "output": "..."},
    {"type": "tool_call", "tool_name": "search", "arguments": "query: \"...\"", "output": "[{\"docid\": \"d123\", ...}]"},
    {"type": "output_text", "tool_name": null, "arguments": null, "output": "Explanation: ...\nExact Answer: Marie Curie"}
  ]
}
```

`retrieved_docids` is one list per search round (not deduped across rounds).
`result` is the step-by-step trace: a `reasoning` entry per turn where the
model returned reasoning content, a `tool_call` entry per `search` call, and
a final `output_text` entry with the model's `Explanation:` / `Exact Answer:`
response.

Add `--save-transcripts` to also dump the full message transcript per query
(useful for debugging/error analysis, strip before final submission).

Add `--debug` (or set `MAST_DEBUG=true` in `.env`) to print each turn's
reasoning, search queries, hit counts/docids, and final answer to stderr as
they happen -- useful for seeing where a run stalls or loops without waiting
for the whole JSONL file.

**Before submitting:** double-check this record shape against the live
example runs / evaluation code in the BrowseComp-Plus repo
(`search_agent/`, `scripts_evaluation/evaluate_run.py`) — this skeleton
mirrors the fields documented on the repo page, but the repo itself is the
source of truth, and the max-3-runs-per-language-per-team rule still applies.

## How the agent works (`mast_indic/agent.py`)

1. System prompt tells the LLM: query is in an Indic language, corpus and
   answer are English, use the `search` tool iteratively, then answer with
   an `Explanation:` / `Exact Answer:` pair.
2. Each turn, the LLM either calls `search(query: str)` (executed against the
   local index, returns top-`MAST_TOP_K` docs with score + snippet) or
   returns a final answer.
3. Each search round's docids are appended as their own list to
   `retrieved_docids`; tool calls are tallied into `tool_call_counts`.
4. On the final allowed turn, `tool_choice` is forced to `"none"` so the model
   must answer rather than call another tool.

No translation step is hard-coded — capable chat models handle the Indic
query directly and formulate English search queries themselves. If you find a
particular language/model combination struggles with this, consider adding an
explicit translate-then-search step in `agent.py`.

## What's stubbed vs. real

- **Real**: corpus loading, chunking, batch embedding, cosine search,
  the tool-calling loop, JSONL output.
- **Not included**: retrieval recall evaluation against qrels, and answer
  accuracy via LLM-judge — both live in the BrowseComp-Plus repo's
  `scripts_evaluation/`; run your generated `runs/` directory through
  `evaluate_run.py` there once you have real relevance judgments for the
  Indic queries.
