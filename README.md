# MAST Indic (Track 2) — agentic search skeleton

A minimal, working skeleton for MAST @ FIRE 2026 Track 2: an OpenAI-compatible
chat LLM drives a tool-calling search loop over a local, embedding-indexed
copy of the BrowseComp-Plus corpus, answering queries from the
`mast-benchmark/indic-queries-2026` dataset in 9 Indic languages.

```
mast_indic/
  config.py          # env-driven settings
  index.py           # build/query the local embedding index (OpenAI-compatible embeddings)
  queries.py         # loads indic-queries-2026 per language
  agent.py           # tool-calling agent loop, one `search` tool (OpenAI-compatible client)
  runner.py          # CLI: runs agent.py over a language's queries -> submission JSONL
  interact_engine.py # Interact-RAG-style retrieval action primitives (arXiv:2510.27566)
  interact_agent.py  # agent loop using interact_engine.py instead of one `search` tool
  interact_runner.py # CLI: runs interact_agent.py over a language's queries -> submission JSONL
  eval.py            # CLI: LLM-judge scoring of a run against gold answers
scripts/
  build_index.sh
  run_track2.sh
  run_interact.sh
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

## 3. Evaluate

```bash
python -m mast_indic.eval \
  --predictions runs/gpt-4o-mini/hi.jsonl \
  --gold data/hi_gold.jsonl
```

`--predictions` accepts one or more run JSONL files, or a directory of them
(each file is scored independently). `--gold` expects one JSON object per
line with `task_id`, `question`, `gold_answer` fields.

There are no relevance judgments (qrels) for this track, so this only scores
answer correctness -- no retrieval recall, no citation precision/recall.

For each query, an LLM judge grades the model's final answer against
`gold_answer` and the script reports:

- **Accuracy** -- % of responses the judge marks `correct: yes`
- **Exact Match / F1** -- SQuAD-style string metrics comparing the model's
  `Exact Answer:` line (or the judge's extracted answer, if parsed) against
  `gold_answer`
- **Calibration Error** -- from the judge's extracted confidence scores
  (needs >=100 scored queries to be meaningful; reported as `N/A` below that)

The judge is any OpenAI-compatible chat endpoint. It defaults to this
project's `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `MAST_CHAT_MODEL` from
`.env`, but you'll usually want a separate/cheaper judge model -- override
with `--judge_base_url` / `--judge_api_key` / `--judge_model` (or the
`MAST_JUDGE_BASE_URL` / `MAST_JUDGE_API_KEY` / `MAST_JUDGE_MODEL` env vars),
e.g. to point at a self-hosted vLLM/Ollama server serving something like
`Qwen/Qwen3-32B`.

Judging is cached per `query_id` in `{run}_eval.jsonl` -- rerunning without
`--force` skips already-judged queries and only judges new/missing ones, so
it's safe to rerun after adding predictions or recovering from a judge
endpoint failure (a `401`/timeout on one query prints an error and records a
parse error for that query instead of aborting the run).

For each predictions file, writes to `--eval_dir` (default `./evals`):

- `{run}_eval.jsonl` -- per-query judge prompt/response/parsed verdict (the resumable cache)
- `{run}_summary.json` -- aggregate Accuracy/Exact Match/F1/Calibration Error + per-query metrics
- `{run}_detailed.csv` -- one row per query: predicted vs. correct answer, judge verdict, EM/F1

If multiple predictions files are evaluated in one invocation, also writes
`evaluation_overview.json` with all of their summaries.

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

## Alternative agent: Interact-RAG-style interaction (`mast_indic/interact_agent.py`)

[Interact-RAG (arXiv:2510.27566)](https://arxiv.org/abs/2510.27566) argues
that a single black-box `search(query)` tool is too coarse: it forces the
agent to encode every retrieval decision into one query string. Instead it
gives the agent an explicit **Corpus Interaction Engine** of composable
actions, and a **Global-Planner / Adaptive-Reasoner / Executor** workflow
for deciding when to use which.

Interaction actions live in `interact_engine.py`, on top of the same
`SearchIndex`:

- `semantic_search(query)` — dense embedding retrieval (what `agent.py`'s
  `search` tool does)
- `exact_search(keywords)` — sparse, case-insensitive keyword-count
  retrieval (a lightweight stand-in for BM25 — see the module docstring)
- `weighted_fusion(query, w_semantic, w_exact)` — blends the two for one query
- `entity_match(entity)` — ranks chunks by literal mentions of a named
  entity, dense similarity only breaking ties
- `include_docs(doc_ids)` / `exclude_docs(doc_ids)` — pin or filter specific
  documents across the rest of that query's retrievals
- `adjust_scale(n)` — changes how many chunks come back per retrieval

Unlike `agent.py` (one system prompt, one model, one growing conversation),
`interact_agent.py` reproduces the paper's three-module workflow as three
*separate* LLM calls with distinct prompts and no shared conversation
thread — each call gets only the context it needs, assembled from a
Python-side plan + evidence log rather than raw chat history:

- **Global-Planner** — one call per query, decomposes the question into a
  numbered plan. Plain text, no tools.
- **Adaptive-Reasoner** — one call per turn. Given the plan and the evidence
  gathered so far, it must call a forced `submit_decision` function
  returning `proceed` / `reflect_refine` / `ready_to_answer` plus a
  `strategy` instruction for the Executor (or, once ready, the final
  `explanation`/`exact_answer`). Structured output, not parsed free text.
- **Executor** — one call per turn (skipped once the Reasoner is ready to
  answer), translates that instruction into one or more of the action calls
  above (multiple tool calls per turn are supported, same as `agent.py`).

Each role defaults to `MAST_CHAT_MODEL`, but can be pointed at its own model
via `MAST_PLANNER_MODEL` / `MAST_REASONER_MODEL` / `MAST_EXECUTOR_MODEL` in
`.env` (e.g. a cheap model for planning, a stronger one for reasoning) — see
`config.py`.

**What this is not:** the paper trains its agent (SFT on synthetic
trajectories, then GRPO) to use these three roles well. There's no training
pipeline here — `interact_agent.py` is a zero-shot, prompted reproduction of
just the *interaction interface and module split*, relying on the chat
model's own instruction-following rather than a fine-tuned policy. Whether
it beats the plain single-`search` agent (or a single-prompt version of this
same workflow) depends entirely on how well your chosen model(s) follow
these narrower, more structured prompts zero-shot — and it costs roughly
2-3x the LLM calls per turn to find out.

```bash
./scripts/run_interact.sh --language hi --limit 5   # smoke test
./scripts/run_interact.sh --language hi              # full Hindi run
```

Writes `runs/{MAST_CHAT_MODEL}/interact_{language}.jsonl` — same record
shape as `runner.py` (`retriever` is tagged `interact-rag/{embed_model}` to
tell the two apart), so it evaluates with `eval.py` unchanged:

```bash
python -m mast_indic.eval \
  --predictions runs/gpt-4o-mini/interact_hi.jsonl \
  --gold data/hi_gold.jsonl
```

## What's stubbed vs. real

- **Real**: corpus loading, chunking, batch embedding, cosine search,
  both tool-calling agent loops (single-`search` and Interact-RAG-style),
  JSONL output, and LLM-judge answer scoring (`mast_indic/eval.py`) —
  Accuracy, Exact Match, F1, and calibration error.
- **Not included**: Interact-RAG's SFT+RL training pipeline — `interact_agent.py`
  is a zero-shot, prompted reproduction of its interaction interface only,
  not a trained policy (see above). Also not included: retrieval recall and
  citation precision/recall against qrels — there are no relevance-judgment
  files for the Indic queries, so `eval.py` only scores final-answer
  correctness. If real qrels become available, see the BrowseComp-Plus
  repo's `scripts_evaluation/` for the retrieval-metric approach to port over.
