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
  agent.py           # tool-calling agent loop, one dense `search` tool (OpenAI-compatible client)
  runner.py          # CLI: runs agent.py over a language's queries -> submission JSONL
  exact_agent.py      # same loop as agent.py, but one sparse/exact-keyword `search` tool
  exact_runner.py     # CLI: runs exact_agent.py over a language's queries -> submission JSONL
  interact_engine.py # Interact-RAG-style retrieval action primitives (arXiv:2510.27566)
  interact_agent.py  # agent loop using interact_engine.py instead of one `search` tool
  interact_runner.py # CLI: runs interact_agent.py over a language's queries -> submission JSONL
  entity_graph.py     # in-memory adjacency list loaded from index_store/entity_graph.jsonl (bring your own extraction)
  graph_builder.py    # optional/legacy: LLM-extracts a simpler graph, but not the schema entity_graph.py reads
  export_corpus.py    # CLI: dumps each corpus document as its own Markdown file
  eval.py            # CLI: LLM-judge scoring of a run against gold answers
scripts/
  build_index.sh
  run_track2.sh
  run_exact.sh
  run_interact.sh
  build_entity_graph.sh
  export_corpus.sh
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
overlapping character-based chunks (default 8192 chars ≈ 2048 tokens, 200
chars ≈ 50 tokens overlap, via a 1-token-≈-4-characters approximation),
embeds every chunk via the configured embedding server, and writes:

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
`MAST_CHUNK_CHARS`/`MAST_EMBED_MODEL`).

**Scaling note:** the full corpus is ~100k docs / ~5,200 words average, so
full-corpus indexing means embedding several hundred thousand chunks locally —
expect this to take a while and to use several GB of RAM for the brute-force
matrix. For iteration, use `--limit`. For a real submission at scale, either
run indexing against a GPU-backed embedding server (e.g. vLLM), or swap
`SearchIndex` for FAISS / the BrowseComp-Plus repo's pre-built BM25 or
Qwen3-Embedding indexes
(see `scripts_build_index/download_indexes.sh` in
[texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)).

**Browsing the raw corpus:** `export_corpus.py` dumps each document as its
own `.md` file, independent of the embedding index -- no API keys, no LLM,
just streams `MAST_CORPUS_DATASET` and writes `{docid}.md` (title, source
URL if present, then the full text) into an output directory:

```bash
./scripts/export_corpus.sh --limit 200                        # dev-scale subset
./scripts/export_corpus.sh --output-dir corpus_markdown        # full corpus (many files)
```

Resumable by default (skips a docid whose `.md` file already exists); pass
`--fresh` to re-export everything. Useful for manually checking what a
document actually says when a search result looks off -- grep/open the file
directly rather than only ever seeing a truncated snippet.

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

**Partial results survive a mid-run failure.** If a query hits a timeout,
connection drop, or 5xx partway through (say, on turn 8 of a long-running
one), the turns already completed aren't discarded -- `result` still
contains everything gathered before the failure, with a trailing
`{"type": "error", "output": "Agent failed mid-run: ..."}` entry and a
top-level `"error"` field on the record (also picked up by `eval.py` and
`tools/trace_viewer.html` the same way an incomplete run already is). Only
a crash *before* the first LLM call (e.g. a bug outside the agent loop
itself) falls back to the old all-or-nothing `"result": []` shape.

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

## Alternative agent: exact-search-only baseline (`mast_indic/exact_agent.py`)

Same loop as `agent.py` above -- one system prompt, one `search` tool, one
growing conversation -- but the tool is backed by
`CorpusInteractionEngine.exact_search` (sparse, distinct-term-coverage
keyword matching) instead of `SearchIndex.search`'s dense cosine similarity.
No other tools, no dense retrieval fallback, and (since exact-search never
calls the embedding server) no embedding calls at runtime at all -- `index.py
build` still needs to have been run once, though, since this searches over
the same `index_store/meta.jsonl` chunk text.

The system prompt is adjusted to match: it tells the model this tool only
does literal keyword matching, so it should write short specific
keywords/names/phrases rather than paraphrased natural-language questions.

**Planning step:** before the search loop starts, a separate LLM call
(mirroring `interact_agent.py`'s Global-Planner, sharing its
`MAST_PLANNER_MODEL` config) decomposes the question into a list of ATOMIC
sub-queries -- short keyword-style phrases, one per distinguishing fact
(e.g. `"founded 1949 1959"`, `"Lady Shri Ram College"`), not the kind of
broader sub-questions a general planner would produce, since exact_search
specifically needs something already shaped like a usable keyword query.
The plan is logged in `result` (tagged `role: "planner"`) and injected into
the conversation as context for the model to use as a starting point -- it's
not a tool the model calls, and the search loop that follows is otherwise
identical to `agent.py`'s: one tool, no forced ordering through the plan.

This exists as a sparse-only counterpart to `agent.py`'s dense-only
baseline, so you can compare dense-only vs. sparse-only vs. the full
multi-tool `interact_agent.py` on the same queries:

```bash
./scripts/run_exact.sh --language hi --limit 5   # smoke test
./scripts/run_exact.sh --language hi              # full Hindi run
```

Writes `runs/{MAST_CHAT_MODEL}/exact_{language}.jsonl` -- same record shape
as the other two agents (`retriever` is tagged `exact-search`), so it
evaluates with `eval.py` unchanged.

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
- `exact_search(keywords)` — sparse retrieval, real Okapi BM25 over an
  in-memory inverted index built once per engine instance (term-frequency
  saturation + document-length normalization + IDF — see the module
  docstring for why that matters over a naive keyword-count)
- `boolean_search(and_terms, or_terms, not_terms)` — exact AND/OR/NOT set
  retrieval over the same inverted index, with **no ranking**: every match
  is equally valid. For when you need strict logical control rather than a
  best-guess ranking
- `weighted_fusion(query, w_semantic, w_exact)` — blends dense with BM25
  for one query
- `entity_match(entity)` — ranks chunks by literal mentions of a named
  entity, dense similarity only breaking ties
- `graph_search(entity, hops)` — multi-hop traversal of a pre-built entity
  relationship graph (see below); returns nothing until you build one
- `include_docs(doc_ids)` / `exclude_docs(doc_ids)` — pin or filter specific
  documents across the rest of that query's retrievals
- `adjust_scale(n)` — changes how many chunks come back per retrieval

Unlike `agent.py` (one system prompt, one model, one growing conversation),
`interact_agent.py` reproduces the paper's three-module workflow as three
*separate* LLM calls with distinct prompts and no shared conversation
thread — each call gets only the context it needs, assembled from a
Python-side plan + evidence log rather than raw chat history:

- **Global-Planner** — one call per query. Plain text, no tools; the prompt
  follows Interact-RAG's actual published planner prompt: a single
  comprehensive query for direct questions, decomposition into simple
  sub-tasks (further subdivided if not simple enough) for complex ones,
  flagging sub-tasks that don't depend on each other and can run in
  parallel, and no answering or inventing facts itself. Output is free-text
  reasoning followed by a `Primary Plan:` numbered list, passed to the
  Reasoner as-is (not parsed).
- **Adaptive-Reasoner** — one call per turn, also free text: Interact-RAG's
  actual published reasoner prompt. Summarizes findings so far, then chooses
  one of three paths -- A) Proceed (propose the next search, up to two in
  parallel), B) Conclude (announce completion, summarize key findings), C)
  Reflect & Refine (diagnose why the last search failed, propose a different
  one; after 3 failed attempts on one sub-task, move on). This is advisory
  analysis only, not a control-flow gate -- **the Reasoner never ends the
  episode itself**, matching the paper. Which path it narrated is recovered
  by regex purely for trace/debug purposes (`_parse_reasoner_response`).
- **Executor** — one call *every* turn, also Interact-RAG's actual published
  prompt. Given the original question plus the Reasoner's analysis, it alone
  decides each turn whether to call up to 2 of the action calls above, or
  formulate the final answer instead -- matching the paper's actual control
  flow, where the Executor (not the Reasoner) is the one that concludes.
  The one deliberate deviation from the paper: its Executor's final answer
  is just "concise and direct words," but ours must end with fixed
  `Explanation:`/`Exact Answer:` lines, since `eval.py` depends on that
  format to score answers (`_ensure_final_answer_format` wraps whatever the
  model actually said if it didn't comply). Termination is still guaranteed:
  on the last allowed turn the Executor's `tool_choice` is forced to
  `"none"` (same trick `agent.py` uses), and if a non-compliant server
  somehow searches anyway on that turn, a fallback answer still gets
  appended rather than ending with no answer at all.

Both the Reasoner and Executor prompts explicitly teach dense-vs-sparse
retrieval tradeoffs, since naively dumping a whole multi-clause question
into one query tends to fail either way: dense retrieval averages a long
sentence into a vague embedding that matches broadly-related-but-wrong
documents, while sparse retrieval is literal keyword matching, so a common
word repeated many times in one irrelevant document (e.g. a dictionary
entry) can outrank a genuinely relevant chunk that only mentions it once.
The prompts push toward decomposing a multi-constraint question into one
distinguishing sub-fact per turn, short single-concept queries for
`semantic_search`, and short exact phrases/names/numbers for
`exact_search`/`entity_match` rather than descriptive paraphrases.

**Bounded evidence context:** every turn's search results get appended to
an evidence log that's restated to the Reasoner on the *next* turn so it
can judge what's been found so far. Restating the full history unbounded
would grow with turn count and can overflow the model's context window on
a long-running query -- or even from a single oversized `adjust_scale` call
(up to 50 chunks x ~400-word snippets is ~26k tokens by itself). Past
`MAST_EVIDENCE_CHAR_BUDGET` (default 60000 characters), older rounds are
dropped first (noted as `"N earlier search round(s) omitted"`); a single
round that alone exceeds the budget is hard-truncated rather than dropped,
so the Reasoner always sees at least the most recent result. This only
bounds what's sent to the LLM -- the full evidence still ends up in the
run's own `result`/`transcript` trace.

Each role defaults to `MAST_CHAT_MODEL`, but can be pointed at its own model
via `MAST_PLANNER_MODEL` / `MAST_REASONER_MODEL` / `MAST_EXECUTOR_MODEL` in
`.env` (e.g. a cheap model for planning, a stronger one for reasoning) — see
`config.py`.

**Tracing what each role did:** every step lands in the run JSONL's `result`
field (always written, no flag needed) tagged with `role`
(`planner`/`reasoner`/`executor`) and `model`; reasoner steps additionally
carry the parsed `decision` object (`decision` and the full free-text
`analysis`), so the analysis that drove the Executor is never lost even
though it was never a structured field to begin with. For the exact
system/user messages sent to and parsed back from each role's API call
(full request/response reproduction, not just the distilled trace), add
`--save-transcripts` to get a `transcript` field alongside `result`. Add
`--debug` to stream the same decisions live to stderr as they happen.

**Visualizing a run:** open `tools/trace_viewer.html` directly in a browser
(no server, no build step) and drop a run JSONL onto it to interactively
walk through each query's Planner → Reasoner → Executor turns -- decisions,
search results with scores/snippets, and the final answer, all rendered
turn by turn. Nothing leaves the browser; it reads the file locally via the
File API. Passing `--save-transcripts` when generating the run lets it also
show the original question text inline.

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

### Entity relationship graph (`graph_search`)

`graph_search` needs an actual graph built first -- it returns nothing
until you do. Bring your own extraction (this project doesn't run one for
you) and place it at `index_store/entity_graph.jsonl`, one JSON object per
chunk:

```json
{
  "doc_id": "41758__chunk_0001",
  "entities": [{"id": "vikings", "name": "Vikings", "type": "Team"}, ...],
  "concepts": [{"id": "concept_historical_drama", "name": "Historical Drama",
                 "description": "Series set in specific historical periods...", "level": 1}, ...],
  "triplets": [{"head": "Vikings", "relation": "aired_on", "tail": "History",
                 "tail_is_concept": false}, ...]
}
```

- `doc_id` is `"{docid}__chunk_NNNN"` (1-indexed) -- `entity_graph.py` splits
  it back into `(docid, chunk_id)` to source a `graph_search` hit's URL and
  provenance. Exact chunk-index alignment with this project's own
  `index_store/meta.jsonl` isn't required for correctness: if the specific
  chunk isn't found, it falls back to any chunk of the same `docid`.
- `entities`/`concepts` are enrichment (types, descriptions), not a strict
  node universe -- a `head`/`tail` naming something outside either list is
  still a perfectly valid graph node, just without a known type/description.
- `tail_is_concept` marks whether a triplet's tail refers to something in
  `concepts` rather than a literal named entity; `graph_search` surfaces
  this as a `[concept]` tag on the relation so the Executor doesn't mistake
  a concept for another entity worth chasing further.

`entity_graph.py` loads that file into an in-memory adjacency list, indexed
in both directions (head→tail and tail→head) so a lookup finds a node
regardless of which side of the triplet named it. The agent's
`graph_search(entity, hops)` action (1-3 hops) BFS-traverses it and turns
each nearby relation back into a source-chunk hit, deduped per document,
with the relation itself as the snippet (e.g. `"Vikings aired_on History;
Vikings distributed_on Prime Video"`) rather than raw chunk text -- so the
Executor sees *why* a document matched. Multi-hop genuinely resolves in one
call this way (entity → relation → entity → relation → entity), which is
the whole reason this action exists alongside the text-search ones.

**This is only as good as your own extraction pipeline** -- `entity_graph.py`
just loads and traverses whatever you hand it; expect the usual
LLM/NER-extraction caveats (missed relations, inconsistent entity naming
across chunks since no canonicalization is attempted here, occasional
hallucinated relations) to be whatever your own pipeline's caveats are.
Treat `graph_search` results as a pointer back to a real chunk worth
verifying, not as ground truth. If `index_store/entity_graph.jsonl` doesn't
exist, `graph_search` just returns an empty list and the agent falls back
to `entity_match`/`exact_search` (both prompts say to do this).

`graph_builder.py`/`scripts/build_entity_graph.sh` are a self-contained,
optional LLM-extraction pipeline (asks the chat model for plain
`(subject, relation, object)` triples per chunk) kept here in case you don't
have your own -- but its output (`index_store/relations.jsonl`, the older,
simpler schema) is a different shape than `entity_graph.jsonl` above and
needs its own loader if you want to use it; `entity_graph.py` no longer
reads it directly.

## What's stubbed vs. real

- **Real**: corpus loading, chunking, batch embedding, cosine search,
  all three tool-calling agent loops (dense-only, sparse-only, and
  Interact-RAG-style), entity-relationship graph loading and multi-hop
  traversal (`entity_graph.py`, plus an optional/legacy LLM-extraction
  pipeline in `graph_builder.py`), JSONL output, and LLM-judge answer
  scoring (`mast_indic/eval.py`) — Accuracy, Exact Match, F1, and
  calibration error.
- **Not included**: Interact-RAG's SFT+RL training pipeline — `interact_agent.py`
  is a zero-shot, prompted reproduction of its interaction interface only,
  not a trained policy (see above). The entity graph's *extraction* isn't
  this project's responsibility -- `entity_graph.py` only loads and
  traverses whatever `index_store/entity_graph.jsonl` you provide (see the
  graph section above); treat `graph_search` results as a hint pointing
  back to a real chunk, not ground truth. Also not included: retrieval
  recall and citation
  precision/recall against qrels — there are no relevance-judgment files
  for the Indic queries, so `eval.py` only scores final-answer correctness.
  If real qrels become available, see the BrowseComp-Plus repo's
  `scripts_evaluation/` for the retrieval-metric approach to port over.
