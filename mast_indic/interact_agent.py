"""A zero-shot Interact-RAG-style agent (arXiv:2510.27566) for Track 2 (Indic).

Where `agent.py` gives the LLM one black-box `search` tool, this agent
exposes the paper's fine-grained Corpus Interaction Engine actions
(`mast_indic/interact_engine.py`): dense/sparse/fused retrieval,
entity-anchored matching, and context shaping (pin/filter documents, resize
result sets). It also reproduces the paper's *three-module* workflow as
three separate LLM calls with distinct prompts, rather than one agent
narrating all three roles in a single call:

- Global-Planner: one call per query, decomposes the question into a plan.
  Free text (Interact-RAG's actual published planner prompt).
- Adaptive-Reasoner: one call per turn, judges the evidence gathered so far
  and chooses one of three paths -- A) Proceed, B) Conclude, C) Reflect &
  Refine (Interact-RAG's actual published reasoner prompt) -- but this is
  advisory analysis only, not a control-flow gate: the Reasoner never
  decides to end the episode itself, matching the paper. Free text; which
  path was chosen is recovered by regex (`_parse_reasoner_response`) purely
  for trace/debug purposes.
- Executor: one call every turn (also Interact-RAG's actual published
  prompt), given the original question plus the Reasoner's analysis. It
  alone decides, each turn, whether to call up to 2 of the actions below or
  formulate the final answer -- matching the paper's actual control flow,
  where the Executor (not the Reasoner) is the one that concludes.

Beyond those three, there's a fourth, smaller call added on top of the
paper's design: after the Executor's tool call(s) each turn, a short LLM
call (`_run_scratchpad`) condenses that turn's goal and raw tool output into
a compact note -- `Goal:`/`Observations:`/`Achieved:`/`Next Steps:` -- which
is what the Adaptive-Reasoner reads on the *next* turn instead of the raw
tool output. This isn't part of the published pipeline; it's a bounded-memory
fix for a real failure mode (a long-running query's raw evidence either grows
past the model's context window, or gets silently dropped once it does) --
a note is a few sentences regardless of how many chunks a search returned,
so nothing needs to be truncated as turns accumulate.

The one deliberate deviation from the paper's *published* roles: its Executor's final answer is
just "concise and direct words," but ours must end with fixed
`Explanation:`/`Exact Answer:` lines, since `mast_indic/eval.py` depends on
that format to score answers. Termination is still guaranteed the same way
`agent.py` guarantees it: on the last allowed turn, the Executor's
`tool_choice` is forced to `"none"`, so it cannot call a tool and must
answer in text.

Each role can optionally run on its own model (`MAST_PLANNER_MODEL` /
`MAST_REASONER_MODEL` / `MAST_EXECUTOR_MODEL` / `MAST_SCRATCHPAD_MODEL` in
`config.py`); all four default to `MAST_CHAT_MODEL` if unset, so nothing
changes unless you split them out.

This is a zero-shot, prompted reproduction of the paper's *interaction
interface and module split* -- there's no SFT+RL training pipeline here,
just three prompted roles driving the same tool-calling machinery this
project already uses elsewhere.

Output record shape matches `agent.py`/`runner.py` exactly (reusing
`AgentResult`), so runs from this agent drop straight into
`mast_indic/eval.py` unchanged. Each `result` trace entry also carries a
`role` field (`planner`/`reasoner`/`executor`) for debugging -- `eval.py`
ignores it.
"""
from __future__ import annotations

import json
import re
import sys

from openai import OpenAI

from .agent import AgentResult
from .config import config
from .index import SearchIndex
from .interact_engine import CorpusInteractionEngine


def _debug(msg: str) -> None:
    if config.debug:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


PLANNER_SYSTEM_PROMPT = """You are the Global-Planner in an Interact-RAG-style \
research pipeline (MAST @ FIRE 2026, Track 2: Indic), an expert research assistant \
focused on high-level planning. You will be given a complex question written in an \
Indian language; the evidence corpus and the eventual answer are in English. An \
Adaptive-Reasoner and Executor downstream will carry out your plan by searching the \
corpus -- your job is only to plan, not to search or answer.

## Your planning process
- Thoroughly analyze the question. Identify key concepts, entities, and any constraints.
- If the question is direct or straightforward, plan for a single, comprehensive search \
  covering it -- e.g. for "when was the last time France hosted the Olympics", the plan \
  is just that one lookup.
- If the question is complex, break it down into clear, specific sub-tasks. Each sub-task \
  must be simple and direct; if one isn't, divide it further into smaller steps.
- Some sub-tasks may be pursued in parallel (they don't depend on each other's results) \
  -- point this out when it's the case.

## Other requirements
Do not rely on uncommon internal knowledge in your analysis or plan, since it may be \
inaccurate -- only reason about what the question itself states. Do not try to answer \
the question yourself; just provide the research plan.

## Expected output
First, briefly think through the analysis above in natural, connected language (e.g. \
"Okay, ... Then, ... Therefore, ..."). Then output the plan as a numbered list prefixed \
with "Primary Plan:", for example:
Primary Plan: 1. Determine the director of the film "Polish-Russian War". 2. Identify \
the birthplace of that director. 3. Formulate the final answer.
"""

REASONER_SYSTEM_PROMPT = """You are an expert research strategist for the MAST @ \
FIRE 2026 benchmark (Track 2: Indic), the Adaptive-Reasoner in an Interact-RAG-style \
research pipeline. Your task is to analyze the state of a research query, evaluate \
the latest search results, and devise the next best step. You should only generate \
the plan for the next action, not execute it yourself -- an Executor downstream will \
carry it out.

## Your instructions
You should first briefly summarize the relevant key findings from the previous \
search(es). State what information has been gathered and what is still missing.

Based on that, reasonably choose one of the following three paths, then analyze and \
propose the next step:
A) Proceed: choose this if the last search successfully answered the current \
   sub-question. State the key information that was found, then propose the next \
   logical search with appropriate parameters. You can propose up to two parallel \
   searches if needed.
B) Conclude: choose this if the whole question is resolved and you have sufficient \
   information to answer the original query. Announce that the research is complete \
   and provide a concise summary of all key findings.
C) Reflect & Refine: choose this if the previous search was ineffective (irrelevant, \
   incomplete, or low-quality results). First, briefly explain why the search failed. \
   Then propose a refined search action with improved parameters -- account for how \
   the retrieval modes actually behave: dense/semantic_search embeds the whole query \
   into one vector, so a long sentence chaining unrelated constraints together dilutes \
   into a vague, often-wrong match; sparse/exact_search is literal keyword matching \
   (BM25-ranked -- graded by relevance, not just presence), good for a name/number/exact \
   phrase; boolean_search is also literal keyword matching but with NO ranking at all, \
   useful when you need strict AND/OR/NOT control rather than a best-guess match. Once a \
   specific named entity is known, prefer graph_search over another exact_search on that \
   same entity -- it can resolve a chain of relations (who founded X, what else did X \
   found, who else is connected to X) in one call instead of one hop at a time. If a \
   sub-task remains unresolved after 3 attempts, consider moving on to the next one.

Do not include your uncommon internal knowledge in this analysis, as it may be \
inaccurate -- only reason about the evidence log below.

## Output format
- For both Proceed and Reflect & Refine, concisely and reasonably analyze and suggest \
  the parameters for the next search.
- Format your entire output as natural language -- use fluent, connective expressions \
  (e.g. "Okay, ... Then, ... Therefore, ..."). This applies to all three paths, \
  including Conclude -- you are analyzing and summarizing, not writing the final answer \
  itself; an Executor downstream does that.
- You don't need to conclusively list every parameter at the end. Keep your output \
  concise but clear.
"""

EXECUTOR_SYSTEM_PROMPT = """You are a specialized searching execution agent, the \
Executor in an Interact-RAG-style research pipeline (MAST @ FIRE 2026, Track 2: \
Indic). You will be presented with the user's original question (possibly written in \
an Indian language; the evidence corpus and your final answer are in English), prior \
search results, and the Adaptive-Reasoner's analysis of them. Your sole purpose is to \
perform one of two specific actions each turn -- either call tool(s) or provide the \
final answer:

1. Execute a Search: based on the Reasoner's analysis, identify the proper action(s) \
   and parameters and call the appropriate tool(s) below -- if the analysis doesn't \
   spell out exact parameters, formulate them yourself from the question and evidence \
   so far. You can make up to 2 separate calls in one turn if needed (e.g. the \
   analysis points at two independent sub-tasks that can be pursued in parallel) -- \
   never more than 2.
2. Formulate the Final Answer: if the Reasoner's analysis says the research is \
   complete, or the evidence gathered is otherwise sufficient to answer the user's \
   whole original question, respond with a final answer instead of calling any tool. \
   The final answer must be concise and direct, in exactly this format:
   Explanation: <one or two sentences citing what you found>
   Exact Answer: <the short factual answer alone, e.g. "Marie Curie" or "1912">

Available actions (all operate over the same English evidence corpus; only relevant \
when taking action 1 above):
- semantic_search(query): dense embedding retrieval -- good for \
  paraphrased/conceptual matches.
- exact_search(keywords): sparse retrieval, BM25-ranked -- good for names, \
  numbers, and exact phrases that embeddings tend to blur together, while \
  still graded by relevance (a chunk covering more of your keywords, \
  proportionally to its length, ranks higher).
- boolean_search(and_terms, or_terms, not_terms): exact set-based retrieval \
  with NO ranking -- every match is equally valid. Use when you need strict \
  logical control ("must mention X and Y, but never Z") rather than a \
  best-guess ranking; returns nothing if you give it only not_terms (too \
  unconstrained to be useful).
- weighted_fusion(query, w_semantic, w_exact): blend dense and BM25 scoring \
  for one query when neither alone is enough (weights need not sum to 1).
- graph_search(entity, hops): traverse a pre-built entity relationship graph \
  outward from a named entity (1-3 hops) -- this is the go-to action once \
  you have a specific entity name, for explicit relationship questions \
  ("who founded X", "what else did X found", "who else is connected to X") \
  and for hopping onto a second fact about an entity you've already found \
  (e.g. you've identified a company's founder as a named person -- now \
  traverse outward from them to find where they were born). It can resolve \
  a multi-hop chain in one call, \
  unlike the keyword/embedding actions above. Returns nothing if the graph \
  hasn't been built for this corpus; fall back to exact_search if so.
- include_docs(doc_ids): pin specific document IDs so they're guaranteed to \
  appear in later retrievals (e.g. a document you've already confirmed is \
  relevant).
- exclude_docs(doc_ids): filter out document IDs you've confirmed are \
  irrelevant or noisy, so later retrievals stop surfacing them.
- adjust_scale(n): change how many chunks come back per retrieval (smaller \
  once you've narrowed in for precision, larger while still exploring).

When writing the actual query/keywords/entity argument, respect what each \
action is good at -- do not just restate the Reasoner's full analysis \
verbatim:
- semantic_search(query): a short, single-concept natural-language phrase \
  (roughly 5-15 words), written the way you'd ask a person or type into a \
  search box. NEVER a structured/SQL-like expression -- bad: "restaurant \
  WHERE cuisine = Italian AND city = Chicago AND price_range = expensive"; \
  good: "upscale Italian restaurant in Chicago". Never concatenate multiple \
  unrelated constraints (dates, categories, locations, etc.) into one query \
  either, structured or not -- that dilutes the embedding and returns \
  vaguely-related noise instead of a precise match.
- exact_search(keywords): a short exact phrase, proper noun, number, or \
  date you expect to appear verbatim in the corpus. Avoid long descriptive \
  phrases here -- a common word repeated many times in an unrelated \
  document can outrank a genuinely relevant chunk that only mentions it once.
- graph_search(entity, hops): just the entity name itself, exactly as it \
  appeared in an earlier result -- not a question or descriptive phrase.
- boolean_search(and_terms, or_terms, not_terms): each entry should be one \
  word or short phrase, not a sentence -- e.g. and_terms=["bank", "ceo"], \
  not_terms=["dictionary"]. Only use this when the Reasoner's analysis (or \
  the question itself) genuinely calls for strict inclusion/exclusion logic \
  rather than "find the best match."
- weighted_fusion(query, w_semantic, w_exact): use for ONE reasonably \
  specific query when you want to hedge between its literal and conceptual \
  interpretation -- not as a way to combine several different facts into a \
  single call.

If the Reasoner's analysis bundles multiple facts together into one proposed \
search, pick the single most distinctive one to search for now and leave the \
rest for a later turn -- unless it explicitly proposed two independent \
parallel searches, in which case call both.
"""

# Not one of the paper's three modules -- an addition on top (see the module
# docstring) that turns each turn's raw tool output into a compact note the
# Adaptive-Reasoner reads instead, so its context stays small and concrete
# no matter how many turns a query takes.
SCRATCHPAD_SYSTEM_PROMPT = """You maintain the running scratchpad for an \
Interact-RAG-style research pipeline (MAST @ FIRE 2026, Track 2: Indic). You will be \
given the current turn's goal (what the Executor was trying to find or verify) and the \
tool call(s) it made along with their raw results. Condense this turn's outcome into a \
compact note -- the Adaptive-Reasoner reads your notes turn by turn instead of the raw \
results, so be concrete about what was actually found (names, dates, docids), not just \
that a search "was run."

Output exactly four lines, in this format, and nothing else:
Goal: <one sentence -- what this turn was trying to find or verify>
Observations: <what the tool result(s) actually showed, concretely>
Achieved: yes|partial|no -- <one short clause on why>
Next Steps: <what to try next, or "none -- ready to conclude" if the whole question is answered>
"""

INTERACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Dense embedding retrieval: ranks chunks by conceptual similarity to a "
                "natural-language query. `query` must be plain prose, written the way "
                "a person would ask it or type it into a search box -- NEVER a "
                "structured/SQL-like expression. Bad: \"restaurant WHERE cuisine = "
                "Italian AND city = Chicago AND price_range = expensive\". "
                "Good: \"upscale Italian restaurant in Chicago\". Best for a single, "
                "self-contained factual question or a paraphrase-tolerant lookup. "
                "Weak on complex multi-hop questions that "
                "chain several linked facts in one query -- it embeds the whole query "
                "into one vector, so packing multiple sub-facts together (structured "
                "or not) dilutes rather than resolves them; resolve one hop per call "
                "instead. Also weak at pinpointing exact names/numbers, which it can "
                "conflate with semantically similar but distinct ones -- use "
                "exact_search for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "An English natural-language search query."},
                    "top_k": {"type": "integer", "description": "Optional: how many chunks to return from just this call (1-50), overriding the current adjust_scale value. Omit to use the current scale."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exact_search",
            "description": (
                "Sparse BM25 retrieval: ranks chunks by how well they cover specific "
                "keywords/names/numbers/phrases, weighted by term rarity and chunk "
                "length. Best for anchoring on a term you expect to appear verbatim -- a "
                "name, date, title, or number embeddings tend to blur together with "
                "similar ones. Weak on paraphrases or synonyms (a chunk describing the "
                "same fact in different words won't match), and -- like semantic_search "
                "-- only resolves one hop per call; it can't itself bridge 'A relates to "
                "B, and B relates to C.'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Space-separated keywords or a phrase to match literally."},
                    "top_k": {"type": "integer", "description": "Optional: how many chunks to return from just this call (1-50), overriding the current adjust_scale value. Omit to use the current scale."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "boolean_search",
            "description": (
                "Exact AND/OR/NOT set retrieval over keywords, with NO relevance "
                "ranking -- every match is equally valid, there's no 'close enough.' "
                "Best for precisely narrowing an already-identified small set of "
                "candidates (must mention X and Y, must not mention Z) once you know "
                "the exact terms that separate them. Weak as a first/exploratory "
                "search since it can't rank or suggest anything close to a term you "
                "didn't specify, and returns nothing if you give only not_terms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "and_terms": {"type": "array", "items": {"type": "string"}, "description": "Chunk must contain ALL of these terms."},
                    "or_terms": {"type": "array", "items": {"type": "string"}, "description": "Chunk must contain AT LEAST ONE of these terms."},
                    "not_terms": {"type": "array", "items": {"type": "string"}, "description": "Chunk must contain NONE of these terms."},
                    "top_k": {"type": "integer", "description": "Optional: how many chunks to return from just this call (1-50), overriding the current adjust_scale value. Omit to use the current scale."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weighted_fusion",
            "description": (
                "Blends dense (semantic) and BM25 (exact) scoring for one query. Best "
                "when you're unsure whether the answer surfaces via conceptual "
                "similarity or a literal keyword match, and want to hedge rather than "
                "commit to one. Inherits both underlying limits: still a single query "
                "per call (no multi-hop), and a vague/generic query with high w_semantic "
                "can wash out what would otherwise be a strong exact-keyword signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "An English natural-language search query."},
                    "top_k": {"type": "integer", "description": "Optional: how many chunks to return from just this call (1-50), overriding the current adjust_scale value. Omit to use the current scale."},
                    "w_semantic": {"type": "number", "description": "Weight for dense/embedding similarity, e.g. 0.7."},
                    "w_exact": {"type": "number", "description": "Weight for sparse/keyword overlap, e.g. 0.3."},
                },
                "required": ["query", "w_semantic", "w_exact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": (
                "Traverses a pre-built entity relationship graph outward from a named "
                "entity, 1-3 hops. This is the go-to action once you have a specific "
                "named entity: unlike every other search action here, it CAN resolve a "
                "multi-hop chain (entity -> relation -> entity -> relation -> entity) in "
                "a single call, and its snippets are the extracted relations themselves "
                "rather than raw chunk text, so you see exactly why a document matched. "
                "It returns nothing if the graph hasn't been built for this corpus -- "
                "fall back to exact_search then. The graph is LLM-extracted and may be "
                "incomplete, so an empty result doesn't prove a relationship doesn't exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The entity name to start from, in English."},
                    "hops": {"type": "integer", "description": "How many relationship hops to traverse outward (1-3), default 1."},
                    "top_k": {"type": "integer", "description": "Optional: how many documents to return from just this call (1-50), overriding the current adjust_scale value. Omit to use the current scale."},
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "include_docs",
            "description": (
                "Pins specific document IDs so they're guaranteed to appear in later "
                "retrievals. Use to lock in a document you've already confirmed is "
                "relevant, so scale limits or reranking in later calls can't drop it. "
                "Not a search action itself -- it only shapes what later searches surface."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "docids from earlier search results."},
                },
                "required": ["doc_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exclude_docs",
            "description": (
                "Filters specific document IDs out of later retrievals. Use to "
                "permanently remove a document you've confirmed is irrelevant or "
                "noisy, so it stops resurfacing across subsequent calls. Not a search "
                "action itself -- it only shapes what later searches surface."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "docids from earlier search results."},
                },
                "required": ["doc_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_scale",
            "description": (
                "Changes the default number of chunks that come back on every later "
                "retrieval that doesn't specify its own top_k. Increase while still "
                "exploring broadly; decrease once you've narrowed in, to cut noise. "
                "Not a search action itself -- it only shapes result size for later "
                "calls. Any search action's own `top_k` argument overrides this for "
                "just that one call, without changing the default for the rest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of chunks to return per retrieval, e.g. 3 or 10."},
                },
                "required": ["n"],
            },
        },
    },
]

# Actions that retrieve evidence (tracked in retrieved_docids); the rest
# (include_docs/exclude_docs/adjust_scale) only mutate interaction state.
RETRIEVAL_ACTIONS = {
    "semantic_search", "exact_search", "boolean_search", "weighted_fusion", "graph_search",
}

# Dense retrieval embeds the whole query into one vector -- packing several
# constraints into one query string ("X founded in Y, alumni A, B, C, and D")
# dilutes that vector toward a generic match for *any* one clause instead of
# a sharp match for the document satisfying *all* of them. The tool
# descriptions and Executor prompt already tell the model not to do this,
# but a model that ignores that guidance can burn an entire run re-issuing
# the same overcompound query with cosmetic rewording (observed on real
# traces: 50+ semantic_search calls, all restatements of one 5-clause
# question, none of them converging). This is a blunt backstop, not a real
# parser -- word count + comma/and/or clause count -- so it only catches the
# common "long list of constraints" shape, not every way a query can be too
# broad.
_MAX_QUERY_WORDS = 14
_MAX_QUERY_CLAUSES = 3
_CLAUSE_SPLIT_RE = re.compile(r",|\band\b|\bor\b", re.IGNORECASE)


def _overcompound_query_error(query: str) -> str | None:
    """Returns an error message if `query` packs multiple constraints into
    one search string, or None if it looks like a single atomic query."""
    words = query.split()
    clauses = len([c for c in _CLAUSE_SPLIT_RE.split(query) if c.strip()])
    if len(words) <= _MAX_QUERY_WORDS and clauses <= _MAX_QUERY_CLAUSES:
        return None
    preview = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
    return (
        f'query rejected as overcompound: "{preview}" packs multiple facts '
        "into one search. Dense retrieval embeds the whole query into a "
        "single vector, so combining several constraints dilutes it into a "
        "vague match instead of resolving any one precisely. Split this into "
        "separate single-fact searches (one per date/entity/attribute) and "
        "narrow down using include_docs/exclude_docs across turns, rather "
        "than combining constraints into one query."
    )


def _dispatch(engine: CorpusInteractionEngine, name: str, args: dict) -> tuple[str, list[str]]:
    """Run one action against the engine; returns (tool_output_text, docids)."""
    top_k = int(args["top_k"]) if args.get("top_k") is not None else None
    if name == "semantic_search":
        query = args.get("query", "")
        error = _overcompound_query_error(query)
        if error:
            return json.dumps({"error": error}), []
        hits = engine.semantic_search(query, top_k)
    elif name == "exact_search":
        hits = engine.exact_search(args.get("keywords", ""), top_k)
    elif name == "boolean_search":
        hits = engine.boolean_search(
            and_terms=args.get("and_terms") or [],
            or_terms=args.get("or_terms") or [],
            not_terms=args.get("not_terms") or [],
            top_k=top_k,
        )
    elif name == "weighted_fusion":
        query = args.get("query", "")
        error = _overcompound_query_error(query)
        if error:
            return json.dumps({"error": error}), []
        hits = engine.weighted_fusion(
            query,
            float(args.get("w_semantic", 0.5)),
            float(args.get("w_exact", 0.5)),
            top_k,
        )
    elif name == "graph_search":
        hits = engine.graph_search(args.get("entity", ""), int(args.get("hops", 1) or 1), top_k)
    elif name == "include_docs":
        return engine.include_docs(args.get("doc_ids", []) or []), []
    elif name == "exclude_docs":
        return engine.exclude_docs(args.get("doc_ids", []) or []), []
    elif name == "adjust_scale":
        return engine.adjust_scale(args.get("n", 5)), []
    else:
        return json.dumps({"error": f"unknown action '{name}'"}), []

    docids = [h.docid for h in hits]
    payload = [
        {"docid": h.docid, "url": h.url, "score": round(h.score, 4), "snippet": h.snippet}
        for h in hits
    ]
    return json.dumps(payload, ensure_ascii=False), docids


def _format_scratchpad(scratchpad_log: list[dict]) -> str:
    """Renders the accumulated per-turn scratchpad notes for the Reasoner.
    Each entry is already a short, LLM-condensed summary (`_run_scratchpad`)
    rather than a turn's raw tool output, so unlike the evidence dump this
    replaces, there's no character budget or truncation here -- even a few
    dozen turns of notes stays tiny next to a single turn's raw JSON hits.
    """
    if not scratchpad_log:
        return "(no searches performed yet)"
    return "\n\n".join(f"--- Turn {e['turn']} ---\n{e['note']}" for e in scratchpad_log)


_DEFAULT_DECISION = {
    "decision": "reflect_refine",
    "analysis": "(reasoner returned no usable response -- broadening the search on the original question.)",
}

# The Reasoner's decision (which of the three paths it chose) is advisory --
# it doesn't gate our control flow (the Executor decides search-vs-answer
# every turn regardless, matching the paper), so this is only ever used for
# trace/debug logging and to phrase the last-turn nudge. Recovered by simple
# keyword signals rather than a structured field, since the paper's Reasoner
# prompt is free text with no forced schema; default to "proceed" whenever
# intent is ambiguous.
_CONCLUDE_SIGNAL_RE = re.compile(r"\bconclude\b|\bresearch is complete\b|\bsufficient information\b", re.IGNORECASE)
_REFLECT_SIGNAL_RE = re.compile(r"\breflect\b|\brefine\b", re.IGNORECASE)

# The paper's Executor is stricter than plain "no forced schema" -- it names
# an exact ceiling ("up to 2 separate calls"). tool_choice="auto" can't
# enforce that structurally, so it's enforced here as a safety net.
_EXECUTOR_MAX_PARALLEL_CALLS = 2

_EXACT_ANSWER_RE = re.compile(r"Exact Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
_EXPLANATION_RE = re.compile(r"Explanation:\s*(.*?)(?=\n\s*Exact Answer:|\Z)", re.IGNORECASE | re.DOTALL)


def _parse_reasoner_response(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return dict(_DEFAULT_DECISION)
    if _CONCLUDE_SIGNAL_RE.search(text):
        return {"decision": "conclude", "analysis": text}
    if _REFLECT_SIGNAL_RE.search(text):
        return {"decision": "reflect_refine", "analysis": text}
    return {"decision": "proceed", "analysis": text}


def _ensure_final_answer_format(text: str) -> str:
    """The Executor is asked to end with Explanation:/Exact Answer: lines,
    but isn't forced to via schema -- wrap whatever it actually said if it
    didn't comply, so eval.py always has something to parse."""
    text = (text or "").strip()
    if _EXACT_ANSWER_RE.search(text):
        return text
    explanation = text or "insufficient evidence gathered"
    return f"Explanation: {explanation}\nExact Answer: Not found in evidence"


class InteractAgent:
    def __init__(self, search_index: SearchIndex | None = None) -> None:
        self.index = search_index or SearchIndex()
        self.engine = CorpusInteractionEngine(self.index)
        self.client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
            max_retries=config.request_max_retries,
        )

    # -- the three roles (plus the scratchpad-writer), each its own call/prompt --
    #
    # Each method appends one record to `transcript`: the exact messages sent
    # and the parsed response, so `--save-transcripts` reproduces precisely
    # what every role saw and said, independent of the more compact `result`
    # trace built up in `answer()`.

    def _run_planner(self, question: str, transcript: list[dict]) -> str:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        response = self.client.chat.completions.create(
            model=config.planner_model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        plan = (response.choices[0].message.content or "").strip()
        transcript.append({
            "role_call": "planner", "turn": 0, "model": config.planner_model,
            "messages": messages, "response": plan,
        })
        return plan

    def _run_reasoner(
        self, question: str, plan: str, scratchpad_log: list[dict], force_answer: bool,
        turn: int, transcript: list[dict],
    ) -> dict:
        user_content = (
            f"Original question (in the source language): {question}\n\n"
            f"Global-Planner's plan:\n{plan}\n\n"
            f"Scratchpad Log:\n{_format_scratchpad(scratchpad_log)}\n"
        )
        if not self.engine.entity_graph.is_built:
            user_content += (
                "\n(No entity relationship graph is available for this corpus -- "
                "do not propose graph_search; use exact_search for "
                "entity-relationship questions instead.)\n"
            )
        if force_answer:
            user_content += (
                "\nThis is the final allowed turn -- you MUST choose path B) Conclude "
                "now, using only the evidence already gathered."
            )
        messages = [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=config.reasoner_model,
            messages=messages,
            temperature=config.temperature,
        )
        text = (response.choices[0].message.content or "").strip()
        decision = _parse_reasoner_response(text)
        transcript.append({
            "role_call": "reasoner", "turn": turn, "model": config.reasoner_model,
            "messages": messages, "response": decision,
        })
        return decision

    def _available_tools(self) -> list[dict]:
        """INTERACT_TOOLS, minus graph_search when no graph has been built
        for this corpus. Documenting "returns nothing if not built" in the
        tool description isn't enough -- a model can still reach for it
        preferentially (it's the one action described as handling multi-hop
        in a single call) and waste a turn on a call that's guaranteed
        empty. Making it physically uncallable removes that failure mode
        instead of relying on the model to notice and self-correct.
        """
        if self.engine.entity_graph.is_built:
            return INTERACT_TOOLS
        return [t for t in INTERACT_TOOLS if t["function"]["name"] != "graph_search"]

    def _run_executor(
        self, question: str, analysis: str, force_answer: bool, turn: int, transcript: list[dict],
    ) -> dict:
        """Runs the Executor -- which, per the paper, decides for itself each
        turn whether to search or to conclude with the final answer. Returns
        {"tool_calls": [...], "final_text": str | None} -- exactly one of the
        two is populated. On the last allowed turn, `tool_choice` is forced
        to "none" so a tool call is impossible and `final_text` is
        guaranteed to be set, the same way `agent.py` guarantees termination.
        """
        user_content = (
            f"Original question (in the source language): {question}\n\n"
            f"Adaptive-Reasoner's analysis of prior search results:\n{analysis}"
        )
        if not self.engine.entity_graph.is_built:
            user_content += (
                "\n\n(No entity relationship graph is available for this corpus -- "
                "graph_search is not offered as an option this run.)"
            )
        if force_answer:
            user_content += (
                "\n\nThis is the final allowed turn -- you cannot search anymore. "
                "Formulate the final answer now, using only the evidence already gathered."
            )
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=config.executor_model,
            messages=messages,
            tools=self._available_tools(),
            tool_choice="none" if force_answer else "auto",
            temperature=config.temperature,
        )
        message = response.choices[0].message
        calls = []
        final_text = None
        if message.tool_calls:
            if len(message.tool_calls) > _EXECUTOR_MAX_PARALLEL_CALLS:
                _debug(f"executor proposed {len(message.tool_calls)} calls, capping at {_EXECUTOR_MAX_PARALLEL_CALLS}")
            for tool_call in message.tool_calls[:_EXECUTOR_MAX_PARALLEL_CALLS]:
                args = json.loads(tool_call.function.arguments or "{}")
                calls.append((tool_call.function.name, args))
        else:
            final_text = (message.content or "").strip()
        transcript.append({
            "role_call": "executor", "turn": turn, "model": config.executor_model,
            "messages": messages,
            "response": {"tool_calls": [{"name": n, "arguments": a} for n, a in calls], "final_text": final_text},
        })
        return {"tool_calls": calls, "final_text": final_text}

    def _run_scratchpad(
        self, question: str, goal: str, tool_results: list[tuple[str, dict, str]],
        turn: int, transcript: list[dict],
    ) -> str:
        """Condenses this turn's tool call(s) and their raw output into a
        short note (see `SCRATCHPAD_SYSTEM_PROMPT`) that the Reasoner reads
        on the next turn instead of the raw output. Not one of the paper's
        three roles -- see the module docstring.
        """
        calls_text = "\n\n".join(f"{name}({args}):\n{output}" for name, args, output in tool_results)
        user_content = (
            f"Original question (in the source language): {question}\n\n"
            f"This turn's goal (from the Adaptive-Reasoner's analysis):\n{goal}\n\n"
            f"Tool call(s) and their results:\n{calls_text}"
        )
        messages = [
            {"role": "system", "content": SCRATCHPAD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=config.scratchpad_model,
            messages=messages,
            temperature=config.temperature,
        )
        note = (response.choices[0].message.content or "").strip()
        if not note:
            note = (
                "Goal: (unspecified)\n"
                "Observations: (scratchpad writer returned no usable response -- "
                "see the raw tool output in the trace)\n"
                "Achieved: no -- scratchpad synthesis failed\n"
                "Next Steps: retry this sub-task or move to the next one"
            )
        transcript.append({
            "role_call": "scratchpad", "turn": turn, "model": config.scratchpad_model,
            "messages": messages, "response": note,
        })
        return note

    # -- the episode loop --------------------------------------------------

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
        self.engine.reset()
        retrieved_docids: list[list[str]] = []
        tool_call_counts: dict[str, int] = {}
        result: list[dict] = []
        scratchpad_log: list[dict] = []
        transcript: list[dict] = []
        final_answer = ""
        error: str | None = None

        _debug(f"=== query {query_id} [{language}]: {query_text!r}")

        # Everything below can raise (timeout, connection drop, 5xx, ...) on
        # any of the three roles' calls, on any turn. Catching here rather
        # than letting it propagate means a failure on turn 15 doesn't
        # discard the 14 turns of plan/analysis/search results already
        # gathered -- whatever's in `result`/`transcript` so far is still
        # returned, just tagged with `error` instead of ending in a final
        # answer.
        try:
            plan = self._run_planner(query_text, transcript)
            _debug(f"planner plan: {plan[:500]}{'...' if len(plan) > 500 else ''}")
            result.append({
                "type": "reasoning", "tool_name": None, "arguments": None,
                "output": plan, "role": "planner", "model": config.planner_model,
            })

            for turn in range(config.max_turns):
                last_turn = turn == config.max_turns - 1
                _debug(f"--- turn {turn + 1}/{config.max_turns}{' (final, must answer)' if last_turn else ''}")

                # Advisory only -- the Reasoner never ends the episode itself
                # (matching the paper); the Executor below decides that every
                # turn, regardless of which path the Reasoner narrated here.
                decision = self._run_reasoner(
                    query_text, plan, scratchpad_log, force_answer=last_turn,
                    turn=turn + 1, transcript=transcript,
                )
                _debug(f"reasoner decision: {decision['decision']} - {decision['analysis'][:300]}")
                result.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": decision["analysis"],
                    "role": "reasoner",
                    "model": config.reasoner_model,
                    "decision": decision,
                })

                exec_result = self._run_executor(
                    query_text, decision["analysis"], force_answer=last_turn,
                    turn=turn + 1, transcript=transcript,
                )

                if exec_result["final_text"] is not None:
                    final_answer = _ensure_final_answer_format(exec_result["final_text"])
                    _debug(f"final answer: {final_answer!r}")
                    result.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": final_answer,
                        "role": "executor",
                        "model": config.executor_model,
                    })
                    break

                tool_calls = exec_result["tool_calls"]
                if not tool_calls:
                    _debug("executor returned no tool calls and no final answer")
                    result.append({
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": "(Executor neither searched nor answered this turn)",
                        "role": "executor",
                        "model": config.executor_model,
                    })
                    continue

                turn_tool_results: list[tuple[str, dict, str]] = []
                for name, args in tool_calls:
                    tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                    tool_output, docids = _dispatch(self.engine, name, args)
                    if name in RETRIEVAL_ACTIONS:
                        retrieved_docids.append(docids)
                    turn_tool_results.append((name, args, tool_output))
                    _debug(f"executor {name}({args}) -> {tool_output[:300]}{'...' if len(tool_output) > 300 else ''}")
                    result.append({
                        "type": "tool_call",
                        "tool_name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                        "output": tool_output,
                        "role": "executor",
                        "model": config.executor_model,
                    })

                if last_turn:
                    # tool_choice="none" *should* have made this impossible, but
                    # don't trust every OpenAI-compatible server to honor it --
                    # there's no next turn to get a real answer on, so fall back
                    # rather than end with no output_text at all.
                    _debug("executor searched anyway on the forced final turn -- falling back")
                    final_answer = _ensure_final_answer_format("")
                    result.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": final_answer,
                        "role": "executor",
                        "model": config.executor_model,
                    })
                else:
                    # Condense this turn's tool output into a scratchpad note
                    # for the Reasoner to read next turn, instead of carrying
                    # the raw output forward (see the module docstring).
                    note = self._run_scratchpad(
                        query_text, decision["analysis"], turn_tool_results,
                        turn=turn + 1, transcript=transcript,
                    )
                    scratchpad_log.append({"turn": turn + 1, "note": note})
                    _debug(f"scratchpad note: {note[:300]}{'...' if len(note) > 300 else ''}")
                    result.append({
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": note,
                        "role": "executor",
                        "model": config.scratchpad_model,
                    })
        except Exception as exc:  # noqa: BLE001 -- save the partial trace rather than losing it
            error = repr(exc)
            _debug(f"answer() failed mid-run: {error}")
            result.append({
                "type": "error",
                "tool_name": None,
                "arguments": None,
                "output": f"Agent failed mid-run: {error}",
            })

        return AgentResult(
            query_id=query_id,
            language=language,
            retrieved_docids=retrieved_docids,
            tool_call_counts=tool_call_counts,
            result=result,
            transcript=transcript,
            error=error,
        )
