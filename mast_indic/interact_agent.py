"""A zero-shot Interact-RAG-style agent (arXiv:2510.27566) for Track 2 (Indic).

Where `agent.py` gives the LLM one black-box `search` tool, this agent
exposes the paper's fine-grained Corpus Interaction Engine actions
(`mast_indic/interact_engine.py`): dense/sparse/fused retrieval,
entity-anchored matching, and context shaping (pin/filter documents, resize
result sets). The system prompt below asks the model to reproduce the
paper's Global-Planner / Adaptive-Reasoner / Executor workflow zero-shot,
inside a single conversation, rather than via the paper's trained
SFT+RL policy -- there's no training pipeline here, just the prompted
interaction interface plus this project's existing tool-calling loop.

Output record shape matches `agent.py`/`runner.py` exactly (reusing
`AgentResult`), so runs from this agent drop straight into
`mast_indic/eval.py` unchanged.
"""
from __future__ import annotations

import json
import sys

from openai import OpenAI

from .agent import AgentResult
from .config import config
from .index import SearchIndex
from .interact_engine import CorpusInteractionEngine


def _debug(msg: str) -> None:
    if config.debug:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


SYSTEM_PROMPT = """You are a multilingual research agent for the MAST @ FIRE 2026 \
benchmark (Track 2: Indic), using an Interact-RAG-style corpus interaction \
interface instead of a single black-box search tool. You will receive a \
complex question written in an Indian language. The evidence corpus is \
entirely in English, and your final answer must also be in English.

You act out three roles internally, in order, at every step:

1. Global-Planner (first turn only): read the question and write a short, \
   numbered plan of the sub-questions you need evidence for.
2. Adaptive-Reasoner (every turn): look at the evidence gathered so far and \
   decide one of:
   - Proceed: the current sub-task has enough evidence -- move to the next \
     one, or answer if all sub-questions are resolved.
   - Reflect & Refine: evidence is missing, contradictory, or the last \
     action(s) came back empty/irrelevant. Diagnose why and change strategy \
     (a different action, different keywords, a narrower/wider scale, \
     excluding a noisy document, etc.) rather than repeating the same call.
   State which of the two applies in one short line before acting.
3. Executor: call one or more of the tools below to carry out that \
   decision. You may call several tools in the same turn when they serve \
   independent sub-questions.

Available actions (all operate over the same evidence corpus):
- semantic_search(query): dense embedding retrieval -- good for \
  paraphrased/conceptual matches.
- exact_search(keywords): sparse exact-keyword retrieval -- good for names, \
  numbers, and exact phrases that embeddings tend to blur together.
- weighted_fusion(query, w_semantic, w_exact): blend dense and sparse \
  scoring for one query when neither alone is enough (weights need not sum \
  to 1).
- entity_match(entity): retrieve chunks that literally mention a specific \
  named entity -- use when you need everything about "that person/place/org."
- include_docs(doc_ids): pin specific document IDs so they're guaranteed to \
  appear in later retrievals (e.g. a document you've already confirmed is \
  relevant).
- exclude_docs(doc_ids): filter out document IDs you've confirmed are \
  irrelevant or noisy, so later retrievals stop surfacing them.
- adjust_scale(n): change how many chunks come back per retrieval (smaller \
  once you've narrowed in for precision, larger while still exploring).

Only rely on facts you found via these actions -- do not use outside \
knowledge to fabricate specifics (names, dates, numbers). When you have \
enough evidence, stop calling tools and give your final answer.

Final answer format -- respond with exactly two lines:
Explanation: <one or two sentences citing what you found>
Exact Answer: <the short factual answer alone, e.g. "Marie Curie" or "1912">
"""

INTERACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Dense embedding retrieval over the evidence corpus -- good for conceptual/paraphrased matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "An English natural-language search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exact_search",
            "description": "Sparse exact-keyword retrieval -- good for names, numbers, and exact phrases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Space-separated keywords or a phrase to match literally."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weighted_fusion",
            "description": "Blend dense and sparse retrieval for one query using the given weights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "An English natural-language search query."},
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
            "name": "entity_match",
            "description": "Retrieve chunks that literally mention a specific named entity (person, place, organization, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The entity name to search for, in English."},
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "include_docs",
            "description": "Pin specific document IDs so they are guaranteed to appear in subsequent retrievals.",
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
            "description": "Filter out specific document IDs from subsequent retrievals.",
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
            "description": "Change how many chunks are returned per subsequent retrieval.",
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
RETRIEVAL_ACTIONS = {"semantic_search", "exact_search", "weighted_fusion", "entity_match"}


def _dispatch(engine: CorpusInteractionEngine, name: str, args: dict) -> tuple[str, list[str]]:
    """Run one action against the engine; returns (tool_output_text, docids)."""
    if name == "semantic_search":
        hits = engine.semantic_search(args.get("query", ""))
    elif name == "exact_search":
        hits = engine.exact_search(args.get("keywords", ""))
    elif name == "weighted_fusion":
        hits = engine.weighted_fusion(
            args.get("query", ""),
            float(args.get("w_semantic", 0.5)),
            float(args.get("w_exact", 0.5)),
        )
    elif name == "entity_match":
        hits = engine.entity_match(args.get("entity", ""))
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

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
        self.engine.reset()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ]
        retrieved_docids: list[list[str]] = []
        tool_call_counts: dict[str, int] = {}
        result: list[dict] = []
        final_answer = ""

        _debug(f"=== query {query_id} [{language}]: {query_text!r}")

        for turn in range(config.max_turns):
            last_turn = turn == config.max_turns - 1
            _debug(f"--- turn {turn + 1}/{config.max_turns}{' (final, tools disabled)' if last_turn else ''}")
            if last_turn:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have used all available turns and cannot call any "
                        "more tools. Based only on the evidence already "
                        "gathered, respond now in exactly this format:\n"
                        "Explanation: <one or two sentences citing what you found>\n"
                        "Exact Answer: <the short factual answer alone>"
                    ),
                })
            response = self.client.chat.completions.create(
                model=config.chat_model,
                messages=messages,
                tools=INTERACT_TOOLS,
                tool_choice="none" if last_turn else "auto",
                temperature=config.temperature,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            if reasoning:
                _debug(f"reasoning: {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}")
                result.append({"type": "reasoning", "tool_name": None, "arguments": None, "output": reasoning})
            elif message.content and message.tool_calls:
                # Planner/Reasoner narration often lands in `content` alongside
                # tool calls rather than in a separate reasoning field.
                _debug(f"reasoning: {message.content[:500]}{'...' if len(message.content) > 500 else ''}")
                result.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": message.content.strip(),
                })

            if not message.tool_calls:
                final_answer = (message.content or "").strip()
                _debug(f"final answer: {final_answer!r}")
                result.append({"type": "output_text", "tool_name": None, "arguments": None, "output": final_answer})
                break

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                tool_output, docids = _dispatch(self.engine, name, args)
                if name in RETRIEVAL_ACTIONS:
                    retrieved_docids.append(docids)
                _debug(f"{name}({args}) -> {tool_output[:300]}{'...' if len(tool_output) > 300 else ''}")
                result.append({
                    "type": "tool_call",
                    "tool_name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                    "output": tool_output,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                })

        return AgentResult(
            query_id=query_id,
            language=language,
            retrieved_docids=retrieved_docids,
            tool_call_counts=tool_call_counts,
            result=result,
            transcript=messages,
        )
