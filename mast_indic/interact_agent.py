"""A zero-shot Interact-RAG-style agent (arXiv:2510.27566) for Track 2 (Indic).

Where `agent.py` gives the LLM one black-box `search` tool, this agent
exposes the paper's fine-grained Corpus Interaction Engine actions
(`mast_indic/interact_engine.py`): dense/sparse/fused retrieval,
entity-anchored matching, and context shaping (pin/filter documents, resize
result sets). It also reproduces the paper's *three-module* workflow as
three separate LLM calls with distinct prompts, rather than one agent
narrating all three roles in a single call:

- Global-Planner: one call per query, decomposes the question into a plan.
- Adaptive-Reasoner: one call per turn, judges the evidence gathered so far
  and either directs the Executor (proceed / reflect & refine) or ends the
  episode with a final answer. Returns a structured decision (forced
  function call) rather than free text, so the loop doesn't have to parse
  natural-language verdicts.
- Executor: one call per turn (only when the Reasoner isn't done yet),
  translates the Reasoner's instruction into one or more concrete calls to
  the interaction primitives.

Each role can optionally run on its own model (`MAST_PLANNER_MODEL` /
`MAST_REASONER_MODEL` / `MAST_EXECUTOR_MODEL` in `config.py`); all three
default to `MAST_CHAT_MODEL` if unset, so nothing changes unless you split
them out.

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
research pipeline (MAST @ FIRE 2026, Track 2: Indic). You will be given a \
complex question written in an Indian language; the evidence corpus and the \
eventual answer are in English.

Read the question and decompose it into a short, numbered plan of the \
sub-questions/facts an evidence-gathering agent will need to resolve, in the \
order they're likely needed (e.g. resolve an entity first, then a date, then \
the final fact). Do not answer the question yourself and do not invent facts \
-- you have no access to the evidence corpus. Output ONLY the numbered plan, \
nothing else.
"""

REASONER_SYSTEM_PROMPT = """You are the Adaptive-Reasoner in the same \
Interact-RAG-style pipeline. You are the cognitive core: given the original \
question, the Global-Planner's plan, and all evidence gathered so far by the \
Executor, you decide what happens next by calling `submit_decision`:

- decision="proceed": the current sub-task from the plan is progressing well \
  and there's enough evidence to move to the next sub-task. Set `strategy` \
  to a concrete instruction for the Executor -- which action to use and with \
  what query/entity/keywords.
- decision="reflect_refine": the process hit an obstacle -- the last \
  action(s) returned nothing useful, evidence is contradictory, or the \
  context is cluttered with irrelevant documents. Diagnose the obstacle in \
  `rationale`, and set `strategy` to a *different* concrete instruction (a \
  different action, different keywords, a broader/narrower scale, or \
  excluding a noisy document) -- never repeat a failing action verbatim.
- decision="ready_to_answer": every sub-question in the plan is resolved by \
  evidence that was actually retrieved. Leave `strategy` empty and instead \
  fill in `explanation` (one or two sentences citing what was found) and \
  `exact_answer` (the short factual answer alone, e.g. "Marie Curie" or \
  "1912").

Only rely on facts that appear in the evidence log below -- never fabricate \
names, dates, or numbers that aren't there.
"""

EXECUTOR_SYSTEM_PROMPT = """You are the Executor in the same Interact-RAG-style \
pipeline. You do not decide strategy -- the Adaptive-Reasoner already has. \
Your only job is to translate its instruction into one or more concrete \
calls to the interaction primitives below. Call at least one tool every \
time; call more than one in the same turn only if the instruction genuinely \
calls for independent lookups.

Available actions (all operate over the same English evidence corpus):
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

REASONER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Report the Adaptive-Reasoner's judgment of the current state and what should happen next.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["proceed", "reflect_refine", "ready_to_answer"],
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences on why the evidence is/isn't sufficient right now.",
                },
                "strategy": {
                    "type": "string",
                    "description": "If proceed/reflect_refine: a concrete instruction for the Executor. Leave empty if ready_to_answer.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Required if decision is ready_to_answer: one or two sentences citing the evidence found.",
                },
                "exact_answer": {
                    "type": "string",
                    "description": "Required if decision is ready_to_answer: the short factual final answer alone.",
                },
            },
            "required": ["decision", "rationale", "strategy"],
        },
    },
}

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


def _format_evidence(evidence_log: list[dict]) -> str:
    if not evidence_log:
        return "(no evidence gathered yet)"
    lines = []
    for entry in evidence_log:
        lines.append(f"[Turn {entry['turn']}] {entry['action']}({entry['args']}):")
        lines.append(entry["output"])
    return "\n".join(lines)


_DEFAULT_DECISION = {
    "decision": "reflect_refine",
    "rationale": "",
    "strategy": "Broaden the search on the original question.",
    "explanation": None,
    "exact_answer": None,
}


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

    # -- the three roles, each its own call/prompt -------------------------

    def _run_planner(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=config.planner_model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=config.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _run_reasoner(self, question: str, plan: str, evidence_log: list[dict], force_answer: bool) -> dict:
        user_content = (
            f"Original question (in the source language): {question}\n\n"
            f"Global-Planner's plan:\n{plan}\n\n"
            f"Evidence gathered so far:\n{_format_evidence(evidence_log)}\n"
        )
        if force_answer:
            user_content += (
                "\nThis is the final allowed turn -- you MUST set decision to "
                '"ready_to_answer" now, using only the evidence already gathered.'
            )
        response = self.client.chat.completions.create(
            model=config.reasoner_model,
            messages=[
                {"role": "system", "content": REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=[REASONER_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_decision"}},
            temperature=config.temperature,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return dict(_DEFAULT_DECISION)
        try:
            decision = json.loads(message.tool_calls[0].function.arguments or "{}")
        except json.JSONDecodeError:
            decision = dict(_DEFAULT_DECISION)
        for key, default in _DEFAULT_DECISION.items():
            decision.setdefault(key, default)
        return decision

    def _run_executor(self, strategy: str) -> list[tuple[str, dict]]:
        response = self.client.chat.completions.create(
            model=config.executor_model,
            messages=[
                {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"Adaptive-Reasoner's instruction: {strategy}"},
            ],
            tools=INTERACT_TOOLS,
            tool_choice="auto",
            temperature=config.temperature,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        calls = []
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            calls.append((tool_call.function.name, args))
        return calls

    # -- the episode loop --------------------------------------------------

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
        self.engine.reset()
        retrieved_docids: list[list[str]] = []
        tool_call_counts: dict[str, int] = {}
        result: list[dict] = []
        evidence_log: list[dict] = []
        final_answer = ""

        _debug(f"=== query {query_id} [{language}]: {query_text!r}")

        plan = self._run_planner(query_text)
        _debug(f"planner plan: {plan[:500]}{'...' if len(plan) > 500 else ''}")
        result.append({"type": "reasoning", "tool_name": None, "arguments": None, "output": plan, "role": "planner"})

        for turn in range(config.max_turns):
            last_turn = turn == config.max_turns - 1
            _debug(f"--- turn {turn + 1}/{config.max_turns}{' (final, must answer)' if last_turn else ''}")

            decision = self._run_reasoner(query_text, plan, evidence_log, force_answer=last_turn)
            _debug(f"reasoner decision: {decision['decision']} - {decision.get('rationale', '')[:300]}")
            rationale_text = f"[{decision['decision']}] {decision.get('rationale', '')}".strip()
            result.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": rationale_text,
                "role": "reasoner",
            })

            if decision["decision"] == "ready_to_answer" or last_turn:
                explanation = decision.get("explanation") or "insufficient evidence gathered"
                exact_answer = decision.get("exact_answer") or "Not found in evidence"
                final_answer = f"Explanation: {explanation}\nExact Answer: {exact_answer}"
                _debug(f"final answer: {final_answer!r}")
                result.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": final_answer,
                    "role": "reasoner",
                })
                break

            tool_calls = self._run_executor(decision["strategy"])
            if not tool_calls:
                _debug("executor returned no tool calls")
                result.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": "(Executor made no tool call this turn)",
                    "role": "executor",
                })
                continue

            for name, args in tool_calls:
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                tool_output, docids = _dispatch(self.engine, name, args)
                if name in RETRIEVAL_ACTIONS:
                    retrieved_docids.append(docids)
                evidence_log.append({"turn": turn + 1, "action": name, "args": args, "output": tool_output})
                _debug(f"executor {name}({args}) -> {tool_output[:300]}{'...' if len(tool_output) > 300 else ''}")
                result.append({
                    "type": "tool_call",
                    "tool_name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                    "output": tool_output,
                    "role": "executor",
                })

        return AgentResult(
            query_id=query_id,
            language=language,
            retrieved_docids=retrieved_docids,
            tool_call_counts=tool_call_counts,
            result=result,
            transcript=evidence_log,
        )
