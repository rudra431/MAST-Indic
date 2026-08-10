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

The one deliberate deviation from the paper: its Executor's final answer is
just "concise and direct words," but ours must end with fixed
`Explanation:`/`Exact Answer:` lines, since `mast_indic/eval.py` depends on
that format to score answers. Termination is still guaranteed the same way
`agent.py` guarantees it: on the last allowed turn, the Executor's
`tool_choice` is forced to `"none"`, so it cannot call a tool and must
answer in text.

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
   the retrieval modes actually behave (dense/semantic_search embeds the whole query \
   into one vector, so a long sentence chaining unrelated constraints together dilutes \
   into a vague, often-wrong match; sparse/exact_search and entity_match are literal \
   keyword matching, good for a name/number/exact phrase but easily dominated by an \
   unrelated document that happens to repeat one common word many times). If a \
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
- exact_search(keywords): sparse exact-keyword retrieval -- good for names, \
  numbers, and exact phrases that embeddings tend to blur together.
- weighted_fusion(query, w_semantic, w_exact): blend dense and sparse \
  scoring for one query when neither alone is enough (weights need not sum \
  to 1).
- entity_match(entity): retrieve chunks that literally mention a specific \
  named entity -- use when you need everything about "that person/place/org."
- graph_search(entity, hops): traverse a pre-built entity relationship graph \
  outward from a named entity (1-3 hops) -- use for explicit relationship \
  questions ("who founded X", "what else did X found", "who else is \
  connected to X") once you have a specific entity name. Returns nothing if \
  the graph hasn't been built for this corpus; fall back to entity_match or \
  exact_search if so.
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
  (roughly 5-15 words). Never concatenate multiple unrelated constraints \
  (dates, alumni types, distances, etc.) into one query -- that dilutes the \
  embedding and returns vaguely-related noise instead of a precise match.
- exact_search(keywords) / entity_match(entity): a short exact phrase, \
  proper noun, number, or date you expect to appear verbatim in the corpus. \
  Avoid long descriptive phrases here -- a common word repeated many times \
  in an unrelated document can outrank a genuinely relevant chunk that only \
  mentions it once.
- weighted_fusion(query, w_semantic, w_exact): use for ONE reasonably \
  specific query when you want to hedge between its literal and conceptual \
  interpretation -- not as a way to combine several different facts into a \
  single call.

If the Reasoner's analysis bundles multiple facts together into one proposed \
search, pick the single most distinctive one to search for now and leave the \
rest for a later turn -- unless it explicitly proposed two independent \
parallel searches, in which case call both.
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
            "name": "graph_search",
            "description": "Traverse a pre-built entity relationship graph outward from a named entity, returning nearby entities and how they're connected. Returns nothing if the graph hasn't been built for this corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The entity name to start from, in English."},
                    "hops": {"type": "integer", "description": "How many relationship hops to traverse outward (1-3), default 1."},
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
RETRIEVAL_ACTIONS = {"semantic_search", "exact_search", "weighted_fusion", "entity_match", "graph_search"}


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
    elif name == "graph_search":
        hits = engine.graph_search(args.get("entity", ""), int(args.get("hops", 1) or 1))
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

    # -- the three roles, each its own call/prompt -------------------------
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
        )
        plan = (response.choices[0].message.content or "").strip()
        transcript.append({
            "role_call": "planner", "turn": 0, "model": config.planner_model,
            "messages": messages, "response": plan,
        })
        return plan

    def _run_reasoner(
        self, question: str, plan: str, evidence_log: list[dict], force_answer: bool,
        turn: int, transcript: list[dict],
    ) -> dict:
        user_content = (
            f"Original question (in the source language): {question}\n\n"
            f"Global-Planner's plan:\n{plan}\n\n"
            f"Evidence gathered so far:\n{_format_evidence(evidence_log)}\n"
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
            tools=INTERACT_TOOLS,
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

    # -- the episode loop --------------------------------------------------

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
        self.engine.reset()
        retrieved_docids: list[list[str]] = []
        tool_call_counts: dict[str, int] = {}
        result: list[dict] = []
        evidence_log: list[dict] = []
        transcript: list[dict] = []
        final_answer = ""

        _debug(f"=== query {query_id} [{language}]: {query_text!r}")

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
                query_text, plan, evidence_log, force_answer=last_turn,
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

        return AgentResult(
            query_id=query_id,
            language=language,
            retrieved_docids=retrieved_docids,
            tool_call_counts=tool_call_counts,
            result=result,
            transcript=transcript,
        )
