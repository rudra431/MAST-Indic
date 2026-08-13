"""A minimal tool-calling agentic search loop using ONLY exact/sparse
keyword retrieval -- no dense embedding search, no other tools.

Same single-tool loop as `agent.py` (mirror it if you're comparing the
two), but the one `search` tool is backed by
`CorpusInteractionEngine.exact_search` (distinct-term-coverage keyword
matching, see `interact_engine.py`) instead of `SearchIndex.search`'s cosine
similarity. Useful as a sparse-only baseline alongside `agent.py`
(dense-only) and `interact_agent.py` (both, plus more).

One addition beyond `agent.py`: a planning step (mirroring
`interact_agent.py`'s Global-Planner, and sharing its `MAST_PLANNER_MODEL`
config) runs once per query before the search loop, decomposing the
question into a list of ATOMIC sub-queries -- short keyword-style phrases,
one per distinguishing fact, since exact_search specifically needs that
(a broader "figure out the founding date" sub-question, the kind a general
planner would produce, isn't itself a usable keyword query). The plan is
injected into the conversation as context, not as a tool the model can
call -- the search loop that follows is otherwise identical to `agent.py`'s:
one tool, one growing conversation, no forced ordering through the plan.

Since exact_search never calls the embedding server, running this agent
makes zero embedding calls -- but `index.py build` must still have been run
once, since it's what produces `index_store/meta.jsonl` (the chunk text
this searches over).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from openai import OpenAI

from .config import config
from .index import SearchIndex
from .interact_engine import CorpusInteractionEngine


def _debug(msg: str) -> None:
    if config.debug:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


SYSTEM_PROMPT = """You are a multilingual research agent for the MAST @ FIRE 2026 \
benchmark (Track 2: Indic). You will receive a complex question written in an \
Indian language. The evidence corpus is entirely in English, and your final \
answer must also be in English.

Your only tool is `search`, which does EXACT sparse keyword matching -- not \
semantic/embedding search. It ranks chunks by how many of your keywords they \
literally contain, so:
- Use short, specific keywords, names, numbers, or exact phrases you expect \
  to appear verbatim in the corpus -- not a paraphrased natural-language \
  question.
- If a search returns nothing useful, don't just repeat similar wording -- try a \
  different specific term, a narrower phrase, or a name/detail you learned \
  from an earlier result.
- Long queries stuffed with many unrelated words are no better here than \
  short ones, since only literal matches count -- prefer the single most \
  distinctive term or phrase for what you're looking for right now over a \
  laundry list of every clue at once.

Before you start, a planning step will give you a list of atomic sub-queries \
this question decomposes into -- use them as your starting `search` calls, \
one atomic fact per call. You aren't required to follow that exact order, and \
you can skip a sub-query you resolve incidentally along the way, or add new \
searches the plan didn't anticipate.

Work iteratively:
1. Understand the question (translate it mentally if needed).
2. Call `search` with focused keyword queries to find evidence. You may call \
   it multiple times, refining your keywords as you learn more (e.g. once \
   you learn a name/date/place from one search, use it to search further).
3. Only rely on facts you found via `search` results -- do not use outside \
   knowledge to fabricate specifics (names, dates, numbers).
4. When you have enough evidence, stop calling tools and give your final answer.

Final answer format -- respond with exactly two lines:
Explanation: <one or two sentences citing what you found>
Exact Answer: <the short factual answer alone, e.g. "Marie Curie" or "1912">
"""

PLANNER_SYSTEM_PROMPT = """You are the planning step for a research agent that \
answers a complex question using ONLY exact/sparse keyword search over an English \
evidence corpus (the question itself may be written in an Indian language; \
translate it mentally).

Decompose the question into a short, ordered list of ATOMIC sub-queries -- each \
one a short, keyword-style phrase (not a full sentence, and not a combination of \
several facts at once) that targets ONE distinguishing fact the agent needs to \
look up, in the order they're likely needed (e.g. resolve an entity name before \
searching for its founder). Since retrieval here is exact/literal keyword \
matching, prefer specific names, numbers, dates, and short distinctive phrases \
over descriptive paraphrases -- a sub-query like "founded 1949 1959" or "Lady \
Shri Ram College" works far better than "institution founded between 1949 and \
1959 with notable alumni". Do not answer the question yourself and do not invent \
facts -- you have no access to the evidence corpus. Report your sub-queries by \
calling `submit_plan`.
"""

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Report the atomic sub-queries this question decomposes into.",
        "parameters": {
            "type": "object",
            "properties": {
                "subqueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short, keyword-style phrases, one per distinguishing fact, "
                        "in the order they're likely needed."
                    ),
                },
            },
            "required": ["subqueries"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the English evidence corpus by exact keyword matching (no semantic/embedding "
            "matching) and return the top matching documents (docid, url, score, and a text snippet)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "Space-separated keywords or an exact phrase to match literally -- "
                        "not a natural-language question paraphrase."
                    ),
                }
            },
            "required": ["keywords"],
        },
    },
}


@dataclass
class AgentResult:
    query_id: str
    language: str
    retrieved_docids: list[list[str]]
    tool_call_counts: dict[str, int]
    result: list[dict]
    transcript: list[dict] = field(default_factory=list)
    error: str | None = None


class ExactSearchAgent:
    def __init__(self, search_index: SearchIndex | None = None) -> None:
        self.index = search_index or SearchIndex()
        self.engine = CorpusInteractionEngine(self.index)
        self.engine.state.scale = config.top_k
        self.client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
            max_retries=config.request_max_retries,
        )

    def _run_planner(self, question: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=config.planner_model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=[PLAN_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_plan"}},
            temperature=config.temperature,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        try:
            args = json.loads(message.tool_calls[0].function.arguments or "{}")
        except json.JSONDecodeError:
            return []
        subqueries = args.get("subqueries") or []
        return [str(q).strip() for q in subqueries if str(q).strip()]

    def _run_search_tool(self, keywords: str) -> tuple[str, list[str]]:
        hits = self.engine.exact_search(keywords)
        docids = [h.docid for h in hits]
        payload = [
            {"docid": h.docid, "url": h.url, "score": round(h.score, 4), "snippet": h.snippet}
            for h in hits
        ]
        return json.dumps(payload, ensure_ascii=False), docids

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
        self.engine.reset()
        self.engine.state.scale = config.top_k
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ]
        retrieved_docids: list[list[str]] = []
        tool_call_counts: dict[str, int] = {"search": 0}
        result: list[dict] = []
        final_answer = ""
        error: str | None = None

        _debug(f"=== query {query_id} [{language}]: {query_text!r}")

        # Everything below can raise (timeout, connection drop, 5xx, ...) on
        # any turn -- including the planner call. Catching here rather than
        # letting it propagate means a failure mid-run doesn't discard the
        # turns of search results already gathered; whatever's in
        # `result`/`messages` so far is still returned, just tagged with
        # `error` instead of ending in a final answer.
        try:
            subqueries = self._run_planner(query_text)
            _debug(f"planner subqueries: {subqueries}")
            if subqueries:
                plan_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(subqueries))
                result.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": plan_text,
                    "role": "planner",
                    "model": config.planner_model,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "A planning step decomposed this question into the following atomic "
                        "sub-queries. Use them as your starting `search` calls, one atomic fact "
                        "per call -- you aren't required to follow this exact order, and can skip "
                        "one you resolve incidentally along the way:\n" + plan_text
                    ),
                })

            for turn in range(config.max_turns):
                last_turn = turn == config.max_turns - 1
                _debug(f"--- turn {turn + 1}/{config.max_turns}{' (final, tools disabled)' if last_turn else ''}")
                if last_turn:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have used all available search turns and cannot call "
                            "any more tools. Based only on the evidence already "
                            "gathered, respond now in exactly this format:\n"
                            "Explanation: <one or two sentences citing what you found>\n"
                            "Exact Answer: <the short factual answer alone>"
                        ),
                    })
                response = self.client.chat.completions.create(
                    model=config.chat_model,
                    messages=messages,
                    tools=[SEARCH_TOOL],
                    tool_choice="none" if last_turn else "auto",
                    temperature=config.temperature,
                )
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))

                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
                if reasoning:
                    _debug(f"reasoning: {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}")
                    result.append({
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": reasoning,
                    })

                if not message.tool_calls:
                    final_answer = (message.content or "").strip()
                    _debug(f"final answer: {final_answer!r}")
                    result.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": final_answer,
                    })
                    break

                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments or "{}")
                    keywords = args.get("keywords", "")
                    tool_call_counts["search"] = tool_call_counts.get("search", 0) + 1
                    tool_output, docids = self._run_search_tool(keywords)
                    retrieved_docids.append(docids)
                    _debug(f"search({keywords!r}) -> {len(docids)} hits: {docids}")
                    result.append({
                        "type": "tool_call",
                        "tool_name": "search",
                        "arguments": f"keywords: {json.dumps(keywords, ensure_ascii=False)}",
                        "output": tool_output,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
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
            transcript=messages,
            error=error,
        )
