"""A minimal tool-calling agentic search loop for MAST Track 2 (Indic).

Flow: an Indic-language query comes in -> the LLM (any OpenAI-compatible
endpoint) issues English `search` tool calls against the local embedding
index -> once it has enough evidence it answers in English. Each search
round's docids and tool-call counts are tracked for the submission JSONL,
in the BrowseComp-Plus record shape (reasoning / tool_call / output_text
steps under `result`).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from openai import OpenAI

from .config import config
from .index import SearchIndex


def _debug(msg: str) -> None:
    if config.debug:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)

SYSTEM_PROMPT = """You are a multilingual research agent for the MAST @ FIRE 2026 \
benchmark (Track 2: Indic). You will receive a complex question written in an \
Indian language. The evidence corpus is entirely in English, and your final \
answer must also be in English.

Work iteratively:
1. Understand the question (translate it mentally if needed).
2. Call the `search` tool with focused English queries to find evidence. You may \
   call it multiple times, refining your query as you learn more (e.g. once you \
   learn a name/date/place from one search, use it to search further).
3. Only rely on facts you found via `search` results -- do not use outside \
   knowledge to fabricate specifics (names, dates, numbers).
4. When you have enough evidence, stop calling tools and give your final answer.

Final answer format -- respond with exactly two lines:
Explanation: <one or two sentences citing what you found>
Exact Answer: <the short factual answer alone, e.g. "Marie Curie" or "1912">
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the English evidence corpus and return the top matching "
            "documents (docid, url, score, and a text snippet)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "An English search query (keywords or a natural-language question).",
                }
            },
            "required": ["query"],
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


class SearchAgent:
    def __init__(self, search_index: SearchIndex | None = None) -> None:
        self.index = search_index or SearchIndex()
        self.client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=config.request_timeout,
            max_retries=config.request_max_retries,
        )

    def _run_search_tool(self, query: str) -> tuple[str, list[str]]:
        hits = self.index.search(query, top_k=config.top_k, snippet_max_words=config.snippet_max_tokens)
        docids = [h.docid for h in hits]
        payload = [
            {"docid": h.docid, "url": h.url, "score": round(h.score, 4), "snippet": h.snippet}
            for h in hits
        ]
        return json.dumps(payload, ensure_ascii=False), docids

    def answer(self, query_id: str, query_text: str, language: str = "") -> AgentResult:
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
        # any turn. Catching here rather than letting it propagate means a
        # failure on turn 8 doesn't discard the 7 turns of search results
        # already gathered -- whatever's in `result`/`messages` so far is
        # still returned, just tagged with `error` instead of ending in a
        # final answer.
        try:
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
                    search_query = args.get("query", "")
                    tool_call_counts["search"] = tool_call_counts.get("search", 0) + 1
                    tool_output, docids = self._run_search_tool(search_query)
                    retrieved_docids.append(docids)
                    _debug(f"search({search_query!r}) -> {len(docids)} hits: {docids}")
                    result.append({
                        "type": "tool_call",
                        "tool_name": "search",
                        "arguments": f"query: {json.dumps(search_query, ensure_ascii=False)}",
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
