"""Central config, populated from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

INDIC_LANGUAGES = ["bn", "gu", "hi", "kn", "ml", "or", "pa", "ta", "te"]


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # OpenAI-compatible chat/reasoning LLM (OpenAI, vLLM, Ollama's /v1, etc.)
    openai_base_url: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_API_KEY", "not-needed"))
    chat_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_CHAT_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: _float("MAST_TEMPERATURE", 1.0))
    # Per-request timeout to the chat LLM, and retries on transient failures
    # (connect/read timeouts, 5xx) before giving up on a single call. Keep
    # this well below your batch script's patience -- a hung/unreachable
    # endpoint should fail one query fast, not stall the whole run.
    request_timeout: float = field(default_factory=lambda: _float("MAST_REQUEST_TIMEOUT", 120.0))
    request_max_retries: int = field(default_factory=lambda: _int("MAST_REQUEST_MAX_RETRIES", 2))

    # LLM judge (mast_indic/eval.py): defaults to the chat LLM above, since a
    # single self-hosted endpoint is often all you have. Set MAST_JUDGE_* to
    # point evaluation at a separate/cheaper judge model instead.
    judge_base_url: str = field(default_factory=lambda: os.environ.get(
        "MAST_JUDGE_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    judge_api_key: str = field(default_factory=lambda: os.environ.get(
        "MAST_JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "not-needed")))
    judge_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_JUDGE_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))

    # Interact-RAG-style agent (mast_indic/interact_agent.py): one model per
    # role (Global-Planner / Adaptive-Reasoner / Executor), plus the
    # scratchpad-writer call added on top of the paper's three roles (see
    # that module's docstring) -- each defaulting to the chat LLM above; set
    # MAST_*_MODEL to split roles across models (e.g. a cheap model for
    # planning, a stronger one for the reasoner).
    planner_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_PLANNER_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))
    reasoner_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_REASONER_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))
    executor_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_EXECUTOR_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))
    scratchpad_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_SCRATCHPAD_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))

    # Entity relationship graph (mast_indic/graph_builder.py, used by
    # CorpusInteractionEngine.graph_search): LLM used to extract
    # (subject, relation, object) triples per chunk. Defaults to the chat
    # LLM above; a cheaper/faster model is usually fine for this.
    graph_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_GRAPH_MODEL", os.environ.get("MAST_CHAT_MODEL", "gpt-4o-mini")))

    # Embeddings: any OpenAI-compatible /v1/embeddings server (Ollama, vLLM, TEI, ...),
    # kept separate from the chat LLM on purpose. Defaults to local Ollama's
    # OpenAI-compat route; point at a vLLM server by setting MAST_EMBED_BASE_URL.
    embed_base_url: str = field(default_factory=lambda: os.environ.get(
        "MAST_EMBED_BASE_URL",
        os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1"))
    embed_api_key: str = field(default_factory=lambda: os.environ.get(
        "MAST_EMBED_API_KEY", "not-needed"))
    embed_model: str = field(default_factory=lambda: os.environ.get(
        "MAST_EMBED_MODEL", "embeddinggemma:300m"))

    # Corpus / index
    corpus_dataset: str = field(default_factory=lambda: os.environ.get(
        "MAST_CORPUS_DATASET", "Tevatron/browsecomp-plus-corpus"))
    queries_dataset: str = field(default_factory=lambda: os.environ.get(
        "MAST_QUERIES_DATASET", "mast-benchmark/indic-queries-2026"))
    index_dir: str = field(default_factory=lambda: os.environ.get(
        "MAST_INDEX_DIR", "index_store"))
    # Character-based chunking (index.py): sizes chunks directly against the
    # embedding model's token budget via a 1-token-~4-characters
    # approximation, rather than counting words. Defaults: 2048 tokens ~
    # 8192 chars per chunk, 50 tokens ~ 200 chars overlap.
    chunk_chars: int = field(default_factory=lambda: _int("MAST_CHUNK_CHARS", 8192))
    chunk_overlap_chars: int = field(default_factory=lambda: _int("MAST_CHUNK_OVERLAP_CHARS", 200))

    # Agent loop / search tool
    max_turns: int = field(default_factory=lambda: _int("MAST_MAX_TURNS", 8))
    top_k: int = field(default_factory=lambda: _int("MAST_TOP_K", 64))
    top_p: float = field(default_factory=lambda: _float("MAST_TOP_P", 0.95))
    snippet_max_tokens: int = field(default_factory=lambda: _int("MAST_SNIPPET_MAX_TOKENS", 512))

    # Output
    runs_dir: str = field(default_factory=lambda: os.environ.get("MAST_RUNS_DIR", "runs"))

    # Debugging: print each turn's reasoning/tool-calls/answer to stderr as they happen
    debug: bool = field(default_factory=lambda: _bool("MAST_DEBUG", False))


config = Config()
