"""Build and query a local embedding index over the BrowseComp-Plus corpus.

Embeddings come from any OpenAI-compatible /v1/embeddings server (default:
local Ollama; also works with vLLM, TEI, etc. via MAST_EMBED_BASE_URL). The
index is a flat numpy matrix of L2-normalized chunk vectors + a sidecar
JSONL of chunk metadata -- fine for a skeleton / dev-scale run. For the full
~100k-doc corpus, build once and reuse; see README for scaling notes (subset
via --limit, or swap in FAISS/pre-built BM25/Qwen3 indexes from the
BrowseComp-Plus repo for a production system).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import httpx
import numpy as np
from tqdm import tqdm

from .config import config

EMBED_PATH_NAME = "embeddings.npy"
META_PATH_NAME = "meta.jsonl"
MANIFEST_NAME = "manifest.json"
EMBED_MAX_RETRIES = 5


def iter_corpus(limit: int | None = None) -> Iterator[dict]:
    """Yield {docid, text, url} rows from the BrowseComp-Plus corpus."""
    from datasets import load_dataset

    ds = load_dataset(config.corpus_dataset, split="train", streaming=limit is not None)
    for i, row in tqdm(enumerate(ds)):
        if limit is not None and i >= limit:
            break
        yield {"docid": str(row["docid"]), "text": row["text"], "url": row.get("url", "")}


MAX_CHUNK_CHARS = 4000  # backstop for whitespace-less runs (URLs, minified code, etc.)
# that would otherwise blow past the embedding model's token context limit.


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap_words, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])[:MAX_CHUNK_CHARS]
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """Batch-embed via an OpenAI-compatible /embeddings endpoint (Ollama, vLLM, TEI, ...).

    Retries on transient connection drops -- a remote-hosted server can be up
    and healthy yet still refuse an occasional connection over a long run.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    payload = {"model": model or config.embed_model, "input": texts}
    headers = {"Authorization": f"Bearer {config.embed_api_key}"}
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{config.embed_base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            break
        except httpx.TransportError as exc:
            if attempt == EMBED_MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"embed request failed ({exc!r}), retrying in {wait}s "
                  f"[{attempt + 1}/{EMBED_MAX_RETRIES}]...")
            time.sleep(wait)
        except httpx.HTTPStatusError as exc:
            print(f"embed server error {exc.response.status_code}: {exc.response.text[:2000]}")
            if attempt == EMBED_MAX_RETRIES - 1 or exc.response.status_code < 500:
                raise
            wait = 2 ** attempt
            print(f"embed request failed ({exc!r}), retrying in {wait}s "
                  f"[{attempt + 1}/{EMBED_MAX_RETRIES}]...")
            time.sleep(wait)
    data = resp.json()
    ranked = sorted(data["data"], key=lambda row: row["index"])
    vecs = np.array([row["embedding"] for row in ranked], dtype=np.float32)
    return vecs


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _atomic_save_matrix(path: str, matrix: np.ndarray) -> None:
    tmp_path = os.path.splitext(path)[0] + ".tmp.npy"
    np.save(tmp_path, matrix)
    os.replace(tmp_path, path)


def _load_checkpoint(meta_path: str, embed_path: str) -> tuple[np.ndarray | None, dict[str, set[int]]]:
    """Resume support: map each already-embedded chunk to its (docid, chunk_id)
    via meta.jsonl, keyed against however many rows embeddings.npy actually has.

    Only meta.jsonl lines covered by the saved matrix are trusted -- any lines
    written after the last successful checkpoint (i.e. the run crashed between
    a meta.jsonl append and the next matrix save) are dropped, since there's no
    embedding vector to match them to.
    """
    if not os.path.exists(embed_path) or not os.path.exists(meta_path):
        return None, {}
    matrix = np.load(embed_path)
    with open(meta_path, encoding="utf-8") as f:
        lines = f.readlines()
    trusted_lines = lines[:matrix.shape[0]]
    if len(trusted_lines) < matrix.shape[0]:
        matrix = matrix[:len(trusted_lines)]
    if len(trusted_lines) != len(lines):
        with open(meta_path, "w", encoding="utf-8") as f:
            f.writelines(trusted_lines)
    done: dict[str, set[int]] = {}
    for line in trusted_lines:
        m = json.loads(line)
        done.setdefault(m["docid"], set()).add(m["chunk_id"])
    return matrix, done


def build_index(
    limit: int | None = None,
    embed_batch_size: int = 32,
    resume: bool = True,
    checkpoint_every: int = 2000,
) -> None:
    """Chunk the corpus, embed every chunk, persist to index_dir.

    Resumes by default: chunks already present in an existing meta.jsonl,
    matched by (docid, chunk_id), are skipped rather than re-embedded, and the
    embeddings matrix is checkpointed to disk every `checkpoint_every` chunks
    -- so an interrupted run only redoes work since its last checkpoint, not
    the whole corpus. Pass resume=False to force a full rebuild from scratch.
    """
    os.makedirs(config.index_dir, exist_ok=True)
    embed_path = os.path.join(config.index_dir, EMBED_PATH_NAME)
    meta_path = os.path.join(config.index_dir, META_PATH_NAME)
    manifest_path = os.path.join(config.index_dir, MANIFEST_NAME)

    existing_matrix, done = _load_checkpoint(meta_path, embed_path) if resume else (None, {})
    if done:
        print(f"Resuming: {sum(len(v) for v in done.values())} chunks already indexed "
              f"across {len(done)} docs.")

    all_vecs: list[np.ndarray] = [existing_matrix] if existing_matrix is not None else []
    pending_texts: list[str] = []
    pending_meta: list[dict] = []
    chunks_since_checkpoint = 0

    def checkpoint() -> np.ndarray | None:
        nonlocal chunks_since_checkpoint
        if not all_vecs:
            return None
        matrix = _normalize(np.concatenate(all_vecs, axis=0))
        _atomic_save_matrix(embed_path, matrix)
        all_vecs[:] = [matrix]
        chunks_since_checkpoint = 0
        return matrix

    with open(meta_path, "a" if done else "w", encoding="utf-8") as meta_f:
        def flush():
            nonlocal chunks_since_checkpoint
            if not pending_texts:
                return
            vecs = embed_texts(pending_texts)
            all_vecs.append(vecs)
            for m in pending_meta:
                meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
            meta_f.flush()
            chunks_since_checkpoint += len(pending_meta)
            pending_texts.clear()
            pending_meta.clear()
            if chunks_since_checkpoint >= checkpoint_every:
                checkpoint()

        for doc in tqdm(iter_corpus(limit=limit), desc="chunking+embedding"):
            already_done = done.get(doc["docid"])
            chunks = chunk_text(doc["text"], config.chunk_words, config.chunk_overlap_words)
            for ci, chunk in enumerate(chunks):
                if already_done and ci in already_done:
                    continue
                pending_texts.append(chunk)
                pending_meta.append({
                    "docid": doc["docid"],
                    "chunk_id": ci,
                    "url": doc["url"],
                    "text": chunk,
                })
                if len(pending_texts) >= embed_batch_size:
                    flush()
        flush()

    matrix = checkpoint()
    if matrix is None:
        raise RuntimeError("No chunks embedded -- is the corpus/limit empty, or already fully indexed?")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": config.embed_model,
            "chunk_words": config.chunk_words,
            "chunk_overlap_words": config.chunk_overlap_words,
            "num_chunks": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "corpus_dataset": config.corpus_dataset,
            "limit": limit,
        }, f, indent=2)

    print(f"Indexed {matrix.shape[0]} chunks ({matrix.shape[1]}-dim) -> {config.index_dir}")


@dataclass
class SearchHit:
    docid: str
    url: str
    score: float
    snippet: str


class SearchIndex:
    """Loads the on-disk index once and serves cosine-similarity search."""

    def __init__(self) -> None:
        embed_path = os.path.join(config.index_dir, EMBED_PATH_NAME)
        meta_path = os.path.join(config.index_dir, META_PATH_NAME)
        if not os.path.exists(embed_path):
            raise FileNotFoundError(
                f"No index found at {config.index_dir}. Run `python -m mast_indic.index build` first."
            )
        self.matrix = np.load(embed_path)
        self.meta: list[dict] = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        assert len(self.meta) == self.matrix.shape[0], "index/meta size mismatch"

    def search(self, query: str, top_k: int = 5, snippet_max_words: int = 400) -> list[SearchHit]:
        qvec = _normalize(embed_texts([query]))[0]
        scores = self.matrix @ qvec  # cosine sim, both sides normalized

        # rank chunks best-first, keep the best-scoring chunk per docid
        order = np.argsort(-scores)
        best_per_doc: dict[str, tuple[float, dict]] = {}
        for idx in order:
            m = self.meta[idx]
            docid = m["docid"]
            score = float(scores[idx])
            if docid not in best_per_doc or score > best_per_doc[docid][0]:
                best_per_doc[docid] = (score, m)

        ranked = sorted(best_per_doc.items(), key=lambda kv: -kv[1][0])[:top_k]
        hits = []
        for docid, (score, m) in ranked:
            words = m["text"].split()
            snippet = " ".join(words[:snippet_max_words])
            hits.append(SearchHit(docid=docid, url=m["url"], score=score, snippet=snippet))
        return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--limit", type=int, default=None, help="only index first N docs (dev/testing)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fresh", action="store_true",
                         help="ignore any existing index and rebuild from scratch")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                         help="save the embeddings matrix to disk every N newly-embedded chunks")
    args = parser.parse_args()

    if args.action == "build":
        build_index(
            limit=args.limit,
            embed_batch_size=args.batch_size,
            resume=not args.fresh,
            checkpoint_every=args.checkpoint_every,
        )
