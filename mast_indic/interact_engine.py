"""Corpus Interaction Engine -- fine-grained retrieval action primitives.

Inspired by Interact-RAG (arXiv:2510.27566), which reframes retrieval from a
single black-box "search" call into an explicit action space an LLM agent
can compose turn by turn:

- Multi-faceted retrieval: dense `semantic_search`, sparse `exact_search`,
  and `weighted_fusion` of the two.
- Anchored matching: `entity_match` (literal mentions in chunk text) and
  `graph_search` (multi-hop traversal of a pre-built entity relationship
  graph -- see `graph_builder.py`/`entity_graph.py`).
- Context shaping: `include_docs` / `exclude_docs` pin or filter specific
  documents across subsequent retrievals in the same query, and
  `adjust_scale` changes how many chunks come back.

This module re-implements only that *interaction interface* on top of the
existing flat-numpy `SearchIndex` -- it does not reproduce the paper's
training pipeline (synthetic trajectory generation, SFT, GRPO). See
`mast_indic/interact_agent.py` for the zero-shot, prompted agent that drives
these actions instead of a fine-tuned policy.

`exact_search` is a simple case-insensitive keyword-count scorer, not real
BM25 -- consistent with this project's existing "flat matrix, brute-force
scan" scale (see `index.py`); swap in `rank_bm25` or an inverted index if
you outgrow it.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .entity_graph import EntityGraph
from .index import SearchHit, SearchIndex, _normalize, embed_texts


@dataclass
class InteractionState:
    """Context-shaping state that persists across actions within one query."""

    included_docids: set[str] = field(default_factory=set)
    excluded_docids: set[str] = field(default_factory=set)
    scale: int = 5


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class CorpusInteractionEngine:
    """Stateful wrapper around `SearchIndex` exposing Interact-RAG-style actions.

    One instance is meant to be reused across queries; call `reset()` between
    queries so pinned/excluded docs and result scale don't leak from one
    question to the next.
    """

    def __init__(self, search_index: SearchIndex, entity_graph: EntityGraph | None = None) -> None:
        self.index = search_index
        self.state = InteractionState()
        self._chunks_by_doc: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(self.index.meta):
            self._chunks_by_doc[m["docid"]].append(i)
        # Lazily loads index_store/relations.jsonl if present; graph_search
        # simply returns nothing if graph_builder.py hasn't been run yet.
        self.entity_graph = entity_graph if entity_graph is not None else EntityGraph()

    def reset(self) -> None:
        self.state = InteractionState()

    # -- scoring ---------------------------------------------------------

    def _dense_scores(self, query: str) -> np.ndarray:
        qvec = _normalize(embed_texts([query]))[0]
        return self.index.matrix @ qvec  # cosine sim, both sides normalized

    def _sparse_scores(self, keywords: str) -> np.ndarray:
        """Rank by number of *distinct* query terms matched, not raw frequency.

        A plain sum of substring counts lets one repeated common word win
        outright -- e.g. a dictionary page listing dozens of "bank ___"
        compound words would outscore a chunk that actually mentions
        "bank", "CEO", "minister", and "civil servant" once each. Distinct
        coverage dominates the score here; total occurrence count only
        breaks ties among chunks that cover the same number of terms.
        """
        unique_terms = set(_tokenize(keywords))
        scores = np.zeros(len(self.index.meta), dtype=np.float32)
        if not unique_terms:
            return scores
        for i, m in enumerate(self.index.meta):
            text_lower = m["text"].lower()
            matched_terms = 0
            total_count = 0
            for t in unique_terms:
                c = text_lower.count(t)
                if c:
                    matched_terms += 1
                    total_count += c
            if matched_terms:
                scores[i] = matched_terms + min(total_count, 20) / 1000.0
        peak = scores.max()
        return scores / peak if peak > 0 else scores  # rescale to ~[0, 1] alongside cosine sim

    # -- ranking / context shaping ----------------------------------------

    def _rank(self, scores: np.ndarray) -> list[SearchHit]:
        top_k = self.state.scale

        best_per_doc: dict[str, tuple[float, dict]] = {}
        for idx in np.argsort(-scores):
            m = self.index.meta[idx]
            docid = m["docid"]
            if docid in self.state.excluded_docids:
                continue
            score = float(scores[idx])
            if docid not in best_per_doc or score > best_per_doc[docid][0]:
                best_per_doc[docid] = (score, m)

        # include_docs guarantees a pinned doc appears even if it didn't
        # naturally score for *this* action -- look up its own best chunk.
        for docid in self.state.included_docids:
            if docid in best_per_doc or docid in self.state.excluded_docids:
                continue
            idxs = self._chunks_by_doc.get(docid)
            if not idxs:
                continue
            best_idx = max(idxs, key=lambda i: scores[i])
            best_per_doc[docid] = (float(scores[best_idx]), self.index.meta[best_idx])

        ranked_docids = sorted(best_per_doc, key=lambda d: -best_per_doc[d][0])
        pinned = [d for d in ranked_docids if d in self.state.included_docids]
        rest = [d for d in ranked_docids if d not in self.state.included_docids]

        hits = []
        for docid in (pinned + rest)[:top_k]:
            score, m = best_per_doc[docid]
            words = m["text"].split()
            hits.append(SearchHit(docid=docid, url=m["url"], score=score, snippet=" ".join(words[:400])))
        return hits

    # -- action primitives -------------------------------------------------

    def semantic_search(self, query: str) -> list[SearchHit]:
        """Dense retrieval: embedding similarity to the query."""
        return self._rank(self._dense_scores(query))

    def exact_search(self, keywords: str) -> list[SearchHit]:
        """Sparse retrieval: exact keyword occurrence ranking."""
        return self._rank(self._sparse_scores(keywords))

    def weighted_fusion(self, query: str, w_semantic: float, w_exact: float) -> list[SearchHit]:
        """Blend dense and sparse scores for `query` with the given weights."""
        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)
        return self._rank(w_semantic * dense + w_exact * sparse)

    def entity_match(self, entity: str) -> list[SearchHit]:
        """Retrieve chunks strongly (literally) associated with a named entity."""
        entity_lower = entity.lower().strip()
        if not entity_lower:
            return []
        mention_counts = np.array(
            [m["text"].lower().count(entity_lower) for m in self.index.meta], dtype=np.float32
        )
        # literal mentions dominate the ranking; embedding similarity only
        # breaks ties among chunks that mention the entity equally often.
        scores = mention_counts * 1000.0 + self._dense_scores(entity)
        return self._rank(scores)

    def graph_search(self, entity: str, hops: int = 1) -> list[SearchHit]:
        """Traverse the entity relationship graph outward from `entity`.

        Returns nothing if `graph_builder.py` hasn't been run for this
        corpus. Results are deduped per source document (richest-in-relations
        document first, up to `adjust_scale`'s current limit); the snippet is
        the extracted relation(s) themselves rather than raw chunk text, so
        the Executor sees *why* a document matched, not just that it did.
        """
        if not self.entity_graph.is_built:
            return []
        edges = self.entity_graph.neighbors(entity, hops=hops)

        per_doc: dict[str, list[dict]] = defaultdict(list)
        for edge in edges:
            if edge["docid"] in self.state.excluded_docids:
                continue
            per_doc[edge["docid"]].append(edge)

        ranked_docids = sorted(per_doc, key=lambda d: -len(per_doc[d]))[: self.state.scale]

        hits = []
        for docid in ranked_docids:
            doc_edges = per_doc[docid]
            relation_desc = "; ".join(
                f'{e["subject"]} {e["relation"]} {e["object"]}' for e in doc_edges[:5]
            )
            idxs = self._chunks_by_doc.get(docid, [])
            chunk_id = doc_edges[0]["chunk_id"]
            idx = next((i for i in idxs if self.index.meta[i]["chunk_id"] == chunk_id), idxs[0] if idxs else None)
            url = self.index.meta[idx]["url"] if idx is not None else ""
            hits.append(SearchHit(docid=docid, url=url, score=float(len(doc_edges)), snippet=relation_desc))
        return hits

    def include_docs(self, doc_ids: Iterable[str]) -> str:
        doc_ids = list(doc_ids)
        self.state.included_docids.update(doc_ids)
        self.state.excluded_docids.difference_update(doc_ids)
        return f"Pinned {len(doc_ids)} doc(s) for inclusion in subsequent retrievals: {doc_ids}"

    def exclude_docs(self, doc_ids: Iterable[str]) -> str:
        doc_ids = list(doc_ids)
        self.state.excluded_docids.update(doc_ids)
        self.state.included_docids.difference_update(doc_ids)
        return f"Excluded {len(doc_ids)} doc(s) from subsequent retrievals: {doc_ids}"

    def adjust_scale(self, n: int) -> str:
        n = max(1, min(int(n), 50))
        self.state.scale = n
        return f"Result scale set to {n} chunk(s) per subsequent retrieval."
