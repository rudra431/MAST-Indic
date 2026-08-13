"""Corpus Interaction Engine -- fine-grained retrieval action primitives.

Inspired by Interact-RAG (arXiv:2510.27566), which reframes retrieval from a
single black-box "search" call into an explicit action space an LLM agent
can compose turn by turn:

- Multi-faceted retrieval: dense `semantic_search`, BM25-ranked sparse
  `exact_search`, exact `boolean_search` (AND/OR/NOT over terms, unranked),
  and `weighted_fusion` blending dense with BM25.
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

`exact_search`/`boolean_search` are backed by a real in-memory inverted
index (`_build_inverted_index`), built once per `CorpusInteractionEngine`
instance rather than per call -- consistent with this project's existing
"flat matrix, brute-force scan" scale (see `index.py`), but a real
improvement over a linear per-query scan: after the one-time build, both
become O(query terms) lookups instead of O(corpus size). Swap in a
persisted index (Lucene/Elasticsearch/`rank_bm25` with disk caching) if you
outgrow rebuilding it every process start.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .entity_graph import EntityGraph
from .index import SearchHit, SearchIndex, _normalize, embed_texts

# Standard Okapi BM25 parameters (Robertson & Zaragoza, 2009 defaults).
BM25_K1 = 1.5
BM25_B = 0.75


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
        # Lazily loads index_store/entity_graph.jsonl if present; graph_search
        # simply returns nothing if that file doesn't exist yet.
        self.entity_graph = entity_graph if entity_graph is not None else EntityGraph()
        self._build_inverted_index()

    def reset(self) -> None:
        self.state = InteractionState()

    # -- inverted index (built once, shared by exact_search/boolean_search) --

    def _build_inverted_index(self) -> None:
        """Tokenize every chunk once and build term -> {chunk_idx: term_freq}
        postings plus per-term document frequency. This is the one-time
        O(corpus size) cost that turns exact_search/boolean_search into
        O(query terms) lookups afterward, instead of a fresh linear scan
        over every chunk on every call.
        """
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_freq: dict[str, int] = defaultdict(int)
        num_docs = len(self.index.meta)
        doc_lens = np.zeros(num_docs, dtype=np.float64)
        for i, m in enumerate(self.index.meta):
            tokens = _tokenize(m["text"])
            doc_lens[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings[term][i] = tf
                self._doc_freq[term] += 1
        self._doc_lens = doc_lens
        self._num_docs = num_docs
        self._avg_doc_len = float(doc_lens.mean()) if num_docs else 0.0

    # -- scoring ---------------------------------------------------------

    def _dense_scores(self, query: str) -> np.ndarray:
        qvec = _normalize(embed_texts([query]))[0]
        return self.index.matrix @ qvec  # cosine sim, both sides normalized

    def _bm25_raw(self, keywords: str, k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
        """Okapi BM25 over the inverted index: term-frequency saturation (a
        term repeated many times in one chunk stops adding score linearly)
        and length normalization (a long chunk needs proportionally more
        hits than a short one to score as well), on top of inverse document
        frequency. This is what actually fixes the earlier ad-hoc scorer's
        failure mode (a long dictionary-style chunk repeating one common
        term beating a chunk that genuinely covers several distinguishing
        terms): BM25 caps how much repetition alone can contribute and
        penalizes it for chunk length, rather than just counting distinct
        terms matched.
        """
        terms = set(_tokenize(keywords))
        scores = np.zeros(self._num_docs, dtype=np.float64)
        if not terms or not self._num_docs:
            return scores
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._doc_freq[term]
            idf = math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue
            for doc_idx, tf in postings.items():
                norm_len = (self._doc_lens[doc_idx] / self._avg_doc_len) if self._avg_doc_len else 0.0
                denom = tf + k1 * (1 - b + b * norm_len)
                scores[doc_idx] += idf * (tf * (k1 + 1)) / denom
        return scores

    def _sparse_scores(self, keywords: str) -> np.ndarray:
        """BM25 ranking, rescaled to ~[0, 1] alongside cosine similarity so
        weighted_fusion's linear blend with dense scores stays meaningful."""
        raw = self._bm25_raw(keywords)
        peak = raw.max()
        return raw / peak if peak > 0 else raw

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

    def _rank_exact(self, matched: set[int]) -> list[SearchHit]:
        """Like `_rank`, but for boolean/exact matches: returns *only* the
        chunks in `matched` (deduped per document, capped at `adjust_scale`)
        -- never padded with non-matching filler the way `_rank`'s
        score-everything-then-take-top-K does, since "no match" has to mean
        zero results here, not the corpus's least-bad guess. Pinned docs
        from `include_docs` are intentionally NOT force-included -- forcing
        in a document that doesn't actually satisfy the boolean expression
        would contradict what this action is for.
        """
        best_per_doc: dict[str, int] = {}
        for idx in sorted(matched):
            docid = self.index.meta[idx]["docid"]
            if docid in self.state.excluded_docids or docid in best_per_doc:
                continue
            best_per_doc[docid] = idx

        hits = []
        for docid, idx in list(best_per_doc.items())[: self.state.scale]:
            m = self.index.meta[idx]
            words = m["text"].split()
            hits.append(SearchHit(docid=docid, url=m["url"], score=1.0, snippet=" ".join(words[:400])))
        return hits

    # -- action primitives -------------------------------------------------

    def semantic_search(self, query: str) -> list[SearchHit]:
        """Dense retrieval: embedding similarity to the query."""
        return self._rank(self._dense_scores(query))

    def exact_search(self, keywords: str) -> list[SearchHit]:
        """Sparse retrieval: BM25 ranking over the inverted index."""
        return self._rank(self._sparse_scores(keywords))

    def boolean_search(
        self,
        and_terms: Iterable[str] | None = None,
        or_terms: Iterable[str] | None = None,
        not_terms: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """Exact boolean retrieval: set operations over the inverted index's
        postings lists, with no ranking at all -- every match is treated as
        equally valid (score 1.0), unlike BM25's graded relevance. AND
        requires every one of `and_terms` present; OR requires at least one
        of `or_terms`; if both are given, both constraints must hold
        (AND-of-ANDs-and-the-OR-group); NOT excludes any chunk containing a
        term in `not_terms`, applied last. Returns nothing if neither
        `and_terms` nor `or_terms` is given, since that would otherwise
        match (most of) the corpus.
        """
        and_terms = [t.lower().strip() for t in (and_terms or []) if t and t.strip()]
        or_terms = [t.lower().strip() for t in (or_terms or []) if t and t.strip()]
        not_terms = [t.lower().strip() for t in (not_terms or []) if t and t.strip()]

        if not and_terms and not or_terms:
            return []

        matched: set[int] | None = None
        for term in and_terms:
            term_docs = set(self._postings.get(term, {}).keys())
            matched = term_docs if matched is None else (matched & term_docs)
            if not matched:
                matched = set()
                break

        if or_terms:
            or_union: set[int] = set()
            for term in or_terms:
                or_union |= set(self._postings.get(term, {}).keys())
            matched = or_union if matched is None else (matched & or_union)

        matched = matched or set()
        for term in not_terms:
            matched -= set(self._postings.get(term, {}).keys())

        return self._rank_exact(matched)

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

        Returns nothing if `index_store/entity_graph.jsonl` doesn't exist for
        this corpus (see `entity_graph.py`). Results are deduped per source
        document (richest-in-relations document first, up to
        `adjust_scale`'s current limit); the snippet is the extracted
        relation(s) themselves rather than raw chunk text, so the Executor
        sees *why* a document matched, not just that it did. A relation
        whose tail is a concept (not a literal entity) is marked `[concept]`
        so the Executor doesn't mistake it for another named entity to chase.
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
                f'{e["subject"]} {e["relation"]} {e["object"]}' + (" [concept]" if e.get("object_is_concept") else "")
                for e in doc_edges[:5]
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
