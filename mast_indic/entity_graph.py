"""In-memory entity relationship graph, loaded from a pre-extracted JSONL.

Loads `index_store/entity_graph.jsonl` -- one JSON object per chunk, shaped
like:

    {
      "doc_id": "41758__chunk_0001",
      "entities": [{"id": "vikings", "name": "Vikings", "type": "Team"}, ...],
      "concepts": [{"id": "concept_historical_drama", "name": "Historical Drama",
                     "description": "...", "level": 1}, ...],
      "triplets": [{"head": "Vikings", "relation": "aired_on", "tail": "History",
                     "tail_is_concept": false}, ...]
    }

into an adjacency list keyed by lowercased name, for multi-hop lookups from
`CorpusInteractionEngine.graph_search`. Each triplet is indexed in both
directions (head->tail and tail->head) so traversal finds a node regardless
of which side named it. `entities`/`concepts` are enrichment, not a strict
node universe -- a triplet's `head`/`tail` can (and often does) name a
string that never appears in either list; those are still valid graph nodes,
just without a known `type`/`description`.

This supersedes `graph_builder.py`'s own (much simpler) extraction format --
see that module's docstring. Bring your own extraction pipeline that
produces this shape; this module only loads and traverses it.

This is a flat in-memory structure, not a graph database -- fine at the
chunk/relation counts a single-pass extraction over this corpus produces;
swap in a real graph store (Neo4j, etc.) if you outgrow it.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

from .config import config

ENTITY_GRAPH_PATH_NAME = "entity_graph.jsonl"

# "{docid}__chunk_{NNNN}" (1-indexed, matching common chunk-export
# conventions) -> (docid, chunk_id). The chunk_id conversion is best-effort:
# `graph_search` already falls back to any chunk of the matched docid if the
# specific chunk_id isn't found in this index's own `meta.jsonl`, so exact
# alignment isn't required for correctness -- it only ever affects which of
# a doc's chunks gets used as the source of the shown snippet/URL.
_DOC_ID_RE = re.compile(r"^(.*)__chunk_(\d+)$")


def _parse_doc_id(doc_id: str) -> tuple[str, int]:
    match = _DOC_ID_RE.match(doc_id or "")
    if not match:
        return doc_id or "", 0
    docid, chunk_num = match.group(1), int(match.group(2))
    return docid, max(chunk_num - 1, 0)


@dataclass
class GraphEdge:
    subject: str
    relation: str
    object: str
    docid: str
    chunk_id: int
    object_is_concept: bool = False


class EntityGraph:
    """Adjacency list over extracted (head, relation, tail) triplets, loaded
    once at startup, plus lowercased-name -> type/description lookups for
    the entities and concepts the extraction recognized.
    """

    def __init__(self, relations_path: str | None = None) -> None:
        self.path = relations_path or os.path.join(config.index_dir, ENTITY_GRAPH_PATH_NAME)
        self.edges: list[GraphEdge] = []
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        self.entity_types: dict[str, str] = {}
        self.concept_descriptions: dict[str, str] = {}
        if not os.path.exists(self.path):
            return

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                docid, chunk_id = _parse_doc_id(row.get("doc_id", ""))

                for e in row.get("entities") or []:
                    name = e.get("name")
                    if name:
                        self.entity_types[name.lower()] = e.get("type", "")
                for c in row.get("concepts") or []:
                    name = c.get("name")
                    if name:
                        self.concept_descriptions[name.lower()] = c.get("description", "")

                for t in row.get("triplets") or []:
                    head, relation, tail = t.get("head"), t.get("relation"), t.get("tail")
                    if not (head and relation and tail):
                        continue
                    is_concept = bool(t.get("tail_is_concept"))
                    edge = GraphEdge(
                        subject=head, relation=relation, object=tail,
                        docid=docid, chunk_id=chunk_id, object_is_concept=is_concept,
                    )
                    self.edges.append(edge)
                    self.adjacency[head.lower()].append(edge)
                    # Index the reverse direction too, so a BFS starting from
                    # either side of the original triplet finds this edge.
                    self.adjacency[tail.lower()].append(GraphEdge(
                        subject=tail, relation=f"<-{relation}-", object=head,
                        docid=docid, chunk_id=chunk_id, object_is_concept=False,
                    ))

    @property
    def is_built(self) -> bool:
        return bool(self.edges)

    def neighbors(self, entity: str, hops: int = 1) -> list[dict]:
        """BFS outward from `entity` up to `hops` hops; returns edges with provenance."""
        hops = max(1, min(hops, 3))
        start = entity.lower().strip()
        if not start:
            return []
        visited = {start}
        frontier = [start]
        found: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for edge in self.adjacency.get(node, []):
                    key = (edge.subject.lower(), edge.relation, edge.object.lower())
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    found.append({
                        "subject": edge.subject, "relation": edge.relation, "object": edge.object,
                        "docid": edge.docid, "chunk_id": edge.chunk_id,
                        "object_is_concept": edge.object_is_concept,
                    })
                    other = edge.object.lower()
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break
        return found
