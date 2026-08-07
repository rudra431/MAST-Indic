"""In-memory entity relationship graph, built by `graph_builder.py`.

Loads `index_store/relations.jsonl` -- (subject, relation, object, docid,
chunk_id) rows extracted per chunk -- into an adjacency list keyed by
lowercased entity name, for multi-hop lookups from
`CorpusInteractionEngine.graph_search`. Each relation is indexed in both
directions (subject->object and object->subject) so traversal finds an
entity regardless of which side of the extracted sentence named it.

This is a flat in-memory structure, not a graph database -- fine at the
chunk/relation counts this project's LLM-based extraction produces (see
`graph_builder.py`); swap in a real graph store (Neo4j, etc.) if you outgrow
it.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

from .config import config

RELATIONS_PATH_NAME = "relations.jsonl"


@dataclass
class GraphEdge:
    subject: str
    relation: str
    object: str
    docid: str
    chunk_id: int


class EntityGraph:
    """Adjacency list over extracted relation triples, loaded once at startup."""

    def __init__(self, relations_path: str | None = None) -> None:
        self.path = relations_path or os.path.join(config.index_dir, RELATIONS_PATH_NAME)
        self.edges: list[GraphEdge] = []
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                edge = GraphEdge(
                    subject=row["subject"], relation=row["relation"], object=row["object"],
                    docid=row["docid"], chunk_id=row["chunk_id"],
                )
                self.edges.append(edge)
                self.adjacency[edge.subject.lower()].append(edge)
                # Index the reverse direction too, so a BFS starting from
                # either side of the original sentence finds this edge.
                self.adjacency[edge.object.lower()].append(GraphEdge(
                    subject=edge.object, relation=f"<-{edge.relation}-", object=edge.subject,
                    docid=edge.docid, chunk_id=edge.chunk_id,
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
                    })
                    other = edge.object.lower()
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break
        return found
