#!/usr/bin/env bash
# Extract an entity relationship graph from an already-chunked corpus
# (mast_indic/graph_builder.py), for the interact agent's `graph_search`.
#
# Usage:
#   ./scripts/build_entity_graph.sh --limit 500   # dev-scale subset for quick testing
#   ./scripts/build_entity_graph.sh                # full corpus (slow, see README)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.graph_builder build "$@"
