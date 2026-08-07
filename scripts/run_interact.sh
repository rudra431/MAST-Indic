#!/usr/bin/env bash
# Run the Interact-RAG-style agent (mast_indic/interact_agent.py) over one
# (or all) Track 2 Indic languages and write submission JSONL.
#
# Usage:
#   ./scripts/run_interact.sh --language hi
#   ./scripts/run_interact.sh --language all --limit 5   # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.interact_runner "$@"
