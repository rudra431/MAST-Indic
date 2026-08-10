#!/usr/bin/env bash
# Run the exact/sparse-only agent (mast_indic/exact_agent.py) over one (or
# all) Track 2 Indic languages and write submission JSONL.
#
# Usage:
#   ./scripts/run_exact.sh --language hi
#   ./scripts/run_exact.sh --language all --limit 5   # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.exact_runner "$@"
