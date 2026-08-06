#!/usr/bin/env bash
# Run the agent over one (or all) Track 2 Indic languages and write submission JSONL.
#
# Usage:
#   ./scripts/run_track2.sh --language hi
#   ./scripts/run_track2.sh --language all --limit 5   # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.runner "$@"
