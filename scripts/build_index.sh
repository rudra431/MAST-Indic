#!/usr/bin/env bash
# Build the local embedding index over the BrowseComp-Plus corpus.
#
# Usage:
#   ./scripts/build_index.sh            # full ~100k-doc corpus (slow, see README)
#   ./scripts/build_index.sh --limit 500  # dev-scale subset for quick testing
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.index build "$@"
