#!/usr/bin/env bash
# Export each corpus document (MAST_CORPUS_DATASET) as its own Markdown
# file, for manual browsing/inspection independent of the embedding index.
#
# Usage:
#   ./scripts/export_corpus.sh --limit 200              # dev-scale subset
#   ./scripts/export_corpus.sh --output-dir corpus_md   # full corpus (many files)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mast_indic.export_corpus "$@"
