"""Export the corpus dataset to one Markdown file per document.

Standalone utility for browsing/inspecting `MAST_CORPUS_DATASET` (the same
dataset `index.py build` streams and chunks into the embedding index) as
plain human-readable files -- useful for manually checking what a document
actually says when debugging retrieval quality, without needing the index,
an LLM, or any API keys.

Reuses `index.py`'s corpus loader so both stay in sync with whatever
dataset you have configured.
"""
from __future__ import annotations

import argparse
import os
import re

from .index import iter_corpus

_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(docid: str) -> str:
    """Sanitize a docid into a filesystem-safe filename.

    Docids can contain slashes, spaces, or other characters that aren't
    safe to use bare as a filename (and a literal "/" would be
    misinterpreted as a subdirectory).
    """
    name = _UNSAFE_FILENAME_RE.sub("_", docid).strip("_")
    return name or "untitled"


def export_corpus(output_dir: str, limit: int | None = None, resume: bool = True) -> str:
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    for doc in iter_corpus(limit=limit):
        path = os.path.join(output_dir, f"{safe_filename(doc['docid'])}.md")

        if resume and os.path.exists(path):
            skipped += 1
            continue

        lines = [f"# {doc['docid']}", ""]
        if doc.get("url"):
            lines += [f"**Source:** {doc['url']}", ""]
        lines.append(doc["text"])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    print(f"Wrote {written} file(s), skipped {skipped} already-existing -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export each corpus document as its own Markdown file."
    )
    parser.add_argument(
        "--output-dir", default="corpus_markdown",
        help="Directory to write one .md file per document into (default: corpus_markdown)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only export the first N documents (dev/testing)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="re-export every document even if its .md file already exists "
        "(default: skip docs already exported, so an interrupted run can resume)",
    )
    args = parser.parse_args()

    export_corpus(output_dir=args.output_dir, limit=args.limit, resume=not args.fresh)
