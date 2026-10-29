"""Query the embedded circuits by natural-language similarity.

Loads data/embeddings.npy + data/embeddings_meta.json, embeds the query, and
prints the nearest circuits by cosine similarity.

Usage:
    python search.py "street circuit in the united states"
    python search.py "high altitude track" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

DEFAULT_EMBEDDINGS = Path("data/embeddings.npy")
DEFAULT_META = Path("data/embeddings_meta.json")
ENV_FILE = Path(__file__).resolve().parent / ".env"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(ENV_FILE)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (see .env.example)")
    if not args.embeddings.exists() or not args.meta.exists():
        raise SystemExit("embeddings not found; run chunk_circuits.py then embed_chunks.py")

    from openai import OpenAI

    with args.meta.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    matrix = np.load(args.embeddings)
    if matrix.shape[0] != len(meta["chunks"]):
        raise SystemExit("embeddings and manifest are out of sync")

    # Reuse the endpoint the index was built with, so query vectors match.
    base_url = os.environ.get("OPENAI_BASE_URL") or meta.get("base_url")
    if base_url and base_url.rstrip("/").endswith("api.openai.com/v1"):
        base_url = None
    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    response = client.embeddings.create(model=meta["model"], input=[args.query])
    query_vec = np.asarray(response.data[0].embedding, dtype=np.float32)

    if query_vec.shape[0] != matrix.shape[1]:
        raise SystemExit(
            f"query dim {query_vec.shape[0]} != index dim {matrix.shape[1]}; "
            "re-embed before searching"
        )

    # Cosine similarity: both query and index vectors are already unit-friendly
    # but we normalise explicitly to be safe.
    index_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    index_norms[index_norms == 0] = 1.0
    unit_index = matrix / index_norms
    unit_query = query_vec / (np.linalg.norm(query_vec) or 1.0)

    scores = unit_index @ unit_query
    order = np.argsort(-scores)[: args.top_k]

    for rank, idx in enumerate(order, start=1):
        entry = meta["chunks"][int(idx)]
        md = entry.get("metadata", {})
        print(f"{rank}. [{scores[idx]:.3f}] {md.get('name', entry['id'])} "
              f"- {md.get('location', '?')}, {md.get('country', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
