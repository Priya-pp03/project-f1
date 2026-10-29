"""Chunk the F1 circuits dataset.

Turns each row of circuits.csv into one self-contained, retrieval-friendly text
chunk plus its metadata, and writes the result to data/chunks.jsonl.

Usage:
    python chunk_circuits.py
    python chunk_circuits.py --input circuits.csv --output data/chunks.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT = Path("circuits.csv")
DEFAULT_OUTPUT = Path("data/chunks.jsonl")


def format_coordinates(lat: str, lng: str) -> str:
    """Return a human-readable coordinate pair, tolerating missing values."""
    lat, lng = lat.strip(), lng.strip()
    if not lat or not lng:
        return "unknown"
    return f"{lat}, {lng}"


def build_chunk_text(row: dict[str, str]) -> str:
    """Build the text that gets embedded for a single circuit row."""
    name = row["name"].strip()
    location = row["location"].strip()
    country = row["country"].strip()

    lines = [
        f"Circuit: {name}",
        f"Location: {location}, {country}",
        f"Coordinates: {format_coordinates(row['lat'], row['lng'])}",
    ]

    alt = row.get("alt", "").strip()
    if alt:
        lines.append(f"Altitude: {alt} m")

    ref = row.get("circuitRef", "").strip()
    circuit_id = row.get("circuitId", "").strip()
    if ref:
        lines.append(f"Reference: {ref} (circuitId {circuit_id})")

    url = row.get("url", "").strip()
    if url:
        lines.append(f"Wikipedia: {url}")

    return "\n".join(lines)


def iter_chunks(input_path: Path):
    """Yield one chunk dict per CSV row, in file order."""
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is empty or has no header row")

        for row in reader:
            if not (row.get("name") or "").strip():
                # Skip blank / malformed trailing lines.
                continue

            circuit_id = (row.get("circuitId") or "").strip()
            yield {
                "id": f"circuit-{circuit_id}",
                "text": build_chunk_text(row),
                "metadata": {
                    "circuit_id": circuit_id,
                    "circuit_ref": (row.get("circuitRef") or "").strip(),
                    "name": row["name"].strip(),
                    "location": (row.get("location") or "").strip(),
                    "country": (row.get("country") or "").strip(),
                    "lat": (row.get("lat") or "").strip(),
                    "lng": (row.get("lng") or "").strip(),
                    "alt": (row.get("alt") or "").strip(),
                    "url": (row.get("url") or "").strip(),
                },
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")

    chunks = list(iter_chunks(args.input))
    if not chunks:
        parser.error(f"no chunks produced from {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"Wrote {len(chunks)} chunks to {args.output}")
    print(f"Average chunk size: {total_chars / len(chunks):.0f} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
