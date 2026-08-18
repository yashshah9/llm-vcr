"""SSE parse and replay helpers."""

from __future__ import annotations


def parse_sse(raw: str) -> list[str]:
    """Split an SSE payload into event chunks (including trailing newlines)."""
    if not raw:
        return []
    parts = raw.split("\n\n")
    chunks = [p + "\n\n" for p in parts if p.strip()]
    return chunks


def join_chunks(chunks: list[str]) -> bytes:
    return "".join(chunks).encode()
