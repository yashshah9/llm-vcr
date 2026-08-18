"""Normalize LLM request bodies so volatile fields do not break cassette matches."""

from __future__ import annotations

from typing import Any

DROP_KEYS = {
    "user",
    "request_id",
    "timestamp",
    "created",
    "seed",
    "n",
}


def normalize_body(body: dict[str, Any] | None) -> dict[str, Any]:
    if not body:
        return {}
    return {k: v for k, v in body.items() if k not in DROP_KEYS}
