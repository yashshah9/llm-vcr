"""Normalize LLM request bodies so volatile fields do not break cassette matches."""

from __future__ import annotations

import re
from typing import Any

DROP_KEYS = {
    "user",
    "request_id",
    "timestamp",
    "created",
    "seed",
    "n",
}

# Strip dated OpenAI-style suffixes: gpt-4o-mini-2024-07-18 → gpt-4o-mini
_MODEL_DATE = re.compile(r"^(.*?)-\d{4}-\d{2}-\d{2}$")


def normalize_model(model: str) -> str:
    match = _MODEL_DATE.match(model)
    return match.group(1) if match else model


def normalize_body(body: dict[str, Any] | None) -> dict[str, Any]:
    if not body:
        return {}
    out: dict[str, Any] = {}
    for key, value in body.items():
        if key in DROP_KEYS:
            continue
        if key == "model" and isinstance(value, str):
            out[key] = normalize_model(value)
        else:
            out[key] = value
    return out


def diff_bodies(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> list[str]:
    """Human-readable differences after normalization."""
    a = normalize_body(left)
    b = normalize_body(right)
    lines: list[str] = []
    keys = sorted(set(a) | set(b))
    for key in keys:
        if key not in a:
            lines.append(f"+ {key}: {b[key]!r}")
        elif key not in b:
            lines.append(f"- {key}: {a[key]!r}")
        elif a[key] != b[key]:
            lines.append(f"~ {key}: {a[key]!r} → {b[key]!r}")
    return lines
