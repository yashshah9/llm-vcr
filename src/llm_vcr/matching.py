"""Normalize LLM request bodies so volatile fields do not break cassette matches."""

from __future__ import annotations

import re
from typing import Any, Literal

DROP_KEYS = {
    "user",
    "request_id",
    "timestamp",
    "created",
    "seed",
    "n",
}

# After date-strip, map uncommon renames onto a stable key.
# Dated OpenAI ids (gpt-4o-mini-2024-07-18) already collapse via _MODEL_DATE.
MODEL_ALIASES: dict[str, str] = {
    "chatgpt-4o-latest": "gpt-4o",
}

MatcherName = Literal["exact", "semantic"]

# Strip dated suffixes: gpt-4o-mini-2024-07-18 → gpt-4o-mini
# Anthropic uses compact dates: claude-3-5-sonnet-20241022 → claude-3-5-sonnet
_MODEL_DATE = re.compile(r"^(.*?)-\d{4}-\d{2}-\d{2}$")
_MODEL_DATE_COMPACT = re.compile(r"^(.*?)-\d{8}$")


def normalize_model(model: str) -> str:
    match = _MODEL_DATE.match(model) or _MODEL_DATE_COMPACT.match(model)
    base = match.group(1) if match else model
    return MODEL_ALIASES.get(base, base)


def normalize_url(url: str) -> str:
    """Provider-aware URL normalize (Anthropic: strip query/fragment)."""
    from llm_vcr.providers.anthropic import is_anthropic_url, normalize_anthropic_url

    if is_anthropic_url(url):
        return normalize_anthropic_url(url)
    return url


def normalize_body(
    body: dict[str, Any] | None,
    *,
    url: str | None = None,
) -> dict[str, Any]:
    """Drop volatile keys and normalize model strings (exact-match baseline)."""
    if url is not None:
        from llm_vcr.providers.anthropic import is_anthropic_url, normalize_anthropic_body

        if is_anthropic_url(url):
            return normalize_anthropic_body(body)
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


def _normalize_tool_call(tc: Any) -> Any:
    if not isinstance(tc, dict):
        return tc
    out: dict[str, Any] = {}
    if "type" in tc:
        out["type"] = tc["type"]
    fn = tc.get("function")
    if isinstance(fn, dict):
        out["function"] = {k: fn[k] for k in ("name", "arguments") if k in fn}
    return out


def _normalize_message(msg: Any) -> Any:
    """Keep role + content (+ tool function name/args); drop tool_call ids."""
    if not isinstance(msg, dict):
        return msg
    out: dict[str, Any] = {}
    if "role" in msg:
        out["role"] = msg["role"]
    if "content" in msg:
        out["content"] = msg["content"]
    if "name" in msg:
        out["name"] = msg["name"]
    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
        out["tool_calls"] = [_normalize_tool_call(tc) for tc in msg["tool_calls"]]
    return out


def _normalize_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool
    out: dict[str, Any] = {}
    if "type" in tool:
        out["type"] = tool["type"]
    fn = tool.get("function")
    if isinstance(fn, dict):
        out["function"] = {k: fn[k] for k in ("name", "description", "parameters") if k in fn}
    return out


def _tool_sort_key(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", ""))
    return str(tool.get("name", ""))


def semantic_normalize_body(
    body: dict[str, Any] | None,
    *,
    url: str | None = None,
) -> dict[str, Any]:
    """Normalize for semantic equality: volatile keys, model aliases, messages, tools."""
    if url is not None:
        from llm_vcr.providers.anthropic import (
            is_anthropic_url,
            semantic_normalize_anthropic_body,
        )

        if is_anthropic_url(url):
            return semantic_normalize_anthropic_body(body)
    base = normalize_body(body)
    out = dict(base)
    if isinstance(out.get("messages"), list):
        out["messages"] = [_normalize_message(m) for m in out["messages"]]
    if isinstance(out.get("tools"), list):
        tools = [_normalize_tool(t) for t in out["tools"]]
        out["tools"] = sorted(tools, key=_tool_sort_key)
    if isinstance(out.get("functions"), list):
        out["functions"] = sorted(
            out["functions"],
            key=lambda f: str(f.get("name", "")) if isinstance(f, dict) else "",
        )
    return out


def bodies_equal_semantic(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    url: str | None = None,
) -> bool:
    return semantic_normalize_body(left, url=url) == semantic_normalize_body(right, url=url)


def diff_bodies(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    url: str | None = None,
) -> list[str]:
    """Human-readable differences after semantic normalization."""
    a = semantic_normalize_body(left, url=url)
    b = semantic_normalize_body(right, url=url)
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
