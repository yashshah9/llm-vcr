"""Anthropic Messages API URL/body normalization for cassette matching."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from llm_vcr.matching import DROP_KEYS, normalize_model

ANTHROPIC_HOST = "api.anthropic.com"

# Anthropic `metadata` is caller-supplied and volatile for matching.
ANTHROPIC_DROP_KEYS = DROP_KEYS | {"metadata"}


def is_anthropic_url(url: str) -> bool:
    host = urlsplit(url).hostname
    return host == ANTHROPIC_HOST


def normalize_anthropic_url(url: str) -> str:
    """Strip query/fragment (e.g. beta flags) so path-based matches stay stable."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def normalize_anthropic_body(body: dict[str, Any] | None) -> dict[str, Any]:
    """Drop volatile keys and normalize model date suffixes."""
    if not body:
        return {}
    out: dict[str, Any] = {}
    for key, value in body.items():
        if key in ANTHROPIC_DROP_KEYS:
            continue
        if key == "model" and isinstance(value, str):
            out[key] = normalize_model(value)
        else:
            out[key] = value
    return out


def _normalize_content_block(block: Any, *, semantic: bool) -> Any:
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": block.get("text")}
    if block_type == "tool_use":
        out: dict[str, Any] = {
            "type": "tool_use",
            "name": block.get("name"),
            "input": block.get("input"),
        }
        if not semantic and "id" in block:
            out["id"] = block["id"]
        return out
    if block_type == "tool_result":
        out = {"type": "tool_result", "content": block.get("content")}
        if "is_error" in block:
            out["is_error"] = block["is_error"]
        if not semantic and "tool_use_id" in block:
            out["tool_use_id"] = block["tool_use_id"]
        return out
    # Unknown block types: drop volatile id when semantic.
    if semantic:
        return {k: v for k, v in block.items() if k != "id"}
    return dict(block)


def _normalize_content(content: Any, *, semantic: bool) -> Any:
    if isinstance(content, list):
        return [_normalize_content_block(b, semantic=semantic) for b in content]
    return content


def _normalize_message(msg: Any, *, semantic: bool) -> Any:
    if not isinstance(msg, dict):
        return msg
    out: dict[str, Any] = {}
    if "role" in msg:
        out["role"] = msg["role"]
    if "content" in msg:
        out["content"] = _normalize_content(msg["content"], semantic=semantic)
    return out


def _normalize_system(system: Any, *, semantic: bool) -> Any:
    if isinstance(system, list):
        return [_normalize_content_block(b, semantic=semantic) for b in system]
    return system


def _normalize_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool
    out: dict[str, Any] = {}
    for key in ("name", "description", "input_schema"):
        if key in tool:
            out[key] = tool[key]
    return out


def semantic_normalize_anthropic_body(body: dict[str, Any] | None) -> dict[str, Any]:
    """Semantic match: also normalize messages/system/tools; strip tool_use ids."""
    base = normalize_anthropic_body(body)
    out = dict(base)
    if "system" in out:
        out["system"] = _normalize_system(out["system"], semantic=True)
    if isinstance(out.get("messages"), list):
        out["messages"] = [_normalize_message(m, semantic=True) for m in out["messages"]]
    if isinstance(out.get("tools"), list):
        tools = [_normalize_tool(t) for t in out["tools"]]
        out["tools"] = sorted(
            tools,
            key=lambda t: str(t.get("name", "")) if isinstance(t, dict) else "",
        )
    return out
