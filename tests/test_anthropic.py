"""Tests for Anthropic Messages API matching/normalization."""

import httpx

from llm_vcr.cassette import Cassette, Interaction, find_interaction
from llm_vcr.matching import (
    bodies_equal_semantic,
    normalize_body,
    normalize_url,
    semantic_normalize_body,
)
from llm_vcr.providers.anthropic import (
    is_anthropic_url,
    normalize_anthropic_body,
    normalize_anthropic_url,
    semantic_normalize_anthropic_body,
)
from llm_vcr.redaction import SECRET_KEYS, redact_dict
from llm_vcr.transport import VCRTransport

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_URL_BETA = "https://api.anthropic.com/v1/messages?beta=true"


def test_is_anthropic_url() -> None:
    assert is_anthropic_url(ANTHROPIC_URL)
    assert is_anthropic_url(ANTHROPIC_URL_BETA)
    assert not is_anthropic_url("https://api.openai.com/v1/chat/completions")


def test_normalize_anthropic_url_strips_query() -> None:
    assert normalize_anthropic_url(ANTHROPIC_URL_BETA) == ANTHROPIC_URL
    assert normalize_url(ANTHROPIC_URL_BETA) == ANTHROPIC_URL
    assert normalize_url("https://api.openai.com/v1/chat/completions?x=1").endswith("?x=1")


def test_normalize_anthropic_body_drops_metadata_and_model_date() -> None:
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 256,
        "metadata": {"user_id": "u-1"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    norm = normalize_anthropic_body(body)
    assert norm["model"] == "claude-3-5-sonnet"
    assert "metadata" not in norm
    assert normalize_body(body, url=ANTHROPIC_URL) == norm


def test_semantic_strips_tool_use_ids() -> None:
    left = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "metadata": {"user_id": "a"},
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_old",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_old",
                        "content": "sunny",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather",
                "input_schema": {"type": "object"},
            }
        ],
    }
    right = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 100,
        "metadata": {"user_id": "b"},
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_new",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_new",
                        "content": "sunny",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather",
                "input_schema": {"type": "object"},
            }
        ],
    }
    assert bodies_equal_semantic(left, right, url=ANTHROPIC_URL)
    sem = semantic_normalize_anthropic_body(left)
    tool_use = sem["messages"][1]["content"][0]
    assert "id" not in tool_use
    assert "tool_use_id" not in sem["messages"][2]["content"][0]
    # Without Anthropic URL, OpenAI path keeps content blocks untouched → ids remain
    openai_sem = semantic_normalize_body(left)
    assert openai_sem["messages"][1]["content"][0]["id"] == "toolu_old"


def test_redaction_covers_x_api_key() -> None:
    assert "x-api-key" in SECRET_KEYS
    assert redact_dict({"x-api-key": "sk-ant-secret", "model": "claude"})["x-api-key"] == (
        "[REDACTED]"
    )


def test_find_and_transport_anthropic_semantic_matcher() -> None:
    recorded = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "metadata": {"user_id": "recorded"},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_old",
                        "name": "ping",
                        "input": {},
                    }
                ],
            }
        ],
    }
    live = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 64,
        "metadata": {"user_id": "live"},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_new",
                        "name": "ping",
                        "input": {},
                    }
                ],
            }
        ],
    }
    cassette = Cassette(
        name="anthropic_sem",
        interactions=[
            Interaction(
                method="POST",
                url=ANTHROPIC_URL,
                request_body=recorded,
                response_body={"ok": True},
            )
        ],
    )
    assert find_interaction(cassette, "POST", ANTHROPIC_URL_BETA, live) is None
    found = find_interaction(
        cassette, "POST", ANTHROPIC_URL_BETA, live, matcher="semantic"
    )
    assert found is not None

    transport = VCRTransport(cassette, record_mode=False, matcher="semantic")
    client = httpx.Client(transport=transport)
    resp = client.post(ANTHROPIC_URL_BETA, json=live)
    assert resp.json() == {"ok": True}
    client.close()


def test_exact_match_anthropic_query_stripped() -> None:
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    cassette = Cassette(
        name="anthropic_exact",
        interactions=[
            Interaction(
                method="POST",
                url=ANTHROPIC_URL,
                request_body=body,
                response_body={"content": [{"type": "text", "text": "hello"}]},
            )
        ],
    )
    found = find_interaction(cassette, "POST", ANTHROPIC_URL_BETA, body)
    assert found is not None
