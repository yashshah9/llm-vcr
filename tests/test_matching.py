"""Tests for request body normalization and diff."""

import httpx

from llm_vcr.cassette import Cassette, Interaction, find_interaction
from llm_vcr.matching import (
    bodies_equal_semantic,
    diff_bodies,
    normalize_body,
    normalize_model,
    semantic_normalize_body,
)
from llm_vcr.transport import VCRTransport


def test_normalize_model_strips_date_suffix() -> None:
    assert normalize_model("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert normalize_model("gpt-4o-mini") == "gpt-4o-mini"


def test_normalize_model_applies_alias() -> None:
    assert normalize_model("chatgpt-4o-latest") == "gpt-4o"
    assert normalize_model("gpt-4o-2024-08-06") == "gpt-4o"


def test_normalize_model_strips_anthropic_compact_date() -> None:
    assert normalize_model("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"
    assert normalize_model("claude-3-5-sonnet") == "claude-3-5-sonnet"


def test_normalize_body_drops_volatile_and_aliases_model() -> None:
    body = {
        "model": "gpt-4o-mini-2024-07-18",
        "user": "u-123",
        "messages": [{"role": "user", "content": "hi"}],
    }
    norm = normalize_body(body)
    assert norm["model"] == "gpt-4o-mini"
    assert "user" not in norm
    assert norm["messages"][0]["content"] == "hi"


def test_diff_bodies_reports_model_and_message_changes() -> None:
    left = {"model": "gpt-4o-mini-2024-07-18", "messages": [{"role": "user", "content": "a"}]}
    right = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "b"}]}
    lines = diff_bodies(left, right)
    assert any("messages" in line for line in lines)
    # model dates normalize equal — should not appear as a diff
    assert not any(line.startswith("~ model") for line in lines)


def test_semantic_ignores_volatile_and_tool_call_ids() -> None:
    left = {
        "model": "gpt-4o-mini-2024-07-18",
        "user": "u-aaa",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
    }
    right = {
        "model": "gpt-4o-mini",
        "user": "u-zzz",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
    }
    assert bodies_equal_semantic(left, right)
    assert diff_bodies(left, right) == []
    # exact normalize still keeps tool_call ids inside messages
    assert normalize_body(left)["messages"] != normalize_body(right)["messages"]
    assert "id" not in semantic_normalize_body(left)["messages"][1]["tool_calls"][0]


def test_semantic_rejects_different_tool_name() -> None:
    left = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}}
                ],
            }
        ]
    }
    right = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "b", "arguments": "{}"}}
                ],
            }
        ]
    }
    assert not bodies_equal_semantic(left, right)


def test_find_and_transport_semantic_matcher() -> None:
    cassette = Cassette(
        name="sem",
        interactions=[
            Interaction(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                request_body={
                    "model": "gpt-4o-mini-2024-07-18",
                    "user": "recorded",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_old",
                                    "type": "function",
                                    "function": {"name": "ping", "arguments": "{}"},
                                }
                            ],
                        }
                    ],
                },
                response_body={"ok": True},
            )
        ],
    )
    live = {
        "model": "gpt-4o-mini",
        "user": "live",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_new",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            }
        ],
    }
    assert find_interaction(cassette, "POST", cassette.interactions[0].url, live) is None
    found = find_interaction(
        cassette, "POST", cassette.interactions[0].url, live, matcher="semantic"
    )
    assert found is not None

    transport = VCRTransport(cassette, record_mode=False, matcher="semantic")
    client = httpx.Client(transport=transport)
    resp = client.post(cassette.interactions[0].url, json=live)
    assert resp.json() == {"ok": True}
    client.close()
