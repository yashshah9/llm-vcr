"""Robustness scenarios for semantic matching, replay, redaction, and diffs."""

from __future__ import annotations

import httpx
import pytest

from llm_vcr.cassette import Cassette, Interaction, find_interaction, request_key
from llm_vcr.matching import (
    MODEL_ALIASES,
    bodies_equal_semantic,
    diff_bodies,
    normalize_body,
    normalize_model,
    semantic_normalize_body,
)
from llm_vcr.redaction import redact_dict
from llm_vcr.transport import VCRTransport

URL = "https://api.openai.com/v1/chat/completions"


def _msg(role: str, content: str | None, **extra: object) -> dict:
    return {"role": role, "content": content, **extra}


def _tool_call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _chat_body(
    *,
    model: str = "gpt-4o-mini",
    user: str | None = None,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages
        or [_msg("user", "hi")],
    }
    if user is not None:
        body["user"] = user
    if tools is not None:
        body["tools"] = tools
    return body


# --- semantic equal despite volatile fields ---------------------------------


def test_semantic_equal_different_user_ids() -> None:
    left = _chat_body(user="u-aaa", messages=[_msg("user", "hello")])
    right = _chat_body(user="u-zzz", messages=[_msg("user", "hello")])
    assert bodies_equal_semantic(left, right)


def test_semantic_equal_different_tool_call_ids() -> None:
    left = _chat_body(
        messages=[
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("call_old", "ping", '{"x":1}')],
            )
        ]
    )
    right = _chat_body(
        messages=[
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("call_new", "ping", '{"x":1}')],
            )
        ]
    )
    assert bodies_equal_semantic(left, right)
    assert "id" not in semantic_normalize_body(left)["messages"][0]["tool_calls"][0]


def test_semantic_equal_model_date_suffix() -> None:
    left = _chat_body(model="gpt-4o-mini-2024-07-18")
    right = _chat_body(model="gpt-4o-mini")
    assert bodies_equal_semantic(left, right)
    assert diff_bodies(left, right) == []


def test_semantic_equal_combined_volatile_fields() -> None:
    left = _chat_body(
        model="gpt-4o-2024-08-06",
        user="recorded",
        messages=[
            _msg("user", "weather?"),
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("call_a", "get_weather", '{"city":"NYC"}')],
            ),
        ],
    )
    right = _chat_body(
        model="gpt-4o",
        user="live",
        messages=[
            _msg("user", "weather?"),
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("call_b", "get_weather", '{"city":"NYC"}')],
            ),
        ],
    )
    assert bodies_equal_semantic(left, right)


# --- semantic unequal on content / tool name / role -------------------------


def test_semantic_unequal_message_content() -> None:
    left = _chat_body(messages=[_msg("user", "alpha")])
    right = _chat_body(messages=[_msg("user", "beta")])
    assert not bodies_equal_semantic(left, right)


def test_semantic_unequal_tool_name() -> None:
    left = _chat_body(
        messages=[
            _msg("assistant", None, tool_calls=[_tool_call("1", "get_weather")])
        ]
    )
    right = _chat_body(
        messages=[
            _msg("assistant", None, tool_calls=[_tool_call("1", "get_time")])
        ]
    )
    assert not bodies_equal_semantic(left, right)


def test_semantic_unequal_role() -> None:
    left = _chat_body(messages=[_msg("user", "same")])
    right = _chat_body(messages=[_msg("system", "same")])
    assert not bodies_equal_semantic(left, right)


def test_semantic_unequal_tool_arguments() -> None:
    left = _chat_body(
        messages=[
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("1", "ping", '{"a":1}')],
            )
        ]
    )
    right = _chat_body(
        messages=[
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("1", "ping", '{"a":2}')],
            )
        ]
    )
    assert not bodies_equal_semantic(left, right)


# --- exact matcher hash-based -----------------------------------------------


def test_exact_matcher_misses_on_tool_call_id_difference() -> None:
    """Exact hashing keeps tool_call ids; semantic would match."""
    recorded = _chat_body(
        user="u-1",
        messages=[
            _msg("assistant", None, tool_calls=[_tool_call("call_old", "ping")])
        ],
    )
    live = _chat_body(
        user="u-1",
        messages=[
            _msg("assistant", None, tool_calls=[_tool_call("call_new", "ping")])
        ],
    )
    assert bodies_equal_semantic(recorded, live)
    assert request_key("POST", URL, recorded) != request_key("POST", URL, live)
    cassette = Cassette(
        name="exact-miss",
        interactions=[
            Interaction(method="POST", url=URL, request_body=recorded, response_body={"ok": 1})
        ],
    )
    assert find_interaction(cassette, "POST", URL, live, matcher="exact") is None
    assert find_interaction(cassette, "POST", URL, live, matcher="semantic") is not None


def test_exact_normalize_drops_user_so_user_alone_still_matches() -> None:
    """User is in DROP_KEYS — exact hash ignores it (not a miss by itself)."""
    a = _chat_body(user="alice", messages=[_msg("user", "hi")])
    b = _chat_body(user="bob", messages=[_msg("user", "hi")])
    assert "user" not in normalize_body(a)
    assert request_key("POST", URL, a) == request_key("POST", URL, b)
    cassette = Cassette(
        name="user-drop",
        interactions=[
            Interaction(method="POST", url=URL, request_body=a, response_body={"ok": True})
        ],
    )
    assert find_interaction(cassette, "POST", URL, b, matcher="exact") is not None


# --- MODEL_ALIASES / normalize_model edges ----------------------------------


def test_normalize_model_empty_string() -> None:
    assert normalize_model("") == ""


def test_normalize_model_already_clean() -> None:
    assert normalize_model("gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_model("gpt-4o") == "gpt-4o"


def test_normalize_model_multi_date_suffix_strips_trailing_only() -> None:
    # Non-greedy date strip: only the final YYYY-MM-DD is removed.
    assert normalize_model("gpt-4o-2024-01-01-2024-02-02") == "gpt-4o-2024-01-01"


def test_normalize_model_alias_chatgpt_4o_latest() -> None:
    assert MODEL_ALIASES["chatgpt-4o-latest"] == "gpt-4o"
    assert normalize_model("chatgpt-4o-latest") == "gpt-4o"


def test_normalize_model_dated_then_alias_path() -> None:
    # Date strip first; alias only applies to the remaining base.
    assert normalize_model("gpt-4o-2024-08-06") == "gpt-4o"
    assert normalize_model("unknown-model-2023-01-15") == "unknown-model"


# --- find_interaction semantic vs exact -------------------------------------


def test_find_interaction_semantic_hits_exact_misses() -> None:
    cassette = Cassette(
        name="find",
        interactions=[
            Interaction(
                method="POST",
                url=URL,
                request_body=_chat_body(
                    model="gpt-4o-mini-2024-07-18",
                    user="cassette",
                    messages=[
                        _msg(
                            "assistant",
                            None,
                            tool_calls=[_tool_call("id-cassette", "echo", "{}")],
                        )
                    ],
                ),
                response_body={"id": "chatcmpl-found"},
            )
        ],
    )
    live = _chat_body(
        model="gpt-4o-mini",
        user="runtime",
        messages=[
            _msg(
                "assistant",
                None,
                tool_calls=[_tool_call("id-live", "echo", "{}")],
            )
        ],
    )
    assert find_interaction(cassette, "POST", URL, live, matcher="exact") is None
    found = find_interaction(cassette, "POST", URL, live, matcher="semantic")
    assert found is not None
    assert found.response_body == {"id": "chatcmpl-found"}


def test_find_interaction_wrong_url_misses_both_matchers() -> None:
    body = _chat_body()
    cassette = Cassette(
        name="url",
        interactions=[
            Interaction(method="POST", url=URL, request_body=body, response_body={})
        ],
    )
    other = "https://api.openai.com/v1/embeddings"
    assert find_interaction(cassette, "POST", other, body, matcher="exact") is None
    assert find_interaction(cassette, "POST", other, body, matcher="semantic") is None


def test_cassette_replay_semantic_transport() -> None:
    cassette = Cassette(
        name="replay-sem",
        interactions=[
            Interaction(
                method="POST",
                url=URL,
                request_body=_chat_body(
                    model="gpt-4o-mini-2024-07-18",
                    user="rec",
                    messages=[_msg("user", "ping")],
                ),
                response_body={"choices": [{"message": {"content": "pong"}}]},
            )
        ],
    )
    transport = VCRTransport(cassette, record_mode=False, matcher="semantic")
    client = httpx.Client(transport=transport)
    resp = client.post(
        URL,
        json=_chat_body(model="gpt-4o-mini", user="live", messages=[_msg("user", "ping")]),
    )
    assert resp.json()["choices"][0]["message"]["content"] == "pong"
    client.close()


# --- redaction of nested secrets --------------------------------------------


def test_redact_nested_authorization_header_payload() -> None:
    data = {
        "request_headers": {"Authorization": "Bearer sk-live", "content-type": "application/json"},
        "meta": {"api_key": "sk-nested", "model": "gpt-4o"},
    }
    out = redact_dict(data)
    assert out["request_headers"]["Authorization"] == "[REDACTED]"
    assert out["request_headers"]["content-type"] == "application/json"
    assert out["meta"]["api_key"] == "[REDACTED]"
    assert out["meta"]["model"] == "gpt-4o"


def test_redact_list_of_dicts_with_tokens() -> None:
    data = {
        "items": [
            {"access_token": "tok-1", "name": "a"},
            {"refresh_token": "tok-2", "name": "b"},
            {"plain": "ok"},
        ]
    }
    out = redact_dict(data)
    assert out["items"][0]["access_token"] == "[REDACTED]"
    assert out["items"][0]["name"] == "a"
    assert out["items"][1]["refresh_token"] == "[REDACTED]"
    assert out["items"][2]["plain"] == "ok"


def test_redact_client_secret_key_substring() -> None:
    data = {"client_secret": "shh", "openai-api-key": "sk-x", "x-api-key": "k"}
    out = redact_dict(data)
    assert out["client_secret"] == "[REDACTED]"
    assert out["openai-api-key"] == "[REDACTED]"
    assert out["x-api-key"] == "[REDACTED]"


# --- diff_bodies / bodies_equal_semantic ------------------------------------


def test_diff_bodies_reports_added_and_removed_keys() -> None:
    left = {"model": "gpt-4o", "temperature": 0.2}
    right = {"model": "gpt-4o", "top_p": 1.0}
    lines = diff_bodies(left, right)
    assert any(line.startswith("- temperature") for line in lines)
    assert any(line.startswith("+ top_p") for line in lines)


def test_diff_bodies_empty_when_semantically_equal() -> None:
    left = _chat_body(model="chatgpt-4o-latest", user="a")
    right = _chat_body(model="gpt-4o", user="b")
    assert bodies_equal_semantic(left, right)
    assert diff_bodies(left, right) == []


def test_diff_bodies_reports_message_change() -> None:
    left = _chat_body(messages=[_msg("user", "a")])
    right = _chat_body(messages=[_msg("user", "b")])
    lines = diff_bodies(left, right)
    assert any(line.startswith("~ messages") for line in lines)


def test_bodies_equal_semantic_none_and_empty() -> None:
    assert bodies_equal_semantic(None, None)
    assert bodies_equal_semantic(None, {})
    assert bodies_equal_semantic({}, None)


# --- sequential unused counting edges ---------------------------------------


def test_sequential_unused_starts_at_full_count() -> None:
    cassette = Cassette(
        name="unused-full",
        interactions=[
            Interaction(method="POST", url=URL, request_body=_chat_body(), response_body={}),
            Interaction(
                method="POST",
                url=URL,
                request_body=_chat_body(messages=[_msg("user", "second")]),
                response_body={},
            ),
        ],
    )
    transport = VCRTransport(cassette, record_mode=False, sequential=True)
    assert transport.unused() == 2


def test_sequential_unused_decrements_and_hits_zero() -> None:
    bodies = [
        _chat_body(messages=[_msg("user", "one")]),
        _chat_body(messages=[_msg("user", "two")]),
    ]
    cassette = Cassette(
        name="unused-dec",
        interactions=[
            Interaction(method="POST", url=URL, request_body=bodies[0], response_body={"n": 1}),
            Interaction(method="POST", url=URL, request_body=bodies[1], response_body={"n": 2}),
        ],
    )
    transport = VCRTransport(cassette, record_mode=False, sequential=True)
    client = httpx.Client(transport=transport)
    assert transport.unused() == 2
    assert client.post(URL, json=bodies[0]).json() == {"n": 1}
    assert transport.unused() == 1
    assert client.post(URL, json=bodies[1]).json() == {"n": 2}
    assert transport.unused() == 0
    client.close()


def test_sequential_unused_empty_cassette_is_zero() -> None:
    transport = VCRTransport(Cassette(name="empty"), record_mode=False, sequential=True)
    assert transport.unused() == 0


def test_non_sequential_unused_tracks_used_set() -> None:
    body = _chat_body()
    cassette = Cassette(
        name="nonseq",
        interactions=[
            Interaction(method="POST", url=URL, request_body=body, response_body={"ok": True}),
            Interaction(
                method="POST",
                url=URL,
                request_body=_chat_body(messages=[_msg("user", "other")]),
                response_body={"ok": False},
            ),
        ],
    )
    transport = VCRTransport(cassette, record_mode=False, sequential=False)
    client = httpx.Client(transport=transport)
    assert transport.unused() == 2
    client.post(URL, json=body)
    assert transport.unused() == 1
    client.close()


def test_sequential_exhaustion_raises() -> None:
    body = _chat_body()
    cassette = Cassette(
        name="exhaust",
        interactions=[
            Interaction(method="POST", url=URL, request_body=body, response_body={}),
        ],
    )
    transport = VCRTransport(cassette, record_mode=False, sequential=True)
    client = httpx.Client(transport=transport)
    client.post(URL, json=body)
    with pytest.raises(httpx.RequestError, match="No unused cassette interactions left"):
        client.post(URL, json=body)
    client.close()
