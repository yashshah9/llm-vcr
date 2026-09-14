"""Tests for request body normalization and diff."""

from llm_vcr.matching import diff_bodies, normalize_body, normalize_model


def test_normalize_model_strips_date_suffix() -> None:
    assert normalize_model("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert normalize_model("gpt-4o-mini") == "gpt-4o-mini"


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
