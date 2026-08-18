"""Tests for secret redaction."""

from llm_vcr.redaction import redact_dict


def test_redacts_api_keys() -> None:
    data = {"api_key": "sk-secret", "model": "gpt-4o-mini"}
    redacted = redact_dict(data)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["model"] == "gpt-4o-mini"
