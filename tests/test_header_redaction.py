"""Regression: Authorization headers must be redacted in cassettes."""

from __future__ import annotations

from llm_vcr.cassette import Cassette, Interaction, record_interaction


def test_record_interaction_redacts_auth_headers() -> None:
    cassette = Cassette(name="sec")
    record_interaction(
        cassette,
        Interaction(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            request_headers={
                "Authorization": "Bearer sk-live-secret",
                "content-type": "application/json",
            },
            request_body={"model": "gpt-4o-mini"},
            status_code=200,
            response_headers={"x-api-key": "sk-resp"},
            response_body={"ok": True},
        ),
    )
    got = cassette.interactions[0]
    assert got.request_headers["Authorization"] == "[REDACTED]"
    assert got.request_headers["content-type"] == "application/json"
    assert got.response_headers["x-api-key"] == "[REDACTED]"
