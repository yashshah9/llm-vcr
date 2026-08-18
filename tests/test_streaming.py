"""Tests for SSE replay and semantic matching."""

from pathlib import Path

import httpx
import pytest

from llm_vcr.cassette import Cassette, Interaction, find_interaction
from llm_vcr.matching import normalize_body
from llm_vcr.streaming import parse_sse
from llm_vcr.transport import VCRTransport

CASSETTE = Path(__file__).parent / "cassettes" / "test_streaming.yaml"


def test_parse_sse_splits_events() -> None:
    raw = "data: a\n\ndata: [DONE]\n\n"
    chunks = parse_sse(raw)
    assert len(chunks) == 2
    assert chunks[-1].strip().endswith("[DONE]")


def test_streaming_replay_yields_chunks() -> None:
    cassette = Cassette.load(CASSETTE)
    client = httpx.Client(transport=VCRTransport(cassette, record_mode=False))
    with client.stream(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    ) as response:
        text = "".join(line.decode() if isinstance(line, bytes) else line for line in response.iter_bytes())
    assert "Hello" in text
    assert "[DONE]" in text
    client.close()


def test_semantic_match_ignores_user_field() -> None:
    cassette = Cassette.load(CASSETTE)
    body = {
        "model": "gpt-4o-mini",
        "stream": True,
        "user": "volatile-id",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    found = find_interaction(
        cassette,
        "POST",
        "https://api.openai.com/v1/chat/completions",
        body,
    )
    assert found is not None
    assert "user" not in normalize_body(body)


def test_missing_chunks_raises() -> None:
    cassette = Cassette(
        name="empty-stream",
        interactions=[
            Interaction(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                request_body={"model": "gpt-4o-mini", "stream": True},
                streaming=True,
                chunks=[],
            )
        ],
    )
    client = httpx.Client(transport=VCRTransport(cassette, record_mode=False))
    with pytest.raises(httpx.RequestError, match="missing streaming chunks"):
        client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini", "stream": True},
        )
    client.close()
