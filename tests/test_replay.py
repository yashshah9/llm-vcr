"""Tests for cassette replay."""

import json
from pathlib import Path

import httpx

from llm_vcr.cassette import Cassette, find_interaction
from llm_vcr.transport import VCRTransport

CASSETTE = Path(__file__).parent / "cassettes" / "test_replay.yaml"


def test_replay_returns_cassette_response() -> None:
    cassette = Cassette.load(CASSETTE)
    transport = VCRTransport(cassette, record_mode=False)
    client = httpx.Client(transport=transport)

    interaction = cassette.interactions[0]
    response = client.post(
        interaction.url,
        json=interaction.request_body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    client.close()


def test_find_interaction_matches() -> None:
    cassette = Cassette.load(CASSETTE)
    i = cassette.interactions[0]
    found = find_interaction(cassette, i.method, i.url, i.request_body)
    assert found is not None
    assert found.response_body["id"] == "chatcmpl-test"
