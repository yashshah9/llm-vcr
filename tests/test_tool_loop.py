"""Sequential multi-step tool-call cassette replay."""

from pathlib import Path

import httpx
import pytest

from llm_vcr.cassette import Cassette
from llm_vcr.transport import VCRTransport

CASSETTE = Path(__file__).parent / "cassettes" / "test_tool_loop.yaml"
URL = "https://api.openai.com/v1/chat/completions"


def _client(sequential: bool = True) -> httpx.Client:
    cassette = Cassette.load(CASSETTE)
    return httpx.Client(transport=VCRTransport(cassette, record_mode=False, sequential=sequential))


def test_three_step_tool_loop_replays_in_order() -> None:
    client = _client()
    first = client.post(
        URL,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
    )
    assert first.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    second = client.post(
        URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "tool", "content": "22C"},
            ],
        },
    )
    assert "22C" in second.json()["choices"][0]["message"]["content"]
    client.close()


def test_out_of_order_sequential_raises() -> None:
    client = _client()
    with pytest.raises(httpx.RequestError, match="Out-of-order"):
        client.post(
            URL,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "What's the weather in Paris?"},
                    {"role": "tool", "content": "22C"},
                ],
            },
        )
    client.close()


def test_unused_interactions_counted() -> None:
    cassette = Cassette.load(CASSETTE)
    transport = VCRTransport(cassette, record_mode=False, sequential=True)
    client = httpx.Client(transport=transport)
    client.post(
        URL,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
    )
    assert transport.unused() == 1
    client.close()
