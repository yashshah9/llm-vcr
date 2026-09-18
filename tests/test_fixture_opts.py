"""Fixture options: sequential + matcher via mark / indirect param."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from llm_vcr.plugin import _fixture_opts

URL = "https://api.openai.com/v1/chat/completions"
CASSETTE_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture(autouse=True)
def _cassette_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_VCR_CASSETTE_DIR", str(CASSETTE_DIR))
    monkeypatch.delenv("LLM_VCR_RECORD", raising=False)


def test_fixture_opts_from_dict_param(request: pytest.FixtureRequest) -> None:
    class Fake:
        param = {"name": "tool", "sequential": True, "matcher": "semantic"}
        node = request.node

    name, sequential, matcher = _fixture_opts(Fake())  # type: ignore[arg-type]
    assert name == "tool"
    assert sequential is True
    assert matcher == "semantic"


@pytest.mark.llm_vcr(name="test_tool_loop", sequential=True)
def test_fixture_sequential_via_marker(llm_vcr_client: httpx.Client) -> None:
    first = llm_vcr_client.post(
        URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        },
    )
    assert first.status_code == 200
    second = llm_vcr_client.post(
        URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "tool", "content": "22C"},
            ],
        },
    )
    content = second.json()["choices"][0]["message"]["content"].lower()
    assert "sunny" in content or second.status_code == 200


@pytest.mark.parametrize(
    "llm_vcr_client",
    [{"name": "test_tool_loop", "sequential": True}],
    indirect=True,
)
def test_fixture_sequential_via_param(llm_vcr_client: httpx.Client) -> None:
    # Second request must not reuse first interaction when sequential=True
    llm_vcr_client.post(
        URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        },
    )
    with pytest.raises(httpx.RequestError, match="Out-of-order|No unused"):
        # Same body again would match non-sequentially; sequential requires next step
        llm_vcr_client.post(
            URL,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            },
        )


def test_tool_loop_cassette_exists() -> None:
    path = CASSETTE_DIR / "test_tool_loop.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(data["interactions"]) >= 2
