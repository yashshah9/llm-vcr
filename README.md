# llm-vcr

Record and replay LLM HTTP traffic for **deterministic, key-free pytest runs**.

> **Status:** v0.1 foundation — httpx transport replay works; streaming SSE and tool-call loops are next.

## Problem

Testing code that calls LLMs is slow, flaky, and expensive. Hand-written mocks drift from reality. Generic HTTP cassettes (VCR.py) don't understand LLM request shapes or redact API keys well.

## Key features (v0.1)

- **pytest plugin** — `@llm_vcr` decorator or `--llm-vcr-record` flag
- **httpx transport** — intercept at HTTP layer, works with any client using httpx
- **YAML cassettes** — human-readable, committable fixtures
- **Automatic redaction** — strips api_key, token, authorization fields

## Architecture

```
@pytest test
    └── @llm_vcr decorator
            └── VCRTransport (httpx)
                    ├── replay mode → read cassette YAML
                    └── record mode → live HTTP + save cassette
```

| Component | Technology | Why |
|-----------|------------|-----|
| HTTP | httpx | Modern, sync+async, transport hooks |
| Cassettes | YAML | Readable diffs in PRs |
| Tests | pytest entry point | Zero-config discovery |

## Installation

```bash
pip install llm-vcr
pip install -e ".[dev]"  # from source
```

## Local development

```bash
pip install -e ".[dev]"
pytest tests/ -v
llm-vcr health
```

## Docker

```bash
docker compose run --rm test
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_VCR_CASSETTE_DIR` | `tests/cassettes` | Cassette storage directory |
| `LLM_VCR_RECORD` | `false` | Force record mode |

## Usage

### Replay (CI — no API key needed)

```python
import httpx
from llm_vcr.plugin import llm_vcr

@llm_vcr("my_test")
def test_chat(client: httpx.Client) -> None:
    resp = client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say hello"}]},
    )
    assert resp.json()["choices"][0]["message"]["content"]
```

### Record new cassettes

```bash
LLM_VCR_RECORD=true pytest tests/ --llm-vcr-record
```

## Running tests

```bash
pytest tests/ -v
```

## Roadmap

- [ ] SSE streaming chunk replay
- [ ] Tool-call multi-step loops
- [ ] Async httpx support
- [ ] OpenAI + Anthropic semantic request matching

## License

MIT

## Known limitations (v0.1)

- Sync httpx only
- Exact JSON body matching (no semantic normalization yet)
- No streaming replay
- Decorator injects `client` kwarg — fixture API coming
