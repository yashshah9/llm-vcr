# llm-vcr

Record and replay LLM HTTP traffic for **deterministic, key-free pytest runs**.

> **Status:** v0.2 — SSE streaming replay, volatile-field matching, and async transport. Multi-step tool-call loops are next.

## Problem

Testing code that calls LLMs is slow, flaky, and expensive. Hand-written mocks drift from reality. Generic HTTP cassettes (VCR.py) don't understand LLM request shapes or redact API keys well.

## Key features (v0.2)

- **pytest plugin** — `@llm_vcr` decorator, `llm_vcr_client` fixture, `--llm-vcr-record`
- **httpx transport** — sync + async, including `client.stream(...)`
- **YAML cassettes** — `streaming: true` + `chunks` for SSE
- **Matching** — drops volatile fields (`user`, `request_id`, timestamps, `seed`)
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

- [x] SSE streaming chunk replay
- [ ] Tool-call multi-step loops
- [x] Async httpx transport (fixture + replay)
- [ ] OpenAI + Anthropic semantic request matching

## Known limitations (v0.2)

- Tool-call multi-step ordered cassettes are not implemented
- Matching drops a small set of volatile keys; not full semantic/alias matching
- Record mode for streaming stores chunks, not per-event timestamps

## License

MIT
