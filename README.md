# llm-vcr

Record and replay LLM HTTP traffic for **deterministic, key-free pytest runs**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/yashshah9/llm-vcr/actions/workflows/ci.yml/badge.svg)](https://github.com/yashshah9/llm-vcr/actions/workflows/ci.yml)

> **Status:** v0.4 — SSE streaming replay, sequential tool-call cassettes, model-date normalize, and `llm-vcr diff`.

## 60-second try

```bash
docker compose run --rm health  # llm-vcr health
docker compose run --rm test    # pytest (replay, no API key)
```

## Why this vs alternatives

| Approach | Strength | Gap |
|----------|----------|-----|
| **llm-vcr** | LLM-aware matching, redaction, SSE + sequential cassettes | httpx-focused today |
| VCR.py / pytest-recording | Mature generic HTTP cassettes | No model-date normalize or LLM-shaped diffs |
| Hand-written mocks | Fast, no network | Drift from live provider payloads |
| Live API in CI | Highest fidelity | Flaky, keyed, and expensive |

## Problem

Testing code that calls LLMs is slow, flaky, and expensive. Hand-written mocks drift from reality. Generic HTTP cassettes (VCR.py) don't understand LLM request shapes or redact API keys well.

## Key features (v0.4)

- **pytest plugin** — `@llm_vcr` decorator, `llm_vcr_client` fixture, `--llm-vcr-record`
- **httpx transport** — sync + async, including `client.stream(...)`
- **YAML cassettes** — `streaming: true` + `chunks` for SSE
- **Matching** — drops volatile fields; strips dated model suffixes (`gpt-4o-mini-2024-07-18` → `gpt-4o-mini`)
- **`llm-vcr diff`** — show normalized differences between JSON bodies or cassette interactions
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
llm-vcr diff left.json right.json
```

## Docker

```bash
docker compose run --rm test
docker compose run --rm health
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

### Diff normalized request bodies

```bash
llm-vcr diff left.json right.json
llm-vcr diff cassette_a.yaml cassette_b.yaml --index 0
```

Model strings with a trailing `-YYYY-MM-DD` normalize equal so dated aliases do not show as diffs.

## Running tests

```bash
pytest tests/ -v
```

## Roadmap

- [x] SSE streaming chunk replay
- [x] Tool-call multi-step loops (`@llm_vcr(..., sequential=True)`)
- [x] Async httpx transport (fixture + replay)
- [x] Model-date normalize + `llm-vcr diff`
- [ ] OpenAI + Anthropic semantic request matching

## Known limitations (v0.4)

- Sequential matching is opt-in (`sequential=True`); default matching is still hash-based
- Matching drops a small set of volatile keys; not full semantic/alias matching
- Record mode for streaming stores chunks, not per-event timestamps

## License

MIT
