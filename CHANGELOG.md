# Changelog

## [0.6.0] - 2026-09-14

### Added
- Anthropic Messages API matching — `providers/anthropic.py` normalizes bodies (drop `metadata`, model date-strip, messages/system/tools; strip `tool_use` ids under `matcher="semantic"`)
- Anthropic URL normalize — `api.anthropic.com` query/fragment stripped for cassette matches
- Compact model date-strip — `-YYYYMMDD` suffixes (Anthropic-style) via `normalize_model`
- Cassette fixture + unit tests for Anthropic record/replay

## [0.5.0] - 2026-09-14

### Added
- Semantic request matcher — compare messages by role+content (ignore tool_call ids), tools by function name, model date-strip + aliases
- `@llm_vcr(matcher="semantic")` and `VCRTransport(..., matcher="semantic"|"exact")` (`exact` remains default)
- `llm-vcr diff` reports when bodies are semantically equal after normalization

## [0.4.0] - 2026-09-14

### Added
- Model-date normalize — strip trailing `-YYYY-MM-DD` from `model` during cassette matching
- `llm-vcr diff` CLI — show normalized differences between JSON bodies or cassette interactions

## [0.3.0] - 2026-08-19

### Added
- Sequential cassette replay for multi-step tool-call loops (`sequential=True`)
- Out-of-order requests raise; unused interactions are countable

## [0.2.0] - 2026-08-19

### Added
- SSE streaming record/replay (`streaming` + `chunks` on cassettes)
- Volatile-field request matching (`user`, `request_id`, timestamps, `seed`)
- `AsyncVCRTransport` and `llm_vcr_client` / `llm_vcr_async_client` fixtures
- stdlib `llm-vcr health` CLI (no extra Click dependency)

## [0.1.0] - 2026-08-18

### Added
- Initial httpx VCR transport with YAML cassettes
- pytest plugin entry point and `@llm_vcr` decorator
- Secret redaction for cassette payloads
- Example cassette and replay tests
