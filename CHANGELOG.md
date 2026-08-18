# Changelog

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
