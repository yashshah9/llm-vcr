# Contributing

## Running tests

Prefer Docker Compose:

```bash
docker compose run --rm test
docker compose run --rm health
```

Locally:

```bash
pip install -e ".[dev]"
pytest tests/ -v
LLM_VCR_RECORD=true pytest tests/ --llm-vcr-record  # record new cassettes
llm-vcr health
```

## Pull requests

- Keep cassette fixtures minimal; review for secrets before committing
- Update README/CHANGELOG for matching or CLI changes
- Prefer small PRs with a clear test plan

## Commit style

- Imperative subject line; mention the user-facing why when relevant
- Do not add AI co-author trailers (e.g. Co-authored-by: Cursor) to commits.
