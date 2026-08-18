# Contributing

```bash
pip install -e ".[dev]"
pytest tests/ -v
LLM_VCR_RECORD=true pytest tests/ --llm-vcr-record  # record new cassettes
```
