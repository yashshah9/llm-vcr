# Publishing to PyPI

API-token upload is currently blocked (old token 403). Prefer **Trusted Publishing** (OIDC).

## One-time setup (per package on pypi.org)

For each project (`duckcheck`, `embedsync`, `agentbox-sandbox`, `mcp-tool-evals`, `pytest-llm-vcr`):

1. PyPI → Project → Publishing → **Add a new pending publisher**
2. Owner: `yashshah9`
3. Repo: matching GitHub repo name
4. Workflow: `publish.yml`
5. Environment: `pypi`

Also create a GitHub Environment named `pypi` on each repo (Settings → Environments).

## Publish a release

```bash
gh release create vX.Y.Z --generate-notes
# or: Actions → "Publish to PyPI" → Run workflow
```

## Manual token upload (fallback)

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # fresh token
cd <repo> && python -m build && twine upload dist/*
```
