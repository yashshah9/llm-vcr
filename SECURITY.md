# Security Policy

## Reporting a vulnerability

Email **yash376351@gmail.com** with the repo name, a short description, and steps to reproduce. Please do not open a public issue for exploitable findings until we have had a reasonable chance to respond.

## Threat model (honest)

llm-vcr records and replays HTTP traffic so tests can run without live API keys.

- Cassettes may still contain **prompt text or model output** — treat the cassette directory as sensitive and review before committing.
- Redaction strips common secret field names; it is **not** a guarantee that every secret is removed.
- Record mode sends real requests with whatever credentials your process already has.
- Replay mode should not need network access, but it does not sandbox your test process.
