"""Redact secrets from cassette payloads."""

from typing import Any

SECRET_KEYS = {
    "authorization",
    "api_key",
    "api-key",
    "x-api-key",
    "openai-api-key",
}


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SECRET_KEYS or "secret" in key.lower() or "token" in key.lower():
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict(value)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            result[key] = value
    return result
