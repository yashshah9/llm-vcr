"""Cassette storage format and I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_vcr.matching import normalize_body
from llm_vcr.redaction import redact_dict


@dataclass
class Interaction:
    """One recorded HTTP exchange."""

    method: str
    url: str
    request_headers: dict[str, str] = field(default_factory=dict)
    request_body: dict[str, Any] | None = None
    status_code: int = 200
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: dict[str, Any] | None = None
    streaming: bool = False
    chunks: list[str] = field(default_factory=list)


@dataclass
class Cassette:
    """Collection of interactions for one test module."""

    name: str
    interactions: list[Interaction] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "interactions": [
                {
                    "method": i.method,
                    "url": i.url,
                    "request_headers": i.request_headers,
                    "request_body": i.request_body,
                    "status_code": i.status_code,
                    "response_headers": i.response_headers,
                    "response_body": i.response_body,
                    "streaming": i.streaming,
                    "chunks": i.chunks,
                }
                for i in self.interactions
            ],
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Cassette:
        if not path.exists():
            raise FileNotFoundError(f"Cassette not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Cassette {path} is not a YAML mapping")
        interactions = [
            Interaction(
                method=i["method"],
                url=i["url"],
                request_headers=i.get("request_headers", {}),
                request_body=i.get("request_body"),
                status_code=i.get("status_code", 200),
                response_headers=i.get("response_headers", {}),
                response_body=i.get("response_body"),
                streaming=bool(i.get("streaming", False)),
                chunks=list(i.get("chunks") or []),
            )
            for i in raw.get("interactions", [])
        ]
        return cls(name=raw.get("name", path.stem), interactions=interactions)


def request_key(method: str, url: str, body: dict[str, Any] | None) -> str:
    """Stable hash for matching requests (volatile fields stripped)."""
    payload = json.dumps(
        {"method": method, "url": url, "body": normalize_body(body)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def find_interaction(cassette: Cassette, method: str, url: str, body: dict[str, Any] | None) -> Interaction | None:
    key = request_key(method, url, body)
    for interaction in cassette.interactions:
        if request_key(interaction.method, interaction.url, interaction.request_body) == key:
            return interaction
    return None


def record_interaction(cassette: Cassette, interaction: Interaction) -> None:
    """Append interaction with redacted secrets."""
    if interaction.request_body:
        interaction.request_body = redact_dict(interaction.request_body)
    if interaction.response_body:
        interaction.response_body = redact_dict(interaction.response_body)
    cassette.interactions.append(interaction)
