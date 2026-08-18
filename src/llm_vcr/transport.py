"""httpx transport that records or replays LLM API calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llm_vcr.cassette import Cassette, Interaction, find_interaction, record_interaction


class VCRTransport(httpx.BaseTransport):
    """Record or replay HTTP exchanges via a cassette."""

    def __init__(self, cassette: Cassette, record_mode: bool, wrapped: httpx.BaseTransport | None = None) -> None:
        self.cassette = cassette
        self.record_mode = record_mode
        self.wrapped = wrapped or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body_dict = _parse_json_body(request)
        if not self.record_mode:
            interaction = find_interaction(
                self.cassette, request.method, str(request.url), body_dict
            )
            if interaction is None:
                raise httpx.RequestError(
                    f"No cassette match for {request.method} {request.url}"
                )
            content = json.dumps(interaction.response_body or {}).encode()
            return httpx.Response(
                status_code=interaction.status_code,
                headers=interaction.response_headers,
                content=content,
                request=request,
            )

        response = self.wrapped.handle_request(request)
        resp_body = _parse_json_response(response)
        record_interaction(
            self.cassette,
            Interaction(
                method=request.method,
                url=str(request.url),
                request_headers=dict(request.headers),
                request_body=body_dict,
                status_code=response.status_code,
                response_headers=dict(response.headers),
                response_body=resp_body,
            ),
        )
        return response


def _parse_json_body(request: httpx.Request) -> dict[str, Any] | None:
    if not request.content:
        return None
    try:
        parsed = json.loads(request.content)
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    except json.JSONDecodeError:
        return None


def _parse_json_response(response: httpx.Response) -> dict[str, Any] | None:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    except Exception:
        return None
