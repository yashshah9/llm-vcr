"""httpx transports that record or replay LLM API calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llm_vcr.cassette import Cassette, Interaction, find_interaction, record_interaction, request_key
from llm_vcr.streaming import join_chunks, parse_sse


def _is_streaming(request: httpx.Request, body: dict[str, Any] | None) -> bool:
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return True
    return bool(body and body.get("stream") is True)


class VCRTransport(httpx.BaseTransport):
    """Record or replay HTTP exchanges via a cassette."""

    def __init__(
        self,
        cassette: Cassette,
        record_mode: bool,
        wrapped: httpx.BaseTransport | None = None,
        sequential: bool = False,
    ) -> None:
        self.cassette = cassette
        self.record_mode = record_mode
        self.wrapped = wrapped or httpx.HTTPTransport()
        self.sequential = sequential
        self.index = 0
        self._used: set[int] = set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body_dict = _parse_json_body(request)
        streaming = _is_streaming(request, body_dict)
        if not self.record_mode:
            interaction = self._lookup(request.method, str(request.url), body_dict)
            if interaction is None:
                raise httpx.RequestError(
                    f"No cassette match for {request.method} {request.url}"
                )
            return self._replay(request, interaction, streaming)

        response = self.wrapped.handle_request(request)
        chunks: list[str] = []
        resp_body: dict[str, Any] | None
        if streaming:
            raw = response.read().decode(errors="replace")
            chunks = parse_sse(raw)
            resp_body = None
            # Rebuild so callers can still iterate the body.
            response = httpx.Response(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=raw.encode(),
                request=request,
            )
        else:
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
                streaming=streaming,
                chunks=chunks,
            ),
        )
        return response

    def unused(self) -> int:
        if self.sequential:
            return max(0, len(self.cassette.interactions) - self.index)
        return max(0, len(self.cassette.interactions) - len(self._used))

    def _lookup(self, method: str, url: str, body: dict[str, Any] | None) -> Interaction | None:
        if not self.sequential:
            key = request_key(method, url, body)
            for i, interaction in enumerate(self.cassette.interactions):
                if i in self._used:
                    continue
                if request_key(interaction.method, interaction.url, interaction.request_body) == key:
                    self._used.add(i)
                    return interaction
            return find_interaction(self.cassette, method, url, body)
        if self.index >= len(self.cassette.interactions):
            raise httpx.RequestError("No unused cassette interactions left.")
        expected = self.cassette.interactions[self.index]
        got = request_key(method, url, body)
        want = request_key(expected.method, expected.url, expected.request_body)
        if got != want:
            raise httpx.RequestError(
                f"Out-of-order cassette request at step {self.index}: "
                f"expected {expected.method} {expected.url}, got {method} {url}"
            )
        self.index += 1
        return expected

    def _replay(
        self, request: httpx.Request, interaction: Interaction, streaming: bool
    ) -> httpx.Response:
        if interaction.streaming or streaming:
            if not interaction.chunks:
                raise httpx.RequestError("Cassette is missing streaming chunks.")
            content = join_chunks(interaction.chunks)
            headers = dict(interaction.response_headers)
            headers.setdefault("content-type", "text/event-stream")
            return httpx.Response(
                status_code=interaction.status_code,
                headers=headers,
                content=content,
                request=request,
            )
        content = json.dumps(interaction.response_body or {}).encode()
        return httpx.Response(
            status_code=interaction.status_code,
            headers=interaction.response_headers,
            content=content,
            request=request,
        )


class AsyncVCRTransport(httpx.AsyncBaseTransport):
    """Async counterpart of VCRTransport."""

    def __init__(
        self,
        cassette: Cassette,
        record_mode: bool,
        wrapped: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.sync = VCRTransport(cassette, record_mode, sequential=False)
        self.wrapped = wrapped or httpx.AsyncHTTPTransport()
        self.record_mode = record_mode
        self.cassette = cassette

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not self.record_mode:
            return self.sync.handle_request(request)
        body_dict = _parse_json_body(request)
        streaming = _is_streaming(request, body_dict)
        response = await self.wrapped.handle_async_request(request)
        chunks: list[str] = []
        resp_body: dict[str, Any] | None
        if streaming:
            raw_bytes = await response.aread()
            raw = raw_bytes.decode(errors="replace")
            chunks = parse_sse(raw)
            resp_body = None
            response = httpx.Response(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=raw_bytes,
                request=request,
            )
        else:
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
                streaming=streaming,
                chunks=chunks,
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
