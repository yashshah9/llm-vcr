"""pytest plugin and @llm_vcr decorator."""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Generator
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest

from llm_vcr.cassette import Cassette
from llm_vcr.matching import MatcherName
from llm_vcr.transport import AsyncVCRTransport, VCRTransport

F = TypeVar("F", bound=Callable[..., Any])


def _cassette_dir() -> Path:
    return Path(os.environ.get("LLM_VCR_CASSETTE_DIR", "tests/cassettes"))


def _recording() -> bool:
    return os.environ.get("LLM_VCR_RECORD", "false").lower() in {"1", "true", "yes"}


def _bind_client(func: Callable[..., Any], kwargs: dict[str, Any], client: httpx.Client) -> None:
    params = inspect.signature(func).parameters
    if "http_client" in params and "http_client" not in kwargs:
        kwargs["http_client"] = client
    elif "client" in params and "client" not in kwargs:
        kwargs["client"] = client


def llm_vcr(
    name: str | None = None,
    sequential: bool = False,
    matcher: MatcherName = "exact",
) -> Callable[[F], F]:
    """Decorator to enable cassette record/replay for a test function."""

    def decorator(func: F) -> F:
        cassette_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
            record = _recording() or not cassette_path.exists()

            if record:
                cassette = Cassette(name=cassette_name)
            else:
                cassette = Cassette.load(cassette_path)

            transport = VCRTransport(
                cassette, record_mode=record, sequential=sequential, matcher=matcher
            )
            client = httpx.Client(transport=transport)

            try:
                _bind_client(func, kwargs, client)
                result = func(*args, **kwargs)
            finally:
                client.close()
                if sequential and not record and transport.unused():
                    warnings.warn(
                        f"cassette {cassette_name} has {transport.unused()} unused interaction(s)",
                        stacklevel=3,
                    )
                if record:
                    cassette.save(cassette_path)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _fixture_opts(request: pytest.FixtureRequest) -> tuple[str, bool, MatcherName]:
    """Resolve cassette name / sequential / matcher from param or @pytest.mark.llm_vcr."""
    param = getattr(request, "param", None)
    name: str
    sequential = False
    matcher: MatcherName = "exact"
    if isinstance(param, dict):
        name = str(param.get("name", request.node.name))
        sequential = bool(param.get("sequential", False))
        matcher = param.get("matcher", "exact")  # type: ignore[assignment]
    elif param is not None:
        name = str(param)
    else:
        name = request.node.name
    marker = request.node.get_closest_marker("llm_vcr")
    if marker is not None:
        if marker.kwargs.get("name") is not None:
            name = str(marker.kwargs["name"])
        sequential = bool(marker.kwargs.get("sequential", sequential))
        matcher = marker.kwargs.get("matcher", matcher)  # type: ignore[assignment]
    return name, sequential, matcher


@pytest.fixture
def llm_vcr_client(request: pytest.FixtureRequest) -> Generator[httpx.Client, None, None]:
    """pytest fixture that yields an httpx client bound to a named cassette.

    Configure via ``@pytest.mark.llm_vcr(sequential=True, matcher="semantic")``
    or ``@pytest.mark.parametrize(..., indirect=True)`` with a dict param.
    """
    cassette_name, sequential, matcher = _fixture_opts(request)
    cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
    record = _recording() or not cassette_path.exists()
    cassette = Cassette(name=str(cassette_name)) if record else Cassette.load(cassette_path)
    transport = VCRTransport(
        cassette, record_mode=record, sequential=sequential, matcher=matcher
    )
    client = httpx.Client(transport=transport)
    yield client
    client.close()
    if sequential and not record and transport.unused():
        warnings.warn(
            f"cassette {cassette_name} has {transport.unused()} unused interaction(s)",
            stacklevel=2,
        )
    if record:
        cassette.save(cassette_path)


@pytest.fixture
async def llm_vcr_async_client(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    cassette_name, sequential, matcher = _fixture_opts(request)
    cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
    record = _recording() or not cassette_path.exists()
    cassette = Cassette(name=str(cassette_name)) if record else Cassette.load(cassette_path)
    transport = AsyncVCRTransport(
        cassette, record_mode=record, sequential=sequential, matcher=matcher
    )
    client = httpx.AsyncClient(transport=transport)
    yield client
    await client.aclose()
    if sequential and not record and transport.sync.unused():
        warnings.warn(
            f"cassette {cassette_name} has {transport.sync.unused()} unused interaction(s)",
            stacklevel=2,
        )
    if record:
        cassette.save(cassette_path)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--llm-vcr-record", action="store_true", help="Record new cassettes")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--llm-vcr-record"):
        os.environ["LLM_VCR_RECORD"] = "true"
    config.addinivalue_line(
        "markers",
        "llm_vcr(name=None, sequential=False, matcher='exact'): "
        "configure llm_vcr_client / llm_vcr_async_client fixtures",
    )
